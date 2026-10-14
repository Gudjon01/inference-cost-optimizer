package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"regexp"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/config"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/export"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/mockengine"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/replica"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/telemetrypb"
)

// capture is a Sink that hands records to the test.
type capture struct {
	records chan *telemetrypb.RequestRecord
}

func (c *capture) Emit(r *telemetrypb.RequestRecord) { c.records <- r }

func (c *capture) next(t *testing.T) *telemetrypb.RequestRecord {
	t.Helper()
	select {
	case r := <-c.records:
		return r
	case <-time.After(5 * time.Second):
		t.Fatal("no record was emitted")
		return nil
	}
}

// seen remembers what reached an engine.
type seen struct {
	mu      sync.Mutex
	headers []http.Header
	bodies  [][]byte
}

func (s *seen) record(r *http.Request, body []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.headers = append(s.headers, r.Header.Clone())
	s.bodies = append(s.bodies, body)
}

func (s *seen) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.bodies)
}

type fixture struct {
	gateway *httptest.Server
	pool    *replica.Pool
	sink    *capture
	seen    []*seen
}

// newFixture starts one mock engine per entry and a gateway in front of them.
// Replicas are named r1, r2, ... in order.
func newFixture(t *testing.T, engines ...*mockengine.Engine) *fixture {
	t.Helper()
	f := &fixture{sink: &capture{records: make(chan *telemetrypb.RequestRecord, 1024)}}
	var cfg []config.Replica
	for i, e := range engines {
		s := &seen{}
		e.Seen = s.record
		srv := httptest.NewServer(e.Handler())
		t.Cleanup(srv.Close)
		f.seen = append(f.seen, s)
		cfg = append(cfg, config.Replica{Name: "r" + string(rune('1'+i)), URL: srv.URL})
	}
	f.start(t, cfg)
	return f
}

func (f *fixture) start(t *testing.T, cfg []config.Replica) {
	t.Helper()
	pool, err := replica.NewPool(cfg)
	if err != nil {
		t.Fatal(err)
	}
	f.pool = pool
	f.gateway = httptest.NewServer(New(Options{
		Pool:          pool,
		Hasher:        testHasher(),
		Sink:          f.sink,
		TenantHeader:  "X-Tenant-Id",
		FeatureHeader: "X-Feature",
		MaxBodyBytes:  1 << 20,
	}))
	t.Cleanup(f.gateway.Close)
}

func (f *fixture) post(t *testing.T, path, body string, headers ...string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, f.gateway.URL+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func readAll(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// deadURL is an address nothing listens on.
func deadURL(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	return "http://" + addr
}

const chatRequest = `{"model":"mock-model","max_tokens":5,"messages":[{"role":"user","content":"hello there"}]}`

var hex32 = regexp.MustCompile(`^[0-9a-f]{32}$`)

func TestNonStreamingChat(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{})
	resp := f.post(t, "/v1/chat/completions", chatRequest, "X-Tenant-Id", "acme", "X-Feature", "search")
	body := readAll(t, resp)

	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	var parsed struct {
		Choices []struct {
			Message struct{ Content string } `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(body), &parsed); err != nil || len(parsed.Choices) != 1 || parsed.Choices[0].Message.Content == "" {
		t.Fatalf("response did not come through intact: %s", body)
	}

	rec := f.sink.next(t)
	if rec.Model != "mock-model" || rec.Tenant != "acme" || rec.Feature != "search" || rec.Endpoint != "chat" {
		t.Errorf("labels: %+v", rec)
	}
	if rec.Replica != "r1" || rec.Status != 200 || rec.Stream || rec.MaxTokens != 5 {
		t.Errorf("outcome: %+v", rec)
	}
	if rec.CompletionTokens != 5 || rec.PromptTokens == 0 {
		t.Errorf("tokens: prompt=%d completion=%d, want >0 and 5", rec.PromptTokens, rec.CompletionTokens)
	}
	if rec.E2EMs <= 0 || rec.ArrivalUnixNanos == 0 || rec.PromptChars == 0 {
		t.Errorf("timing or size missing: %+v", rec)
	}
	if !hex32.MatchString(rec.TraceId) || rec.RequestId == "" {
		t.Errorf("ids: trace=%q request=%q", rec.TraceId, rec.RequestId)
	}

	// The engine got the same trace and request ids, so its spans and logs join.
	h := f.seen[0].headers[0]
	if tp := h.Get("Traceparent"); !strings.HasPrefix(tp, "00-"+rec.TraceId+"-") {
		t.Errorf("traceparent upstream = %q, want trace %s", tp, rec.TraceId)
	}
	if h.Get("X-Request-Id") != rec.RequestId || resp.Header.Get("X-Request-Id") != rec.RequestId {
		t.Errorf("request id not shared: upstream %q, client %q, record %q", h.Get("X-Request-Id"), resp.Header.Get("X-Request-Id"), rec.RequestId)
	}
	if string(f.seen[0].bodies[0]) != chatRequest {
		t.Errorf("non-streaming body was changed on the way: %s", f.seen[0].bodies[0])
	}
}

func TestCallerTraceAndRequestIDAreKept(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{})
	readAll(t, f.post(t, "/v1/chat/completions", chatRequest,
		"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
		"X-Request-Id", "caller-id-1"))
	rec := f.sink.next(t)
	if rec.TraceId != "4bf92f3577b34da6a3ce929d0e0e4736" || rec.RequestId != "caller-id-1" {
		t.Errorf("trace=%q request=%q", rec.TraceId, rec.RequestId)
	}
}

func TestStreamingHidesTheUsageChunkItAskedFor(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{TTFT: 20 * time.Millisecond, TokenDelay: 5 * time.Millisecond})
	resp := f.post(t, "/v1/chat/completions", `{"model":"mock-model","stream":true,"max_tokens":6,"messages":[{"role":"user","content":"hi"}]}`)
	body := readAll(t, resp)

	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("content type %q", ct)
	}
	if strings.Contains(body, `"usage"`) {
		t.Errorf("client did not ask for usage but got it:\n%s", body)
	}
	if got := strings.Count(body, `"content":" tok"`); got != 6 {
		t.Errorf("client saw %d content chunks, want 6:\n%s", got, body)
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Errorf("stream does not end with [DONE]:\n%s", body)
	}
	if !strings.Contains(string(f.seen[0].bodies[0]), `"include_usage":true`) {
		t.Errorf("engine was not asked for usage: %s", f.seen[0].bodies[0])
	}

	rec := f.sink.next(t)
	if !rec.Stream || rec.Status != 200 || rec.CompletionTokens != 6 || rec.PromptTokens == 0 {
		t.Errorf("record: %+v", rec)
	}
	if rec.TtftMs < 20 || rec.TtftMs > rec.E2EMs {
		t.Errorf("ttft %.1f ms with e2e %.1f ms; the engine waits 20 ms before its first token", rec.TtftMs, rec.E2EMs)
	}
	if rec.GapP50Ms < 4 || rec.GapP95Ms < rec.GapP50Ms {
		t.Errorf("gaps p50=%.1f p95=%.1f; the engine spaces tokens 5 ms apart", rec.GapP50Ms, rec.GapP95Ms)
	}
}

func TestStreamingKeepsUsageTheClientAskedFor(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{})
	request := `{"model":"mock-model","stream":true,"stream_options":{"include_usage":true},"max_tokens":3,"prompt":"hi"}`
	body := readAll(t, f.post(t, "/v1/completions", request))
	if !strings.Contains(body, `"usage"`) {
		t.Errorf("client asked for usage and did not get it:\n%s", body)
	}
	if string(f.seen[0].bodies[0]) != request {
		t.Errorf("body was changed: %s", f.seen[0].bodies[0])
	}
	rec := f.sink.next(t)
	if rec.Endpoint != "completions" || rec.CompletionTokens != 3 || rec.TtftMs <= 0 {
		t.Errorf("record: %+v", rec)
	}
}

func TestStreamIsRelayedAsItArrives(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{TokenDelay: 150 * time.Millisecond})
	resp := f.post(t, "/v1/chat/completions", `{"model":"mock-model","stream":true,"max_tokens":4,"messages":[]}`)
	defer resp.Body.Close()

	// The first token must reach the client long before the stream ends
	// (about 450 ms later): a buffering proxy would deliver everything at once.
	start := time.Now()
	lines := bufio.NewReader(resp.Body)
	for {
		line, err := lines.ReadString('\n')
		if err != nil {
			t.Fatalf("stream ended before any content: %v", err)
		}
		if strings.Contains(line, `"content":" tok"`) {
			break
		}
	}
	if waited := time.Since(start); waited > 200*time.Millisecond {
		t.Errorf("first token took %v to reach the client", waited)
	}
	_, _ = io.Copy(io.Discard, lines)
	f.sink.next(t)
}

func TestEmbeddingsRecordPromptTokens(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{})
	body := readAll(t, f.post(t, "/v1/embeddings", `{"model":"mock-model","input":"some text to embed"}`))
	if !strings.Contains(body, `"embedding"`) {
		t.Fatalf("response: %s", body)
	}
	rec := f.sink.next(t)
	if rec.Endpoint != "embeddings" || rec.PromptTokens == 0 || rec.CompletionTokens != 0 || rec.Stream {
		t.Errorf("record: %+v", rec)
	}
}

func TestModelsIsProxiedWithoutARecord(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{Model: "served-model"})
	resp, err := http.Get(f.gateway.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	if body := readAll(t, resp); !strings.Contains(body, "served-model") {
		t.Errorf("models response: %s", body)
	}
	select {
	case rec := <-f.sink.records:
		t.Errorf("unexpected record for /v1/models: %+v", rec)
	case <-time.After(50 * time.Millisecond):
	}
}

func TestEngineErrorsPassThrough(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{})
	resp := f.post(t, "/v1/chat/completions", `{"model": broken`)
	body := readAll(t, resp)
	if resp.StatusCode != 400 || !strings.Contains(body, "invalid JSON") {
		t.Errorf("status %d body %s; want the engine's own 400", resp.StatusCode, body)
	}
	if rec := f.sink.next(t); rec.Status != 400 || rec.Replica != "r1" {
		t.Errorf("record: %+v", rec)
	}
}

func TestRoundRobinSkipsDrainedReplicas(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{}, &mockengine.Engine{})
	send := func(n int) map[string]int {
		byReplica := map[string]int{}
		for i := 0; i < n; i++ {
			readAll(t, f.post(t, "/v1/chat/completions", chatRequest))
			byReplica[f.sink.next(t).Replica]++
		}
		return byReplica
	}

	if got := send(10); got["r1"] != 5 || got["r2"] != 5 {
		t.Errorf("round robin gave %v, want 5 and 5", got)
	}
	f.pool.SetDrained("r1", true)
	if got := send(10); got["r2"] != 10 {
		t.Errorf("with r1 drained got %v, want everything on r2", got)
	}
	f.pool.SetDrained("r1", false)
	if got := send(10); got["r1"] != 5 || got["r2"] != 5 {
		t.Errorf("after undrain got %v, want 5 and 5", got)
	}
}

func TestDrainWaitsOutInflightRequests(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{TokenDelay: 20 * time.Millisecond})
	resp := f.post(t, "/v1/chat/completions", `{"model":"mock-model","stream":true,"max_tokens":10,"messages":[]}`)
	r1, _ := f.pool.Get("r1")
	f.pool.SetDrained("r1", true)
	if r1.Inflight() != 1 {
		t.Errorf("inflight = %d while a stream is open, want 1", r1.Inflight())
	}
	// A request that started before the drain still finishes.
	if body := readAll(t, resp); !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Errorf("in-flight stream was cut short by the drain:\n%s", body)
	}
	f.sink.next(t)
	if r1.Inflight() != 0 {
		t.Errorf("inflight = %d after the stream ended, want 0", r1.Inflight())
	}

	resp = f.post(t, "/v1/chat/completions", chatRequest)
	body := readAll(t, resp)
	if resp.StatusCode != 503 || resp.Header.Get("Retry-After") == "" || !strings.Contains(body, "drained") {
		t.Errorf("status %d, Retry-After %q, body %s; want a 503 that says why", resp.StatusCode, resp.Header.Get("Retry-After"), body)
	}
	if rec := f.sink.next(t); rec.Status != 503 || rec.Replica != "" {
		t.Errorf("record: %+v", rec)
	}
}

func TestUnreachableReplicaFailsOver(t *testing.T) {
	engine := &mockengine.Engine{}
	live := httptest.NewServer(engine.Handler())
	t.Cleanup(live.Close)
	f := &fixture{sink: &capture{records: make(chan *telemetrypb.RequestRecord, 64)}}
	f.start(t, []config.Replica{{Name: "dead", URL: deadURL(t)}, {Name: "live", URL: live.URL}})

	for i := 0; i < 6; i++ {
		resp := f.post(t, "/v1/chat/completions", chatRequest)
		if body := readAll(t, resp); resp.StatusCode != 200 {
			t.Fatalf("request %d: status %d: %s", i, resp.StatusCode, body)
		}
		if rec := f.sink.next(t); rec.Replica != "live" || rec.Status != 200 {
			t.Errorf("request %d: record %+v", i, rec)
		}
	}
}

func TestNoReachableReplicaIs502(t *testing.T) {
	f := &fixture{sink: &capture{records: make(chan *telemetrypb.RequestRecord, 4)}}
	f.start(t, []config.Replica{{Name: "dead", URL: deadURL(t)}})
	resp := f.post(t, "/v1/chat/completions", chatRequest)
	body := readAll(t, resp)
	if resp.StatusCode != 502 || !strings.Contains(body, `"error"`) {
		t.Errorf("status %d body %s", resp.StatusCode, body)
	}
	if rec := f.sink.next(t); rec.Status != 502 || rec.Replica != "dead" {
		t.Errorf("record: %+v", rec)
	}
}

func TestModelListsRouteRequests(t *testing.T) {
	a, b := &mockengine.Engine{Model: "model-a"}, &mockengine.Engine{Model: "model-b"}
	srvA, srvB := httptest.NewServer(a.Handler()), httptest.NewServer(b.Handler())
	t.Cleanup(srvA.Close)
	t.Cleanup(srvB.Close)
	f := &fixture{sink: &capture{records: make(chan *telemetrypb.RequestRecord, 64)}}
	f.start(t, []config.Replica{
		{Name: "ra", URL: srvA.URL, Models: []string{"model-a"}},
		{Name: "rb", URL: srvB.URL, Models: []string{"model-b"}},
	})

	for i := 0; i < 4; i++ {
		readAll(t, f.post(t, "/v1/chat/completions", `{"model":"model-b","messages":[]}`))
		if rec := f.sink.next(t); rec.Replica != "rb" {
			t.Errorf("model-b went to %q", rec.Replica)
		}
	}
	resp := f.post(t, "/v1/chat/completions", `{"model":"model-c","messages":[]}`)
	if body := readAll(t, resp); resp.StatusCode != 404 || !strings.Contains(body, "model-c") {
		t.Errorf("unknown model: status %d body %s", resp.StatusCode, body)
	}
}

func TestClientDisconnectIsRecordedAs499(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{TokenDelay: 30 * time.Millisecond, Tokens: 1000})
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, f.gateway.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"mock-model","stream":true,"messages":[]}`))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 64)
	_, _ = resp.Body.Read(buf)
	cancel()
	resp.Body.Close()

	rec := f.sink.next(t)
	if rec.Status != 499 || !rec.Stream {
		t.Errorf("record: %+v", rec)
	}
	r1, _ := f.pool.Get("r1")
	if r1.Inflight() != 0 {
		t.Errorf("inflight = %d after the client left, want 0", r1.Inflight())
	}
}

func TestGatewayOwnErrors(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{})

	resp := f.post(t, "/v1/audio/speech", `{}`)
	if body := readAll(t, resp); resp.StatusCode != 404 || !strings.Contains(body, `"error"`) {
		t.Errorf("unknown path: status %d body %s", resp.StatusCode, body)
	}

	big := `{"model":"m","prompt":"` + strings.Repeat("x", 2<<20) + `"}`
	resp = f.post(t, "/v1/completions", big)
	if readAll(t, resp); resp.StatusCode != 413 {
		t.Errorf("oversized body: status %d, want 413", resp.StatusCode)
	}
	if f.seen[0].count() != 0 {
		t.Error("rejected requests must not reach the engine")
	}
}

func TestHopByHopHeadersAreNotForwarded(t *testing.T) {
	f := newFixture(t, &mockengine.Engine{})
	readAll(t, f.post(t, "/v1/chat/completions", chatRequest,
		"Authorization", "Bearer engine-key",
		"Accept-Encoding", "gzip",
		"Connection", "X-Private",
		"X-Private", "secret"))
	f.sink.next(t)
	h := f.seen[0].headers[0]
	if h.Get("Authorization") != "Bearer engine-key" {
		t.Error("Authorization must reach the engine")
	}
	if h.Get("X-Private") != "" || h.Get("Accept-Encoding") != "" {
		t.Errorf("connection-scoped or encoding headers leaked upstream: %v", h)
	}
}

// TestNoRequestFailsWhileCollectorIsDown is the spec's guarantee: the gateway
// drops telemetry, never requests. It sends more requests than the export
// queue holds, so the overflow path runs too.
func TestNoRequestFailsWhileCollectorIsDown(t *testing.T) {
	if testing.Short() {
		t.Skip("sends 20,000 requests")
	}
	exporter, err := export.New(strings.TrimPrefix(deadURL(t), "http://"), "gw", slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go exporter.Run(ctx)

	engine := httptest.NewServer((&mockengine.Engine{}).Handler())
	t.Cleanup(engine.Close)
	pool, _ := replica.NewPool([]config.Replica{{Name: "r1", URL: engine.URL}})
	gateway := httptest.NewServer(New(Options{
		Pool: pool, Hasher: testHasher(), Sink: exporter, MaxBodyBytes: 1 << 20,
		TenantHeader: "X-Tenant-Id", FeatureHeader: "X-Feature",
	}))
	t.Cleanup(gateway.Close)

	const workers, perWorker = 50, 400
	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: workers}}
	var failed atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				body := chatRequest
				if i%4 == 0 {
					body = `{"model":"mock-model","stream":true,"max_tokens":3,"messages":[]}`
				}
				resp, err := client.Post(gateway.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
				if err != nil {
					failed.Add(1)
					continue
				}
				out, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				if resp.StatusCode != 200 || len(out) == 0 {
					failed.Add(1)
				}
			}
		}()
	}
	wg.Wait()

	if n := failed.Load(); n != 0 {
		t.Errorf("%d of %d requests failed with the collector down", n, workers*perWorker)
	}
	s := exporter.Stats()
	t.Logf("with the collector down: %d requests served, %d records dropped, %d still queued", workers*perWorker, s.Dropped, s.Queued)
	if s.Exported != 0 || s.Dropped == 0 {
		t.Errorf("stats %+v: nothing can have been exported, and the overflow must have been dropped", s)
	}
}

// TestOverheadBudget checks the spec's limit: the gateway adds under 2 ms at
// p50 to a non-streaming request. Direct and proxied requests alternate so
// machine noise hits both alike.
func TestOverheadBudget(t *testing.T) {
	if testing.Short() {
		t.Skip("timing test")
	}
	engine := &mockengine.Engine{}
	direct := httptest.NewServer(engine.Handler())
	t.Cleanup(direct.Close)
	f := &fixture{sink: &capture{records: make(chan *telemetrypb.RequestRecord, 1<<16)}}
	f.start(t, []config.Replica{{Name: "r1", URL: direct.URL}})

	body, _ := json.Marshal(map[string]any{
		"model": "mock-model", "max_tokens": 16,
		"messages": []map[string]string{
			{"role": "system", "content": strings.Repeat("You are a helpful assistant. ", 60)},
			{"role": "user", "content": strings.Repeat("Summarise the following paragraph. ", 20)},
		},
	})
	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 4}}
	call := func(url string) time.Duration {
		start := time.Now()
		resp, err := client.Post(url+"/v1/chat/completions", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return time.Since(start)
	}

	const rounds = 2000
	for i := 0; i < 200; i++ { // warm up connections and caches
		call(direct.URL)
		call(f.gateway.URL)
	}
	directTimes, proxiedTimes := make([]time.Duration, rounds), make([]time.Duration, rounds)
	for i := 0; i < rounds; i++ {
		directTimes[i] = call(direct.URL)
		proxiedTimes[i] = call(f.gateway.URL)
	}
	p50 := func(d []time.Duration) time.Duration {
		slices.Sort(d)
		return d[len(d)/2]
	}
	d, p := p50(directTimes), p50(proxiedTimes)
	added := p - d
	t.Logf("%d-byte request: direct p50 %v, through gateway p50 %v, added %v", len(body), d, p, added)
	if added > 2*time.Millisecond {
		t.Errorf("gateway adds %v at p50, over the 2 ms budget", added)
	}
}

func BenchmarkInspect(b *testing.B) {
	for _, size := range []int{1 << 10, 16 << 10, 256 << 10} {
		body, _ := json.Marshal(map[string]any{
			"model": "m", "stream": true,
			"messages": []map[string]string{{"role": "user", "content": strings.Repeat("lorem ipsum dolor sit amet ", size/27)}},
		})
		b.Run(fmt.Sprintf("%dKB", len(body)>>10), func(b *testing.B) {
			h := testHasher()
			b.SetBytes(int64(len(body)))
			for b.Loop() {
				inspect(endpointChat, body, h)
			}
		})
	}
}
