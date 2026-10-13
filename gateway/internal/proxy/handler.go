// Package proxy is the request path of the gateway: an OpenAI-compatible
// pass-through that balances across engine replicas and emits one telemetry
// record per request. It never stores request or response text.
package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/prefixhash"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/replica"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/telemetrypb"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/trace"
)

const (
	// statusClientClosed marks a request the client abandoned before the
	// response finished. It is recorded, never sent.
	statusClientClosed = 499

	// maxAttempts bounds how many replicas one request is tried on when
	// replicas refuse the connection.
	maxAttempts = 3

	// maxUsageCapture bounds how much of a non-streaming response is held to
	// read its usage object. Larger responses are relayed without token counts.
	maxUsageCapture = 8 << 20

	// inlineParseLimit is the largest captured response parsed before the
	// handler returns. Parsing a bigger one would delay the end of a chunked
	// response, so it happens after.
	inlineParseLimit = 64 << 10

	// pooledBufferLimit keeps a rare huge response from pinning its buffer
	// in the pool.
	pooledBufferLimit = 1 << 20
)

// Sink receives one record per finished request. Emit must not block.
type Sink interface {
	Emit(*telemetrypb.RequestRecord)
}

type Options struct {
	Pool          *replica.Pool
	Hasher        *prefixhash.Hasher
	Sink          Sink
	TenantHeader  string
	FeatureHeader string
	MaxBodyBytes  int64
	// Transport reaches the replicas. Nil uses NewTransport.
	Transport http.RoundTripper
	Logger    *slog.Logger
}

type Handler struct {
	Options
	mux *http.ServeMux
}

func New(o Options) *Handler {
	if o.Transport == nil {
		o.Transport = NewTransport()
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	h := &Handler{Options: o, mux: http.NewServeMux()}
	h.mux.HandleFunc("POST /v1/chat/completions", h.inference(endpointChat))
	h.mux.HandleFunc("POST /v1/completions", h.inference(endpointCompletions))
	h.mux.HandleFunc("POST /v1/embeddings", h.inference(endpointEmbeddings))
	h.mux.HandleFunc("GET /v1/models", h.models)
	h.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "ok\n")
	})
	h.mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusNotFound, "invalid_request_error", "the gateway does not proxy "+r.Method+" "+r.URL.Path)
	})
	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) { h.mux.ServeHTTP(w, r) }

// NewTransport returns a transport tuned for long-lived streaming responses
// to a small, fixed set of replicas.
func NewTransport() *http.Transport {
	return &http.Transport{
		DialContext:         (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConnsPerHost: 1024,
		IdleConnTimeout:     90 * time.Second,
		// Responses are relayed as the engine wrote them. The gateway removes
		// Accept-Encoding upstream so the usage object stays readable.
		DisableCompression: true,
	}
}

func (h *Handler) inference(ep endpoint) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		body, err := readBody(w, r, h.MaxBodyBytes)
		if err != nil {
			var tooBig *http.MaxBytesError
			if errors.As(err, &tooBig) {
				writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "request body is larger than the gateway's limit")
			} else {
				writeError(w, http.StatusBadRequest, "invalid_request_error", "could not read the request body")
			}
			return
		}

		in := inspect(ep, body, h.Hasher)
		tc := trace.Next(r.Header.Get("Traceparent"))
		requestID := r.Header.Get("X-Request-Id")
		if requestID == "" {
			requestID = trace.NewRequestID()
		}
		rec := &telemetrypb.RequestRecord{
			RequestId:        requestID,
			TraceId:          tc.TraceID,
			ArrivalUnixNanos: start.UnixNano(),
			Model:            in.model,
			Tenant:           r.Header.Get(h.TenantHeader),
			Feature:          r.Header.Get(h.FeatureHeader),
			Endpoint:         ep.String(),
			Stream:           in.stream,
			MaxTokens:        in.maxTokens,
			PromptChars:      in.promptChars,
			PrefixHashes:     in.hashes,
		}

		resp, rep, failure := h.dispatch(r, in.body, in.model, tc, requestID)
		if rep != nil {
			rec.Replica = rep.Name
		}
		if resp == nil {
			if failure.status != statusClientClosed {
				w.Header().Set("X-Request-Id", requestID)
				if failure.status == http.StatusServiceUnavailable {
					w.Header().Set("Retry-After", "1")
				}
				writeError(w, failure.status, failure.kind, failure.message)
			}
			rec.Status = uint32(failure.status)
			rec.E2EMs = msSince(start)
			h.Sink.Emit(rec)
			return
		}
		defer rep.Release()
		defer resp.Body.Close()

		copyHeaders(w.Header(), resp.Header)
		if w.Header().Get("X-Request-Id") == "" {
			w.Header().Set("X-Request-Id", requestID)
		}
		w.WriteHeader(resp.StatusCode)

		if isEventStream(resp) {
			h.relayStream(w, r, resp, rec, in.stripUsage, start)
		} else {
			h.relayBody(w, r, resp, rec, start)
		}
	}
}

// models answers GET /v1/models from the next replica. It emits no record.
func (h *Handler) models(w http.ResponseWriter, r *http.Request) {
	resp, rep, failure := h.dispatch(r, nil, "", trace.Next(r.Header.Get("Traceparent")), "")
	if resp == nil {
		if failure.status != statusClientClosed {
			writeError(w, failure.status, failure.kind, failure.message)
		}
		return
	}
	defer rep.Release()
	defer resp.Body.Close()
	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

type failure struct {
	status  int
	kind    string
	message string
}

// dispatch sends the request to a replica and returns its response with the
// replica still acquired. When a replica cannot be reached at all the next
// one is tried; a nil response comes with the reason and, if any replica was
// tried, the last one.
func (h *Handler) dispatch(r *http.Request, body []byte, model string, tc trace.Context, requestID string) (*http.Response, *replica.Replica, failure) {
	var tried []*replica.Replica
	for len(tried) < maxAttempts {
		rep := h.Pool.Acquire(model, tried)
		if rep == nil {
			break
		}
		resp, err := h.Transport.RoundTrip(upstreamRequest(r, rep, body, tc, requestID))
		if err == nil {
			return resp, rep, failure{}
		}
		rep.Release()
		if r.Context().Err() != nil {
			return nil, rep, failure{status: statusClientClosed}
		}
		h.Logger.Warn("replica unreachable", "replica", rep.Name, "error", err)
		tried = append(tried, rep)
	}

	switch {
	case len(tried) > 0:
		return nil, tried[len(tried)-1], failure{http.StatusBadGateway, "api_error", "no replica could be reached"}
	case !h.Pool.Serves(model):
		return nil, nil, failure{http.StatusNotFound, "invalid_request_error", "no replica serves model " + model}
	default:
		return nil, nil, failure{http.StatusServiceUnavailable, "api_error", "every replica for this model is drained"}
	}
}

func upstreamRequest(r *http.Request, rep *replica.Replica, body []byte, tc trace.Context, requestID string) *http.Request {
	u := *rep.URL
	u.Path = strings.TrimRight(u.Path, "/") + r.URL.Path
	u.RawQuery = r.URL.RawQuery

	req := (&http.Request{
		Method:     r.Method,
		URL:        &u,
		Host:       u.Host,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     make(http.Header, len(r.Header)+2),
	}).WithContext(r.Context())
	if body != nil {
		// GetBody lets the transport resend the request when a kept-alive
		// connection turns out to have been closed by the replica.
		req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
		req.Body, _ = req.GetBody()
		req.ContentLength = int64(len(body))
	}

	copyHeaders(req.Header, r.Header)
	// The usage object has to arrive uncompressed to be read.
	req.Header.Del("Accept-Encoding")
	req.Header.Set("Traceparent", tc.Header())
	if requestID != "" {
		req.Header.Set("X-Request-Id", requestID)
	}
	return req
}

// relayBody forwards a non-streaming response and reads its usage object.
func (h *Handler) relayBody(w http.ResponseWriter, r *http.Request, resp *http.Response, rec *telemetrypb.RequestRecord, start time.Time) {
	captured := bodyBuffers.Get().(*bytes.Buffer)
	captured.Reset()
	fits := true

	buf := copyBuffers.Get().(*[]byte)
	clientGone, upstreamBroke := false, false
	for {
		n, err := resp.Body.Read(*buf)
		if n > 0 {
			if fits && captured.Len()+n <= maxUsageCapture {
				captured.Write((*buf)[:n])
			} else {
				fits = false
			}
			if _, werr := w.Write((*buf)[:n]); werr != nil {
				clientGone = true
				break
			}
		}
		if err != nil {
			upstreamBroke = err != io.EOF
			break
		}
	}
	copyBuffers.Put(buf)

	rec.E2EMs = msSince(start)
	rec.Status = finalStatus(r, resp.StatusCode, clientGone, upstreamBroke)

	finish := func() {
		if fits && resp.StatusCode == http.StatusOK {
			applyUsage(rec, usageOf(captured.Bytes()))
		}
		if captured.Cap() <= pooledBufferLimit {
			bodyBuffers.Put(captured)
		}
		h.Sink.Emit(rec)
	}
	if captured.Len() > inlineParseLimit {
		go finish()
		return
	}
	finish()
}

// relayStream forwards an event stream chunk by chunk, timing it as it goes.
func (h *Handler) relayStream(w http.ResponseWriter, r *http.Request, resp *http.Response, rec *telemetrypb.RequestRecord, stripUsage bool, start time.Time) {
	relay := sseRelay{stripUsage: stripUsage}
	flusher := http.NewResponseController(w)
	clientGone, upstreamBroke := false, false
	write := func(p []byte) error {
		if len(p) == 0 {
			return nil
		}
		_, err := w.Write(p)
		return err
	}

	buf := copyBuffers.Get().(*[]byte)
	for {
		n, err := resp.Body.Read(*buf)
		if n > 0 {
			if relay.feed((*buf)[:n], time.Now(), write) != nil {
				clientGone = true
				break
			}
			if ferr := flusher.Flush(); ferr != nil && !errors.Is(ferr, http.ErrNotSupported) {
				clientGone = true
				break
			}
		}
		if err != nil {
			upstreamBroke = err != io.EOF
			_ = relay.finish(write)
			break
		}
	}
	copyBuffers.Put(buf)

	rec.E2EMs = msSince(start)
	rec.Status = finalStatus(r, resp.StatusCode, clientGone, upstreamBroke)
	if !relay.firstContent.IsZero() {
		rec.TtftMs = float32(relay.firstContent.Sub(start)) / float32(time.Millisecond)
	}
	rec.GapP50Ms, rec.GapP95Ms = relay.gaps.quantiles()
	applyUsage(rec, relay.usage)
	h.Sink.Emit(rec)
}

func applyUsage(rec *telemetrypb.RequestRecord, u *usage) {
	if u == nil {
		return
	}
	rec.PromptTokens = u.PromptTokens
	rec.CompletionTokens = u.CompletionTokens
	rec.CachedTokens = u.cached()
}

// finalStatus is the status the record carries. The client may already have
// received a 200 by the time the stream breaks, so the outcome can differ
// from the status line that was sent.
func finalStatus(r *http.Request, upstream int, clientGone, upstreamBroke bool) uint32 {
	switch {
	case clientGone || r.Context().Err() != nil:
		return statusClientClosed
	case upstreamBroke:
		return http.StatusBadGateway
	default:
		return uint32(upstream)
	}
}

func isEventStream(resp *http.Response) bool {
	return strings.HasPrefix(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream")
}

func msSince(t time.Time) float32 {
	return float32(time.Since(t)) / float32(time.Millisecond)
}

func readBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, error) {
	if r.ContentLength > limit {
		return nil, &http.MaxBytesError{Limit: limit}
	}
	body := http.MaxBytesReader(w, r.Body, limit)
	if r.ContentLength > 0 {
		buf := make([]byte, r.ContentLength)
		if _, err := io.ReadFull(body, buf); err != nil {
			return nil, err
		}
		return buf, nil
	}
	return io.ReadAll(body)
}

// hopByHop are headers that describe one connection and must not be relayed.
var hopByHop = map[string]bool{
	"Connection":          true,
	"Keep-Alive":          true,
	"Proxy-Authenticate":  true,
	"Proxy-Authorization": true,
	"Proxy-Connection":    true,
	"Te":                  true,
	"Trailer":             true,
	"Transfer-Encoding":   true,
	"Upgrade":             true,
}

func copyHeaders(dst, src http.Header) {
	var named map[string]bool
	for _, v := range src["Connection"] {
		for _, name := range strings.Split(v, ",") {
			if name = strings.TrimSpace(name); name != "" {
				if named == nil {
					named = map[string]bool{}
				}
				named[http.CanonicalHeaderKey(name)] = true
			}
		}
	}
	for k, vs := range src {
		if hopByHop[k] || named[k] {
			continue
		}
		dst[k] = append(dst[k], vs...)
	}
}

// writeError answers in the error shape OpenAI clients parse.
func writeError(w http.ResponseWriter, status int, kind, message string) {
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{"message": message, "type": kind, "param": nil, "code": status},
	})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

var (
	copyBuffers = sync.Pool{New: func() any {
		b := make([]byte, 32<<10)
		return &b
	}}
	bodyBuffers = sync.Pool{New: func() any { return new(bytes.Buffer) }}
)
