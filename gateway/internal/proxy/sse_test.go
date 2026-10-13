package proxy

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

const (
	roleChunk    = `data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}` + "\n\n"
	contentChunk = `data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" hi"},"finish_reason":null}]}` + "\n\n"
	finishChunk  = `data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}` + "\n\n"
	usageChunk   = `data: {"id":"c1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"prompt_tokens_details":{"cached_tokens":8}}}` + "\n\n"
	doneEvent    = "data: [DONE]\n\n"
)

// relayAll feeds stream to a relay in pieces of the given size and returns
// what came out.
func relayAll(t *testing.T, s *sseRelay, stream string, piece int) string {
	t.Helper()
	var out bytes.Buffer
	write := func(p []byte) error { out.Write(p); return nil }
	now := time.Unix(0, 0)
	for len(stream) > 0 {
		n := min(piece, len(stream))
		// feed may keep no reference to its input, so hand it a throwaway copy.
		if err := s.feed([]byte(stream[:n]), now, write); err != nil {
			t.Fatal(err)
		}
		stream = stream[n:]
		now = now.Add(time.Millisecond)
	}
	if err := s.finish(write); err != nil {
		t.Fatal(err)
	}
	return out.String()
}

func TestRelayStripsOnlyTheUsageChunk(t *testing.T) {
	stream := roleChunk + contentChunk + contentChunk + finishChunk + usageChunk + doneEvent
	want := roleChunk + contentChunk + contentChunk + finishChunk + doneEvent

	// Every way of cutting the stream into reads must give the same bytes.
	for piece := 1; piece <= len(stream); piece++ {
		s := &sseRelay{stripUsage: true}
		if got := relayAll(t, s, stream, piece); got != want {
			t.Fatalf("piece size %d:\n got %q\nwant %q", piece, got, want)
		}
		if s.usage == nil || s.usage.PromptTokens != 11 || s.usage.CompletionTokens != 3 || s.usage.cached() != 8 {
			t.Fatalf("piece size %d: usage = %+v", piece, s.usage)
		}
	}
}

func TestRelayKeepsUsageChunkTheClientAskedFor(t *testing.T) {
	stream := roleChunk + contentChunk + finishChunk + usageChunk + doneEvent
	for _, piece := range []int{1, 7, len(stream)} {
		s := &sseRelay{stripUsage: false}
		if got := relayAll(t, s, stream, piece); got != stream {
			t.Fatalf("piece size %d: stream changed:\n got %q\nwant %q", piece, got, stream)
		}
		if s.usage == nil || s.usage.CompletionTokens != 3 {
			t.Fatalf("piece size %d: usage = %+v", piece, s.usage)
		}
	}
}

func TestRelayHandlesCRLFAndComments(t *testing.T) {
	crlf := func(s string) string { return strings.ReplaceAll(s, "\n", "\r\n") }
	stream := ": keep-alive\r\n\r\n" + crlf(roleChunk+contentChunk+usageChunk+doneEvent)
	want := ": keep-alive\r\n\r\n" + crlf(roleChunk+contentChunk+doneEvent)
	for piece := 1; piece <= len(stream); piece++ {
		s := &sseRelay{stripUsage: true}
		if got := relayAll(t, s, stream, piece); got != want {
			t.Fatalf("piece size %d:\n got %q\nwant %q", piece, got, want)
		}
	}
}

func TestRelayForwardsAnUnterminatedTail(t *testing.T) {
	stream := contentChunk + `data: {"cut off mid-event`
	s := &sseRelay{stripUsage: true}
	if got := relayAll(t, s, stream, 5); got != stream {
		t.Fatalf("got %q, want %q", got, stream)
	}
}

func TestRelayFallsBackToRawOnOversizedEvent(t *testing.T) {
	stream := "data: " + strings.Repeat("x", 2*maxPendingEvent) + "\n\n" + doneEvent
	s := &sseRelay{stripUsage: true}
	if got := relayAll(t, s, stream, 64<<10); got != stream {
		t.Fatalf("oversized event was not passed through intact (%d bytes out, %d in)", len(got), len(stream))
	}
	if !s.raw {
		t.Error("relay should have given up framing")
	}
}

func TestRelayTimesFirstContentAndGaps(t *testing.T) {
	s := &sseRelay{}
	write := func([]byte) error { return nil }
	at := func(ms int) time.Time { return time.Unix(0, 0).Add(time.Duration(ms) * time.Millisecond) }

	_ = s.feed([]byte(roleChunk), at(40), write) // role only: no token yet
	if !s.firstContent.IsZero() {
		t.Fatal("a role-only chunk must not count as the first token")
	}
	_ = s.feed([]byte(contentChunk), at(50), write)
	_ = s.feed([]byte(contentChunk), at(60), write)
	_ = s.feed([]byte(contentChunk), at(90), write)
	_ = s.feed([]byte(usageChunk+doneEvent), at(500), write) // neither is a content chunk

	if !s.firstContent.Equal(at(50)) {
		t.Errorf("first content at %v, want %v", s.firstContent, at(50))
	}
	p50, p95 := s.gaps.quantiles()
	if p50 != 10 || p95 != 30 {
		t.Errorf("gaps p50=%v p95=%v, want 10 and 30", p50, p95)
	}
}

func TestCompletionsStreamTextCountsAsContent(t *testing.T) {
	if !hasContent([]byte(`{"object":"text_completion","choices":[{"index":0,"text":" hi","finish_reason":null}]}`)) {
		t.Error("a completions chunk with text is content")
	}
	if !hasContent([]byte(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]}}]}`)) {
		t.Error("a tool-call delta is content")
	}
	if hasContent([]byte(`{"choices":[{"delta":{"role":"assistant","content":""}}]}`)) {
		t.Error("a role-only delta is not content")
	}
}

func TestGapSampleStaysBounded(t *testing.T) {
	var g gapSample
	for i := 0; i < 10*gapSampleCap; i++ {
		g.add(5 * time.Millisecond)
	}
	if len(g.ms) != gapSampleCap {
		t.Errorf("sample holds %d gaps, want %d", len(g.ms), gapSampleCap)
	}
	if p50, p95 := g.quantiles(); p50 != 5 || p95 != 5 {
		t.Errorf("p50=%v p95=%v, want 5 and 5", p50, p95)
	}
}
