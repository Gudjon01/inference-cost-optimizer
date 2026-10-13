package proxy

import (
	"bytes"
	"encoding/json"
	"math"
	"math/rand/v2"
	"slices"
	"time"
)

// maxPendingEvent bounds how much of an unterminated event is held back. A
// stream that exceeds it is not framed like SSE, so it is passed through raw.
const maxPendingEvent = 1 << 20

var (
	dataField  = []byte("data:")
	doneMarker = []byte("[DONE]")
	usageKey   = []byte(`"usage"`)
)

// sseRelay frames an upstream event stream so each event can be timed and the
// usage chunk the gateway asked for can be withheld. Every other event is
// forwarded byte for byte.
type sseRelay struct {
	stripUsage bool

	pending []byte
	raw     bool

	firstContent time.Time
	lastChunk    time.Time
	gaps         gapSample
	usage        *usage
}

// feed takes bytes just read from upstream and calls write with the bytes to
// forward, in order. data is only valid during the call.
func (s *sseRelay) feed(data []byte, now time.Time, write func([]byte) error) error {
	if s.raw {
		return write(data)
	}
	if len(s.pending) > 0 {
		s.pending = append(s.pending, data...)
		data = s.pending
	}

	consumed, unwritten := 0, 0
	for {
		n := nextEvent(data[consumed:])
		if n == 0 {
			break
		}
		if s.observe(data[consumed:consumed+n], now) {
			if err := write(data[unwritten:consumed]); err != nil {
				return err
			}
			unwritten = consumed + n
		}
		consumed += n
	}
	if err := write(data[unwritten:consumed]); err != nil {
		return err
	}

	rest := data[consumed:]
	if len(rest) > maxPendingEvent {
		s.raw = true
		s.pending = nil
		return write(rest)
	}
	// rest may alias s.pending; append moves it to the front safely.
	s.pending = append(s.pending[:0], rest...)
	return nil
}

// finish forwards whatever is left when the upstream closes mid-event.
func (s *sseRelay) finish(write func([]byte) error) error {
	if len(s.pending) == 0 {
		return nil
	}
	rest := s.pending
	s.pending = nil
	return write(rest)
}

// nextEvent returns the length of the first complete event in b, including
// the blank line that ends it, or 0 when b holds no complete event. Lines may
// end in LF, CRLF or CR.
func nextEvent(b []byte) int {
	lineStart := 0
	for i := 0; i < len(b); i++ {
		switch b[i] {
		case '\n':
			if i == lineStart {
				return i + 1
			}
			lineStart = i + 1
		case '\r':
			if i+1 == len(b) {
				return 0 // the next byte decides whether this is CRLF
			}
			end := i + 1
			if b[end] == '\n' {
				end++
			}
			if i == lineStart {
				return end
			}
			lineStart = end
			i = end - 1
		}
	}
	return 0
}

// observe records what an event says and reports whether to withhold it.
func (s *sseRelay) observe(event []byte, now time.Time) (drop bool) {
	payload, ok := dataPayload(event)
	if !ok || bytes.Equal(payload, doneMarker) {
		return false
	}

	if bytes.Contains(payload, usageKey) {
		var chunk struct {
			Choices []json.RawMessage `json:"choices"`
			Usage   *usage            `json:"usage"`
		}
		_ = json.Unmarshal(payload, &chunk)
		if chunk.Usage != nil {
			s.usage = chunk.Usage
			if len(chunk.Choices) == 0 {
				// The usage-only chunk that include_usage adds at the end.
				return s.stripUsage
			}
		}
	}

	if s.firstContent.IsZero() {
		// Only the opening chunks are parsed: an engine may send a role-only
		// chunk before the first token exists.
		if !hasContent(payload) {
			return false
		}
		s.firstContent = now
	} else {
		s.gaps.add(now.Sub(s.lastChunk))
	}
	s.lastChunk = now
	return false
}

// dataPayload returns the value of an event's data field.
func dataPayload(event []byte) ([]byte, bool) {
	for len(event) > 0 {
		line := event
		if i := bytes.IndexAny(event, "\r\n"); i >= 0 {
			line, event = event[:i], event[i+1:]
		} else {
			event = nil
		}
		if rest, ok := bytes.CutPrefix(line, dataField); ok {
			return bytes.TrimPrefix(rest, []byte(" ")), true
		}
	}
	return nil, false
}

// hasContent reports whether a chunk carries generated output: text, reasoning
// or a tool call.
func hasContent(payload []byte) bool {
	var chunk struct {
		Choices []struct {
			Text  string `json:"text"`
			Delta struct {
				Content          string            `json:"content"`
				ReasoningContent string            `json:"reasoning_content"`
				Reasoning        string            `json:"reasoning"`
				ToolCalls        []json.RawMessage `json:"tool_calls"`
			} `json:"delta"`
		} `json:"choices"`
	}
	_ = json.Unmarshal(payload, &chunk)
	for _, c := range chunk.Choices {
		d := c.Delta
		if c.Text != "" || d.Content != "" || d.ReasoningContent != "" || d.Reasoning != "" || len(d.ToolCalls) > 0 {
			return true
		}
	}
	return false
}

// gapSample keeps the gaps between a stream's chunks. Past its capacity it
// becomes a uniform random sample, so memory per stream stays bounded.
type gapSample struct {
	ms   []float32
	seen int
}

const gapSampleCap = 4096

func (g *gapSample) add(d time.Duration) {
	v := float32(d) / float32(time.Millisecond)
	g.seen++
	if len(g.ms) < gapSampleCap {
		g.ms = append(g.ms, v)
		return
	}
	if j := rand.IntN(g.seen); j < gapSampleCap {
		g.ms[j] = v
	}
}

func (g *gapSample) quantiles() (p50, p95 float32) {
	if len(g.ms) == 0 {
		return 0, 0
	}
	slices.Sort(g.ms)
	return g.ms[quantileIndex(len(g.ms), 0.50)], g.ms[quantileIndex(len(g.ms), 0.95)]
}

// quantileIndex is the nearest-rank position of quantile q among n sorted values.
func quantileIndex(n int, q float64) int {
	i := int(math.Ceil(float64(n)*q)) - 1
	return max(0, min(i, n-1))
}
