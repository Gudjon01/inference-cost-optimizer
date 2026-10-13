package trace

import (
	"strings"
	"testing"
)

func TestNextContinuesIncomingTrace(t *testing.T) {
	in := "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
	c := Next(in)
	if c.TraceID != "4bf92f3577b34da6a3ce929d0e0e4736" {
		t.Errorf("trace id = %q, want the incoming one", c.TraceID)
	}
	if c.SpanID == "00f067aa0ba902b7" {
		t.Error("span id must be the gateway's own, not the caller's")
	}
	if !strings.HasSuffix(c.Header(), "-00") {
		t.Errorf("header %q should keep the caller's unsampled flag", c.Header())
	}
}

func TestNextStartsTraceOnBadHeader(t *testing.T) {
	for _, in := range []string{
		"",
		"garbage",
		"01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", // unknown version
		"00-00000000000000000000000000000000-00f067aa0ba902b7-01", // zero trace id
		"00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01", // upper case
	} {
		c := Next(in)
		h := c.Header()
		if _, _, ok := parse(h); !ok {
			t.Errorf("Next(%q) produced an invalid header %q", in, h)
		}
		if strings.Contains(in, c.TraceID) && in != "" {
			t.Errorf("Next(%q) reused a trace id from a rejected header", in)
		}
		if !strings.HasSuffix(h, "-01") {
			t.Errorf("new trace %q should be sampled", h)
		}
	}
}
