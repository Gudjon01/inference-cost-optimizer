// Package trace builds the W3C traceparent header the gateway sends upstream.
package trace

import (
	"crypto/rand"
	"encoding/hex"
)

// Context is one hop of a W3C trace: the trace the request belongs to and
// the gateway's own span within it.
type Context struct {
	TraceID string // 32 lowercase hex characters
	SpanID  string // 16 lowercase hex characters
	flags   string
}

// Next continues the trace named by an incoming traceparent header, or starts
// a new sampled trace when the header is missing or malformed.
func Next(incoming string) Context {
	c := Context{flags: "01"}
	if traceID, flags, ok := parse(incoming); ok {
		c.TraceID, c.flags = traceID, flags
	} else {
		c.TraceID = randomHex(16)
	}
	c.SpanID = randomHex(8)
	return c
}

func (c Context) Header() string {
	return "00-" + c.TraceID + "-" + c.SpanID + "-" + c.flags
}

// parse reads version-00 traceparent: 00-<32 hex>-<16 hex>-<2 hex>.
func parse(h string) (traceID, flags string, ok bool) {
	if len(h) != 55 || h[0:3] != "00-" || h[35] != '-' || h[52] != '-' {
		return "", "", false
	}
	traceID, spanID, flags := h[3:35], h[36:52], h[53:55]
	if !isLowerHex(traceID) || !isLowerHex(spanID) || !isLowerHex(flags) {
		return "", "", false
	}
	if allZero(traceID) || allZero(spanID) {
		return "", "", false
	}
	return traceID, flags, true
}

func isLowerHex(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

func allZero(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] != '0' {
			return false
		}
	}
	return true
}

func randomHex(nBytes int) string {
	b := make([]byte, nBytes)
	// crypto/rand.Read never fails on supported platforms; a zero id would
	// only make this one trace unjoinable.
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// NewRequestID returns a random 128-bit id in hex.
func NewRequestID() string { return randomHex(16) }
