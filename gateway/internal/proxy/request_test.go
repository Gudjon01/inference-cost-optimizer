package proxy

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/prefixhash"
)

func testHasher() *prefixhash.Hasher { return prefixhash.New([]byte("test-salt"), 16, 64) }

func chatBody(system, user string, extra string) []byte {
	b, _ := json.Marshal(map[string]any{
		"model": "m",
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
	})
	if extra != "" {
		b = append(b[:len(b)-1], ("," + extra + "}")...)
	}
	return b
}

func TestInspectReadsRecordFields(t *testing.T) {
	in := inspect(endpointChat, chatBody("sys", "hello", `"max_tokens":77,"stream":false`), testHasher())
	if in.model != "m" || in.stream || in.maxTokens != 77 || in.stripUsage {
		t.Errorf("got %+v", in)
	}
	// max_completion_tokens is the newer name and wins when both are set.
	in = inspect(endpointChat, chatBody("sys", "hello", `"max_tokens":77,"max_completion_tokens":50`), testHasher())
	if in.maxTokens != 50 {
		t.Errorf("maxTokens = %d, want 50", in.maxTokens)
	}
}

func TestInspectSharedSystemPromptSharesHashes(t *testing.T) {
	system := strings.Repeat("You are a careful assistant. ", 8) // well over several blocks
	a := inspect(endpointChat, chatBody(system, "first question", ""), testHasher())
	b := inspect(endpointChat, chatBody(system, "a different question", ""), testHasher())
	c := inspect(endpointChat, chatBody("Another system prompt entirely, also long enough.", "first question", ""), testHasher())

	shared := 0
	for shared < min(len(a.hashes), len(b.hashes)) && a.hashes[shared] == b.hashes[shared] {
		shared++
	}
	if want := len(system) / 16; shared < want {
		t.Errorf("requests with the same %d-char system prompt share %d blocks, want at least %d", len(system), shared, want)
	}
	if len(c.hashes) > 0 && c.hashes[0] == a.hashes[0] {
		t.Error("different system prompts must not share the first block")
	}
	if slices.Equal(a.hashes, b.hashes) {
		t.Error("different user turns must diverge")
	}
}

func TestInspectContentPartsHashLikePlainText(t *testing.T) {
	plain := []byte(`{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("abcdefgh", 10) + `"}]}`)
	parts := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("abcdefgh", 5) + `"},{"type":"text","text":"` + strings.Repeat("abcdefgh", 5) + `"}]}]}`)
	a, b := inspect(endpointChat, plain, testHasher()), inspect(endpointChat, parts, testHasher())
	if !slices.Equal(a.hashes, b.hashes) || a.promptChars != b.promptChars {
		t.Errorf("text parts should hash like the same text as one string: %x (%d) vs %x (%d)", a.hashes, a.promptChars, b.hashes, b.promptChars)
	}

	image := func(url string) inspected {
		return inspect(endpointChat, []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+url+`"}}]}]}`), testHasher())
	}
	if x, y := image("https://example.com/a.png"), image("https://example.com/b.png"); slices.Equal(x.hashes, y.hashes) {
		t.Error("two different images must not hash the same")
	}
}

func TestInspectCompletionsAndEmbeddingsInputs(t *testing.T) {
	text := strings.Repeat("0123456789abcdef", 3)
	one := inspect(endpointCompletions, []byte(`{"model":"m","prompt":"`+text+`"}`), testHasher())
	if len(one.hashes) != 3 || one.promptChars != 48 {
		t.Errorf("string prompt: %d hashes over %d chars, want 3 over 48", len(one.hashes), one.promptChars)
	}
	emb := inspect(endpointEmbeddings, []byte(`{"model":"m","input":["`+text+`","more"],"stream":true}`), testHasher())
	if len(emb.hashes) != 3 || emb.stream {
		t.Errorf("embeddings: %d hashes, stream=%v; want 3 hashes and never streaming", len(emb.hashes), emb.stream)
	}
	if !slices.Equal(one.hashes, emb.hashes) {
		t.Error("the same leading text should hash the same on both endpoints")
	}
	tokens := inspect(endpointCompletions, []byte(`{"model":"m","prompt":[1,2,3,4,5,6,7,8,9,10,11,12]}`), testHasher())
	if tokens.promptChars == 0 {
		t.Error("a token-id prompt should still be measured")
	}
}

func TestInspectLeavesUnparseableBodiesAlone(t *testing.T) {
	for _, body := range []string{``, `not json`, `[1,2,3]`, `{"model":"m","stream":true`} {
		in := inspect(endpointChat, []byte(body), testHasher())
		if string(in.body) != body || in.stream || in.stripUsage {
			t.Errorf("body %q: got body %q stream=%v strip=%v; want it untouched", body, in.body, in.stream, in.stripUsage)
		}
	}
	// A field of the wrong type must not hide the fields that are fine.
	in := inspect(endpointChat, []byte(`{"model":"m","max_tokens":"lots","messages":[]}`), testHasher())
	if in.model != "m" {
		t.Errorf("model = %q, want m despite the bad max_tokens", in.model)
	}
}

func TestStreamingRequestsGetUsageTurnedOn(t *testing.T) {
	type options struct {
		IncludeUsage         bool `json:"include_usage"`
		ContinuousUsageStats bool `json:"continuous_usage_stats"`
	}
	decode := func(t *testing.T, body []byte) (stream bool, opts options) {
		t.Helper()
		var v struct {
			Model         string  `json:"model"`
			Stream        bool    `json:"stream"`
			StreamOptions options `json:"stream_options"`
		}
		if err := json.Unmarshal(body, &v); err != nil {
			t.Fatalf("body sent upstream is not valid JSON: %v\n%s", err, body)
		}
		if v.Model != "m" {
			t.Fatalf("model lost in rewrite: %s", body)
		}
		return v.Stream, v.StreamOptions
	}

	t.Run("no stream_options", func(t *testing.T) {
		orig := chatBody("s", "u", `"stream":true`)
		in := inspect(endpointChat, orig, testHasher())
		if !in.stream || !in.stripUsage {
			t.Fatalf("stream=%v strip=%v, want both true", in.stream, in.stripUsage)
		}
		if _, opts := decode(t, in.body); !opts.IncludeUsage {
			t.Errorf("include_usage not set: %s", in.body)
		}
		// Everything the client sent is still there, byte for byte.
		if !strings.HasPrefix(string(in.body), string(orig[:len(orig)-1])) {
			t.Errorf("client bytes changed:\n%s\n%s", orig, in.body)
		}
	})

	t.Run("stream_options without include_usage", func(t *testing.T) {
		for _, so := range []string{`{"continuous_usage_stats":true}`, `{"include_usage":false,"continuous_usage_stats":true}`} {
			in := inspect(endpointChat, chatBody("s", "u", `"stream":true,"stream_options":`+so), testHasher())
			_, opts := decode(t, in.body)
			if !in.stripUsage || !opts.IncludeUsage || !opts.ContinuousUsageStats {
				t.Errorf("stream_options %s: strip=%v, sent %+v", so, in.stripUsage, opts)
			}
		}
		in := inspect(endpointChat, chatBody("s", "u", `"stream":true,"stream_options":null`), testHasher())
		if _, opts := decode(t, in.body); !in.stripUsage || !opts.IncludeUsage {
			t.Errorf("null stream_options: strip=%v, sent %s", in.stripUsage, in.body)
		}
	})

	t.Run("client asked for usage itself", func(t *testing.T) {
		orig := chatBody("s", "u", `"stream":true,"stream_options":{"include_usage":true}`)
		in := inspect(endpointChat, orig, testHasher())
		if in.stripUsage || string(in.body) != string(orig) {
			t.Errorf("request must pass through untouched and unstripped; strip=%v body=%s", in.stripUsage, in.body)
		}
	})

	t.Run("not streaming", func(t *testing.T) {
		orig := chatBody("s", "u", "")
		in := inspect(endpointChat, orig, testHasher())
		if in.stream || in.stripUsage || string(in.body) != string(orig) {
			t.Errorf("non-streaming request must pass through untouched")
		}
	})
}
