package proxy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/prefixhash"
)

type endpoint int

const (
	endpointChat endpoint = iota
	endpointCompletions
	endpointEmbeddings
)

func (e endpoint) String() string {
	switch e {
	case endpointChat:
		return "chat"
	case endpointCompletions:
		return "completions"
	default:
		return "embeddings"
	}
}

// requestShape is the part of an OpenAI-style request body the gateway reads.
type requestShape struct {
	Model               string          `json:"model"`
	Stream              bool            `json:"stream"`
	StreamOptions       json.RawMessage `json:"stream_options"`
	MaxTokens           *int64          `json:"max_tokens"`
	MaxCompletionTokens *int64          `json:"max_completion_tokens"`
	Tools               json.RawMessage `json:"tools"`
	Messages            []message       `json:"messages"`
	Prompt              textInput       `json:"prompt"`
	Input               textInput       `json:"input"`
}

type message struct {
	Role      string          `json:"role"`
	Content   contentText     `json:"content"`
	ToolCalls json.RawMessage `json:"tool_calls"`
}

// contentText is a chat message's content flattened to text pieces. Content
// is either a string or a list of typed parts; a part that is not text is
// reduced to its type and a digest so two different images still give two
// different prefixes.
type contentText []string

func (c *contentText) UnmarshalJSON(b []byte) error {
	switch firstByte(b) {
	case '"':
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		*c = contentText{s}
	case '[':
		var parts []json.RawMessage
		if err := json.Unmarshal(b, &parts); err != nil {
			return err
		}
		out := make(contentText, 0, len(parts))
		for _, raw := range parts {
			var p struct {
				Type string  `json:"type"`
				Text *string `json:"text"`
			}
			if json.Unmarshal(raw, &p) == nil && p.Text != nil {
				out = append(out, *p.Text)
				continue
			}
			out = append(out, "\x1d"+p.Type+":"+digest(raw))
		}
		*c = out
	}
	return nil
}

// textInput is the prompt of a completions request or the input of an
// embeddings request: a string, a list of strings, or lists of token ids.
type textInput []string

func (t *textInput) UnmarshalJSON(b []byte) error {
	switch firstByte(b) {
	case '"':
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		*t = textInput{s}
	case '[':
		var items []json.RawMessage
		if err := json.Unmarshal(b, &items); err != nil {
			return err
		}
		out := make(textInput, 0, len(items))
		for _, raw := range items {
			var s string
			if firstByte(raw) == '"' && json.Unmarshal(raw, &s) == nil {
				out = append(out, s)
				continue
			}
			// Token ids: the digits are the text.
			out = append(out, string(raw))
		}
		*t = out
	}
	return nil
}

func firstByte(b []byte) byte {
	b = bytes.TrimLeft(b, " \t\r\n")
	if len(b) == 0 {
		return 0
	}
	return b[0]
}

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// inspected is what the gateway learned from a request body, plus the body to
// send upstream.
type inspected struct {
	model  string
	stream bool
	// stripUsage is set when the gateway turned on usage reporting itself, so
	// the extra usage chunk must not reach the client.
	stripUsage  bool
	maxTokens   uint32
	promptChars uint32
	hashes      []uint64
	body        []byte
}

const (
	fieldSep  = "\x1f"
	recordSep = "\x1e"
)

// inspect reads the fields the record needs and hashes the request text. A
// body it cannot parse is forwarded untouched: rejecting it is the engine's
// job, and the gateway must never be the reason a request fails.
func inspect(ep endpoint, body []byte, hasher *prefixhash.Hasher) inspected {
	in := inspected{body: body}
	var shape requestShape
	if err := json.Unmarshal(body, &shape); err != nil {
		// A field of the wrong type still leaves the others filled in.
		if _, wrongType := err.(*json.UnmarshalTypeError); !wrongType {
			return in
		}
	}
	in.model = shape.Model
	in.maxTokens = clampUint32(shape.MaxCompletionTokens, shape.MaxTokens)

	h := hasher.Start()
	switch ep {
	case endpointChat:
		if len(shape.Tools) > 0 {
			h.WriteString(string(shape.Tools))
			h.WriteString(recordSep)
		}
		for _, m := range shape.Messages {
			h.WriteString(m.Role)
			h.WriteString(fieldSep)
			for _, piece := range m.Content {
				h.WriteString(piece)
			}
			if len(m.ToolCalls) > 0 && firstByte(m.ToolCalls) == '[' {
				h.WriteString(fieldSep)
				h.WriteString(string(m.ToolCalls))
			}
			h.WriteString(recordSep)
		}
	case endpointCompletions:
		writePieces(h, shape.Prompt)
	case endpointEmbeddings:
		writePieces(h, shape.Input)
	}
	in.hashes = h.Hashes()
	in.promptChars = uint32(min(h.Chars(), 1<<32-1))

	if shape.Stream && ep != endpointEmbeddings {
		in.stream = true
		if !clientWantsUsage(shape.StreamOptions) {
			if withUsage, ok := enableUsage(body, shape.StreamOptions); ok {
				in.body = withUsage
				in.stripUsage = true
			}
		}
	}
	return in
}

func writePieces(h *prefixhash.State, pieces []string) {
	for i, p := range pieces {
		if i > 0 {
			h.WriteString(recordSep)
		}
		h.WriteString(p)
	}
}

func clampUint32(candidates ...*int64) uint32 {
	for _, c := range candidates {
		if c != nil && *c > 0 {
			return uint32(min(*c, 1<<32-1))
		}
	}
	return 0
}

func clientWantsUsage(streamOptions json.RawMessage) bool {
	var o struct {
		IncludeUsage bool `json:"include_usage"`
	}
	return len(streamOptions) > 0 && json.Unmarshal(streamOptions, &o) == nil && o.IncludeUsage
}

const usageOption = `"stream_options":{"include_usage":true}`

// enableUsage returns body with stream_options.include_usage set to true.
// Without it a streaming response carries no token counts.
func enableUsage(body []byte, streamOptions json.RawMessage) ([]byte, bool) {
	if len(streamOptions) == 0 {
		// No stream_options key: add one before the closing brace and leave
		// every other byte of the client's body as it was.
		end := bytes.LastIndexByte(body, '}')
		if end < 0 {
			return nil, false
		}
		out := make([]byte, 0, len(body)+len(usageOption)+1)
		out = append(out, body[:end]...)
		out = append(out, ',')
		out = append(out, usageOption...)
		out = append(out, body[end:]...)
		return out, true
	}

	// The client sent stream_options without include_usage: keep its other
	// options and set ours.
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return nil, false
	}
	opts := map[string]json.RawMessage{}
	if firstByte(streamOptions) == '{' {
		if err := json.Unmarshal(streamOptions, &opts); err != nil {
			return nil, false
		}
	}
	opts["include_usage"] = json.RawMessage("true")
	merged, err := json.Marshal(opts)
	if err != nil {
		return nil, false
	}
	top["stream_options"] = merged
	out, err := json.Marshal(top)
	if err != nil {
		return nil, false
	}
	return out, true
}

// usage is the token accounting of an OpenAI-style response.
type usage struct {
	PromptTokens        uint32 `json:"prompt_tokens"`
	CompletionTokens    uint32 `json:"completion_tokens"`
	PromptTokensDetails *struct {
		CachedTokens uint32 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (u *usage) cached() uint32 {
	if u.PromptTokensDetails == nil {
		return 0
	}
	return u.PromptTokensDetails.CachedTokens
}

// usageOf reads the usage object of a complete, non-streaming response body.
func usageOf(body []byte) *usage {
	var r struct {
		Usage *usage `json:"usage"`
	}
	_ = json.Unmarshal(body, &r)
	return r.Usage
}
