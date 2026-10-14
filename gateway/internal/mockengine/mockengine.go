// Package mockengine is a stand-in for a vLLM replica: the OpenAI-compatible
// endpoints the gateway proxies, with the response shapes vLLM produces, and
// a /metrics page in vLLM's naming. It generates filler tokens on a timer, so
// the gateway and the collector can be developed and tested without a GPU.
package mockengine

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Engine struct {
	// Model is the name reported in responses and metrics.
	Model string
	// TTFT is the wait before the first token.
	TTFT time.Duration
	// TokenDelay is the wait between tokens.
	TokenDelay time.Duration
	// Tokens is how many tokens a request gets when it sets no max_tokens.
	Tokens int
	// MaxNumSeqs is how many requests generate at once; the rest wait their
	// turn, which is what makes an overloaded engine slow to its first
	// token. Zero means no limit.
	MaxNumSeqs int
	// Seen, when set, is called with every inference request and its body.
	Seen func(r *http.Request, body []byte)

	slots     chan struct{}
	slotsOnce sync.Once

	running          atomic.Int64
	requests         atomic.Uint64
	promptTokens     atomic.Uint64
	generationTokens atomic.Uint64
}

type request struct {
	// A prompt may be text or, as the replay lab sends it, a list of token ids.
	Prompt              json.RawMessage `json:"prompt"`
	Stream              bool            `json:"stream"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	StreamOptions       *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

func (e *Engine) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", e.generate(true))
	mux.HandleFunc("POST /v1/completions", e.generate(false))
	mux.HandleFunc("POST /v1/embeddings", e.embed)
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"object": "list", "data": []any{
			map[string]any{"id": e.model(), "object": "model", "owned_by": "mockengine"},
		}})
	})
	mux.HandleFunc("GET /health", func(http.ResponseWriter, *http.Request) {})
	mux.HandleFunc("POST /tokenize", e.tokenize)
	mux.HandleFunc("GET /metrics", e.metrics)
	return mux
}

func (e *Engine) model() string {
	if e.Model == "" {
		return "mock-model"
	}
	return e.Model
}

func (e *Engine) generate(chat bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if e.Seen != nil {
			e.Seen(r, body)
		}
		var req request
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, `{"error":{"message":"invalid JSON","type":"BadRequestError","code":400}}`, http.StatusBadRequest)
			return
		}

		if !e.acquire(r) {
			return
		}
		defer e.release()
		e.running.Add(1)
		defer e.running.Add(-1)

		prompt := len(body) / 4
		var ids []int
		if json.Unmarshal(req.Prompt, &ids) == nil && len(ids) > 0 {
			prompt = len(ids)
		}
		completion := e.Tokens
		if completion == 0 {
			completion = 16
		}
		for _, limit := range []*int{req.MaxCompletionTokens, req.MaxTokens} {
			if limit != nil && *limit < completion {
				completion = *limit
			}
		}
		id := r.Header.Get("X-Request-Id")
		if id == "" {
			id = fmt.Sprintf("%016x", e.requests.Load()+1)
		}
		id, object := "cmpl-"+id, "text_completion"
		if chat {
			id, object = "chat"+id, "chat.completion"
		}
		created := time.Now().Unix()
		usage := map[string]any{"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}

		defer func() {
			e.requests.Add(1)
			e.promptTokens.Add(uint64(prompt))
			e.generationTokens.Add(uint64(completion))
		}()

		if !req.Stream {
			if !sleep(r, e.TTFT+time.Duration(completion)*e.TokenDelay) {
				return
			}
			text := strings.Repeat(" tok", completion)
			choice := map[string]any{"index": 0, "logprobs": nil, "finish_reason": "length"}
			if chat {
				choice["message"] = map[string]any{"role": "assistant", "content": text}
			} else {
				choice["text"] = text
			}
			writeJSON(w, map[string]any{
				"id": id, "object": object, "created": created, "model": e.model(),
				"choices": []any{choice}, "usage": usage,
			})
			return
		}

		if chat {
			object = "chat.completion.chunk"
		}
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, _ := w.(http.Flusher)
		send := func(choices []any, extra map[string]any) {
			chunk := map[string]any{"id": id, "object": object, "created": created, "model": e.model(), "choices": choices}
			for k, v := range extra {
				chunk[k] = v
			}
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
		choice := func(text string, finish any, role bool) []any {
			c := map[string]any{"index": 0, "logprobs": nil, "finish_reason": finish}
			switch {
			case !chat:
				c["text"] = text
			case role:
				c["delta"] = map[string]any{"role": "assistant", "content": ""}
			default:
				c["delta"] = map[string]any{"content": text}
			}
			return []any{c}
		}

		if !sleep(r, e.TTFT) {
			return
		}
		if chat {
			// vLLM opens a chat stream with a role-only chunk.
			send(choice("", nil, true), nil)
		}
		for i := 0; i < completion; i++ {
			if i > 0 && !sleep(r, e.TokenDelay) {
				return
			}
			send(choice(" tok", nil, false), nil)
		}
		send(choice("", "length", false), nil)
		if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
			send([]any{}, map[string]any{"usage": usage})
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}
}

// acquire waits for a free generation slot. It reports false if the client
// went away first.
func (e *Engine) acquire(r *http.Request) bool {
	if e.MaxNumSeqs <= 0 {
		return true
	}
	e.slotsOnce.Do(func() { e.slots = make(chan struct{}, e.MaxNumSeqs) })
	select {
	case e.slots <- struct{}{}:
		return true
	case <-r.Context().Done():
		return false
	}
}

func (e *Engine) release() {
	if e.slots != nil {
		<-e.slots
	}
}

// tokenize answers like vLLM's /tokenize with a made-up tokenizer: one token
// per word, the same word always the same id.
func (e *Engine) tokenize(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Prompt string `json:"prompt"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":{"message":"invalid JSON","type":"BadRequestError","code":400}}`, http.StatusBadRequest)
		return
	}
	words := strings.Fields(req.Prompt)
	tokens := make([]int, len(words))
	for i, word := range words {
		h := fnv.New32a()
		h.Write([]byte(word))
		tokens[i] = 1000 + int(h.Sum32()%30000)
	}
	writeJSON(w, map[string]any{"count": len(tokens), "max_model_len": 32768, "tokens": tokens})
}

func (e *Engine) embed(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	if e.Seen != nil {
		e.Seen(r, body)
	}
	prompt := len(body) / 4
	e.requests.Add(1)
	e.promptTokens.Add(uint64(prompt))
	writeJSON(w, map[string]any{
		"object": "list",
		"model":  e.model(),
		"data":   []any{map[string]any{"object": "embedding", "index": 0, "embedding": []float64{0.25, -0.5, 0.125}}},
		"usage":  map[string]any{"prompt_tokens": prompt, "total_tokens": prompt},
	})
}

// metrics mimics the counters and gauges of vLLM's Prometheus page.
func (e *Engine) metrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	label := fmt.Sprintf(`{model_name=%q}`, e.model())
	fmt.Fprintf(w, "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running%s %d\n", label, e.running.Load())
	fmt.Fprintf(w, "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting%s 0\n", label)
	fmt.Fprintf(w, "# TYPE vllm:prompt_tokens_total counter\nvllm:prompt_tokens_total%s %d\n", label, e.promptTokens.Load())
	fmt.Fprintf(w, "# TYPE vllm:generation_tokens_total counter\nvllm:generation_tokens_total%s %d\n", label, e.generationTokens.Load())
	fmt.Fprintf(w, "# TYPE vllm:request_success_total counter\nvllm:request_success_total{finished_reason=\"length\",model_name=%q} %d\n", e.model(), e.requests.Load())
}

// sleep waits for d and reports false if the client went away first.
func sleep(r *http.Request, d time.Duration) bool {
	if d <= 0 {
		return r.Context().Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-r.Context().Done():
		return false
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
