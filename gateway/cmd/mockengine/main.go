// Command mockengine serves the fake vLLM replica used for GPU-free
// development: the Compose "mock" profile runs it behind the gateway.
package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/mockengine"
)

func main() {
	listen := flag.String("listen", ":8000", "address to listen on")
	model := flag.String("model", "mock-model", "model name to report")
	ttft := flag.Duration("ttft", 80*time.Millisecond, "wait before the first token")
	tokenDelay := flag.Duration("token-delay", 15*time.Millisecond, "wait between tokens")
	tokens := flag.Int("tokens", 64, "tokens generated when a request sets no max_tokens")
	// Named like the vLLM flag, so a replay sweep can change it on the fake engine too.
	maxNumSeqs := flag.Int("max-num-seqs", 0, "requests generating at once; the rest wait (0 = no limit)")
	flag.Parse()

	engine := &mockengine.Engine{Model: *model, TTFT: *ttft, TokenDelay: *tokenDelay, Tokens: *tokens, MaxNumSeqs: *maxNumSeqs}
	log.Printf("mockengine listening on %s as %q", *listen, *model)
	log.Fatal(http.ListenAndServe(*listen, engine.Handler()))
}
