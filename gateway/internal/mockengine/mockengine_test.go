package mockengine

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func post(t *testing.T, url, body string) string {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, out)
	}
	return string(out)
}

func TestTokenIDPromptsCountTheirTokens(t *testing.T) {
	srv := httptest.NewServer((&Engine{}).Handler())
	defer srv.Close()
	out := post(t, srv.URL+"/v1/completions", `{"model":"m","prompt":[11,12,13,14,15,16,17],"max_tokens":3}`)
	var got struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil || got.Usage.PromptTokens != 7 || got.Usage.CompletionTokens != 3 {
		t.Errorf("usage for a 7-token prompt asking for 3 tokens: %s", out)
	}
}

func TestTokenizeGivesTheSameIDForTheSameWord(t *testing.T) {
	srv := httptest.NewServer((&Engine{}).Handler())
	defer srv.Close()
	out := post(t, srv.URL+"/tokenize", `{"model":"m","prompt":"the cat saw the dog"}`)
	var got struct {
		Count  int   `json:"count"`
		Tokens []int `json:"tokens"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil || got.Count != 5 || len(got.Tokens) != 5 {
		t.Fatalf("tokenize answered %s", out)
	}
	if got.Tokens[0] != got.Tokens[3] || got.Tokens[1] == got.Tokens[4] {
		t.Errorf("tokens %v: \"the\" twice must match, \"cat\" and \"dog\" must not", got.Tokens)
	}
}

func TestMaxNumSeqsMakesTheRestWait(t *testing.T) {
	// One slot and 60 ms per request: four requests at once take about four times as long as one.
	srv := httptest.NewServer((&Engine{MaxNumSeqs: 1, TTFT: 60 * time.Millisecond}).Handler())
	defer srv.Close()
	start := time.Now()
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			post(t, srv.URL+"/v1/completions", `{"model":"m","prompt":"hi","max_tokens":1}`)
		}()
	}
	wg.Wait()
	if took := time.Since(start); took < 220*time.Millisecond {
		t.Errorf("four requests through one slot took %v, want about 240 ms", took)
	}
}
