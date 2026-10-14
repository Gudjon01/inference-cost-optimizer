package admin

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/config"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/replica"
)

func newPool(t *testing.T) *replica.Pool {
	t.Helper()
	pool, err := replica.NewPool([]config.Replica{
		{Name: "r1", URL: "http://10.0.0.1:8000", Host: "gpu-1", GPUModel: "L4", GPUCount: 1},
		{Name: "r2", URL: "http://10.0.0.2:8000", Host: "gpu-2", GPUModel: "L4", GPUCount: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func call(h http.Handler, method, path, token string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, nil)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

func TestTokenIsRequired(t *testing.T) {
	h := Handler(newPool(t), "s3cret", func() any { return nil })
	for _, token := range []string{"", "wrong", "s3cre", "s3cret-and-more"} {
		if w := call(h, "GET", "/admin/replicas", token); w.Code != http.StatusUnauthorized {
			t.Errorf("token %q: status %d, want 401", token, w.Code)
		}
	}
	if w := call(h, "GET", "/admin/replicas", "s3cret"); w.Code != http.StatusOK {
		t.Errorf("right token: status %d, want 200", w.Code)
	}
}

func TestAdminIsOffWithoutAConfiguredToken(t *testing.T) {
	pool := newPool(t)
	h := Handler(pool, "", func() any { return nil })
	// An empty configured token must not be matched by an empty bearer token.
	for _, token := range []string{"", "anything"} {
		if w := call(h, "POST", "/admin/replicas/r1/drain", token); w.Code != http.StatusForbidden {
			t.Errorf("token %q: status %d, want 403", token, w.Code)
		}
	}
	req := httptest.NewRequest("POST", "/admin/replicas/r1/drain", nil)
	req.Header.Set("Authorization", "Bearer ")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if r1, _ := pool.Get("r1"); w.Code != http.StatusForbidden || r1.Drained() {
		t.Errorf("empty bearer token: status %d, drained=%v", w.Code, r1.Drained())
	}
}

func TestDrainAndUndrain(t *testing.T) {
	pool := newPool(t)
	h := Handler(pool, "tok", func() any { return map[string]int{"exported": 3} })

	type answer struct {
		replicaView
		WasDrained bool `json:"was_drained"`
	}
	w := call(h, "POST", "/admin/replicas/r2/drain", "tok")
	var got answer
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil || w.Code != 200 {
		t.Fatalf("drain: status %d body %s", w.Code, w.Body)
	}
	if !got.Drained || got.Name != "r2" || got.Inflight != 0 || got.WasDrained {
		t.Errorf("drain answered %+v", got)
	}
	// A second drain succeeds too, but is told it was not the first: that is
	// how two replay runs find out which of them holds the replica.
	w = call(h, "POST", "/admin/replicas/r2/drain", "tok")
	var again answer
	if err := json.Unmarshal(w.Body.Bytes(), &again); err != nil || !again.Drained || !again.WasDrained {
		t.Errorf("second drain answered %s", w.Body)
	}
	// Traffic now avoids r2.
	for i := 0; i < 4; i++ {
		r := pool.Acquire("", nil)
		if r == nil || r.Name != "r1" {
			t.Fatalf("acquired %v while r2 is drained", r)
		}
		r.Release()
	}

	var list struct {
		Replicas []replicaView `json:"replicas"`
	}
	w = call(h, "GET", "/admin/replicas", "tok")
	if err := json.Unmarshal(w.Body.Bytes(), &list); err != nil || len(list.Replicas) != 2 {
		t.Fatalf("list: %s", w.Body)
	}
	if list.Replicas[0].Drained || !list.Replicas[1].Drained || list.Replicas[1].GPUModel != "L4" {
		t.Errorf("list: %+v", list.Replicas)
	}

	if w = call(h, "POST", "/admin/replicas/r2/undrain", "tok"); w.Code != 200 {
		t.Fatalf("undrain: status %d", w.Code)
	}
	if r2, _ := pool.Get("r2"); r2.Drained() {
		t.Error("r2 still drained after undrain")
	}

	if w = call(h, "POST", "/admin/replicas/nope/drain", "tok"); w.Code != http.StatusNotFound {
		t.Errorf("unknown replica: status %d, want 404", w.Code)
	}
	if w = call(h, "GET", "/admin/stats", "tok"); w.Body.String() != "{\"exported\":3}\n" {
		t.Errorf("stats: %s", w.Body)
	}
}
