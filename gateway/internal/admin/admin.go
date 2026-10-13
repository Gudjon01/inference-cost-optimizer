// Package admin is the gateway's control API: list replicas, drain one for a
// replay, put it back. It listens on its own port and needs a bearer token.
package admin

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/replica"
)

type replicaView struct {
	Name     string `json:"name"`
	URL      string `json:"url"`
	Host     string `json:"host"`
	GPUModel string `json:"gpu_model"`
	GPUCount int    `json:"gpu_count"`
	Drained  bool   `json:"drained"`
	// Inflight is the number of requests the replica is still answering. A
	// drained replica is idle once it reaches zero.
	Inflight int64 `json:"inflight"`
}

func view(r *replica.Replica) replicaView {
	return replicaView{
		Name:     r.Name,
		URL:      r.URL.String(),
		Host:     r.Host,
		GPUModel: r.GPUModel,
		GPUCount: r.GPUCount,
		Drained:  r.Drained(),
		Inflight: r.Inflight(),
	}
}

// Handler serves the admin API. stats is included in GET /admin/stats.
func Handler(pool *replica.Pool, token string, stats func() any) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /admin/replicas", func(w http.ResponseWriter, _ *http.Request) {
		views := make([]replicaView, 0, len(pool.All()))
		for _, r := range pool.All() {
			views = append(views, view(r))
		}
		writeJSON(w, http.StatusOK, map[string]any{"replicas": views})
	})
	setDrained := func(drained bool) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			rep, ok := pool.SetDrained(r.PathValue("name"), drained)
			if !ok {
				writeJSON(w, http.StatusNotFound, map[string]string{"error": "no replica named " + r.PathValue("name")})
				return
			}
			writeJSON(w, http.StatusOK, view(rep))
		}
	}
	mux.HandleFunc("POST /admin/replicas/{name}/drain", setDrained(true))
	mux.HandleFunc("POST /admin/replicas/{name}/undrain", setDrained(false))
	mux.HandleFunc("GET /admin/stats", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, stats())
	})
	return requireToken(token, mux)
}

func requireToken(token string, next http.Handler) http.Handler {
	want := sha256.Sum256([]byte(token))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if token == "" {
			writeJSON(w, http.StatusForbidden, map[string]string{"error": "the admin API is off until ICO_ADMIN_TOKEN is set"})
			return
		}
		given, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		got := sha256.Sum256([]byte(given))
		if !ok || subtle.ConstantTimeCompare(got[:], want[:]) != 1 {
			w.Header().Set("WWW-Authenticate", "Bearer")
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "missing or wrong bearer token"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
