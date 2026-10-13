// Command gateway is the OpenAI-compatible pass-through that sits in front of
// the engine replicas and reports one telemetry record per request.
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/admin"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/config"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/export"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/prefixhash"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/proxy"
	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/replica"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	if err := run(logger); err != nil {
		logger.Error("gateway stopped", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger) error {
	configPath := flag.String("config", envOr("ICO_CONFIG", "/etc/ico/config.yaml"), "path to the YAML config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}
	pool, err := replica.NewPool(cfg.Replicas)
	if err != nil {
		return err
	}
	salt, err := hashSalt(cfg.Gateway.StateDir, logger)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var sink proxy.Sink = export.Discard{}
	stats := func() any { return map[string]string{"export": "disabled"} }
	var exporting sync.WaitGroup
	exportCtx, stopExport := context.WithCancel(context.Background())
	defer stopExport()
	if cfg.Gateway.Collector == "" {
		logger.Warn("no collector configured; requests are proxied but nothing is recorded")
	} else {
		exporter, err := export.New(cfg.Gateway.Collector, cfg.Gateway.ID, logger)
		if err != nil {
			return fmt.Errorf("collector address %q: %w", cfg.Gateway.Collector, err)
		}
		sink = exporter
		stats = func() any { return exporter.Stats() }
		exporting.Add(1)
		go func() {
			defer exporting.Done()
			exporter.Run(exportCtx)
		}()
	}

	adminToken := os.Getenv("ICO_ADMIN_TOKEN")
	if adminToken == "" {
		logger.Warn("ICO_ADMIN_TOKEN is not set; the admin API refuses every call")
	}

	public := &http.Server{
		Addr: cfg.Gateway.Listen,
		Handler: proxy.New(proxy.Options{
			Pool:          pool,
			Hasher:        prefixhash.New(salt, cfg.Gateway.PrefixHash.BlockChars, cfg.Gateway.PrefixHash.MaxBlocks),
			Sink:          sink,
			TenantHeader:  cfg.Headers.Tenant,
			FeatureHeader: cfg.Headers.Feature,
			MaxBodyBytes:  cfg.Gateway.MaxBodyBytes,
			Logger:        logger,
		}),
		ReadHeaderTimeout: 10 * time.Second,
	}
	control := &http.Server{
		Addr:              cfg.Gateway.AdminListen,
		Handler:           admin.Handler(pool, adminToken, stats),
		ReadHeaderTimeout: 10 * time.Second,
	}

	failed := make(chan error, 2)
	for _, srv := range []*http.Server{public, control} {
		go func() {
			if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
				failed <- fmt.Errorf("listen on %s: %w", srv.Addr, err)
			}
		}()
	}
	logger.Info("gateway started", "listen", public.Addr, "admin_listen", control.Addr,
		"replicas", len(pool.All()), "collector", cfg.Gateway.Collector)

	select {
	case err = <-failed:
	case <-ctx.Done():
		logger.Info("shutting down; waiting for in-flight requests", "timeout", cfg.Gateway.ShutdownTimeout.String())
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.Gateway.ShutdownTimeout)
	defer cancel()
	_ = control.Shutdown(shutdownCtx)
	if serr := public.Shutdown(shutdownCtx); serr != nil {
		logger.Warn("some requests were still running at shutdown", "error", serr)
	}
	// Records of the last requests are queued by now; let the exporter flush.
	stopExport()
	exporting.Wait()
	return err
}

// hashSalt returns the per-install salt for prefix hashes. ICO_HASH_SALT wins;
// otherwise the salt is read from the state directory, and created there on
// first start.
func hashSalt(stateDir string, logger *slog.Logger) ([]byte, error) {
	if s := os.Getenv("ICO_HASH_SALT"); s != "" {
		return []byte(s), nil
	}
	path := filepath.Join(stateDir, "hash-salt")
	if raw, err := os.ReadFile(path); err == nil && len(strings.TrimSpace(string(raw))) >= 32 {
		return []byte(strings.TrimSpace(string(raw))), nil
	}

	fresh := make([]byte, 32)
	if _, err := rand.Read(fresh); err != nil {
		return nil, err
	}
	salt := []byte(hex.EncodeToString(fresh))
	if err := os.MkdirAll(stateDir, 0o700); err == nil {
		err = os.WriteFile(path, salt, 0o600)
		if err == nil {
			return salt, nil
		}
	}
	logger.Warn("could not persist the hash salt; prefix hashes will not match across restarts",
		"state_dir", stateDir, "fix", "mount a writable state_dir or set ICO_HASH_SALT")
	return salt, nil
}

func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
