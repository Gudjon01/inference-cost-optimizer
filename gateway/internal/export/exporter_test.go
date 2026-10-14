package export

import (
	"context"
	"io"
	"log/slog"
	"net"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/telemetrypb"
)

// fakeCollector is a Telemetry server that remembers what it was sent.
type fakeCollector struct {
	telemetrypb.UnimplementedTelemetryServer
	mu      sync.Mutex
	gateway string
	records []*telemetrypb.RequestRecord
}

func (c *fakeCollector) Export(_ context.Context, req *telemetrypb.ExportRequest) (*telemetrypb.ExportResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.gateway = req.GatewayId
	c.records = append(c.records, req.Records...)
	return &telemetrypb.ExportResponse{Accepted: uint32(len(req.Records))}, nil
}

func (c *fakeCollector) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.records)
}

// serve starts c on addr ("127.0.0.1:0" picks a port) and returns the address.
func serve(t *testing.T, c *fakeCollector, addr string) (string, *grpc.Server) {
	t.Helper()
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	telemetrypb.RegisterTelemetryServer(srv, c)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return lis.Addr().String(), srv
}

func quiet() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func waitFor(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestRecordsReachTheCollector(t *testing.T) {
	collector := &fakeCollector{}
	addr, _ := serve(t, collector, "127.0.0.1:0")
	e, err := New(addr, "gw-1", quiet())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { e.Run(ctx); close(done) }()

	const n = 3*batchSize + 7 // full batches go out at once, the tail on the timer
	for i := 0; i < n; i++ {
		e.Emit(&telemetrypb.RequestRecord{RequestId: "r", PrefixHashes: []uint64{uint64(i), 1 << 63}})
	}
	waitFor(t, "all records", func() bool { return collector.count() == n })
	if collector.gateway != "gw-1" {
		t.Errorf("gateway id = %q", collector.gateway)
	}
	if got := collector.records[n-1].PrefixHashes; len(got) != 2 || got[1] != 1<<63 {
		t.Errorf("prefix hashes did not survive the wire: %v", got)
	}
	if s := e.Stats(); s.Exported != n || s.Dropped != 0 {
		t.Errorf("stats: %+v", s)
	}
	cancel()
	<-done
}

func TestShutdownFlushesTheQueue(t *testing.T) {
	collector := &fakeCollector{}
	addr, _ := serve(t, collector, "127.0.0.1:0")
	e, _ := New(addr, "gw", quiet())
	for i := 0; i < 5; i++ {
		e.Emit(&telemetrypb.RequestRecord{})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // stop before the first timer tick: only the final flush can deliver
	e.Run(ctx)
	if collector.count() != 5 {
		t.Errorf("collector got %d records after shutdown, want 5", collector.count())
	}
}

func TestCollectorDownNeverBlocksEmit(t *testing.T) {
	lis, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := lis.Addr().String()
	lis.Close() // nothing listens here

	e, _ := New(addr, "gw", quiet())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go e.Run(ctx)

	// Far more than the queue holds: every Emit must still return at once.
	start := time.Now()
	for i := 0; i < 4*queueSize; i++ {
		e.Emit(&telemetrypb.RequestRecord{})
	}
	if took := time.Since(start); took > time.Second {
		t.Errorf("emitting with the collector down took %v", took)
	}
	waitFor(t, "a failed export call", func() bool { return e.Stats().FailedCalls > 0 })
	s := e.Stats()
	if s.Dropped == 0 || s.Exported != 0 {
		t.Errorf("stats: %+v; want drops and no exports", s)
	}
	if s.Queued > queueSize {
		t.Errorf("queue grew to %d, past its bound of %d", s.Queued, queueSize)
	}
}

func TestExportResumesWhenTheCollectorReturns(t *testing.T) {
	lis, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := lis.Addr().String()
	lis.Close()

	e, _ := New(addr, "gw", quiet())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go e.Run(ctx)

	e.Emit(&telemetrypb.RequestRecord{RequestId: "while-down"})
	waitFor(t, "a failed export call", func() bool { return e.Stats().FailedCalls > 0 })

	collector := &fakeCollector{}
	serve(t, collector, addr) // the collector comes up on the address the gateway knows
	e.Emit(&telemetrypb.RequestRecord{RequestId: "after-recovery"})
	waitFor(t, "records after recovery", func() bool { return collector.count() == 2 })
}
