// Package export ships request records to the collector over gRPC. It sits
// off the request path: Emit never blocks, and when the collector is slow or
// down records are dropped and counted instead of holding anything up.
package export

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/telemetrypb"
)

const (
	queueSize   = 16384
	batchSize   = 512
	flushEvery  = time.Second
	callTimeout = 5 * time.Second
	minBackoff  = time.Second
	maxBackoff  = 30 * time.Second
)

// Stats counts what happened to the records handed to Emit.
type Stats struct {
	Exported uint64 `json:"exported"`
	Dropped  uint64 `json:"dropped"`
	Queued   int    `json:"queued"`
	// FailedCalls counts Export calls the collector did not acknowledge.
	FailedCalls uint64 `json:"failed_calls"`
}

type Exporter struct {
	gatewayID string
	logger    *slog.Logger
	conn      *grpc.ClientConn
	client    telemetrypb.TelemetryClient
	queue     chan *telemetrypb.RequestRecord

	exported    atomic.Uint64
	dropped     atomic.Uint64
	failedCalls atomic.Uint64
}

// New prepares an exporter for the collector at addr. It does not connect:
// the collector may come up later, or never.
func New(addr, gatewayID string, logger *slog.Logger) (*Exporter, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &Exporter{
		gatewayID: gatewayID,
		logger:    logger,
		conn:      conn,
		client:    telemetrypb.NewTelemetryClient(conn),
		queue:     make(chan *telemetrypb.RequestRecord, queueSize),
	}, nil
}

// Emit queues a record. A full queue drops it.
func (e *Exporter) Emit(rec *telemetrypb.RequestRecord) {
	select {
	case e.queue <- rec:
	default:
		e.dropped.Add(1)
	}
}

func (e *Exporter) Stats() Stats {
	return Stats{
		Exported:    e.exported.Load(),
		Dropped:     e.dropped.Load(),
		Queued:      len(e.queue),
		FailedCalls: e.failedCalls.Load(),
	}
}

// Run sends batches until ctx ends, then makes one last attempt to flush.
func (e *Exporter) Run(ctx context.Context) {
	defer e.conn.Close()
	ticker := time.NewTicker(flushEvery)
	defer ticker.Stop()

	batch := make([]*telemetrypb.RequestRecord, 0, batchSize)
	var backoff time.Duration
	var retryAt time.Time
	for {
		// A full batch that failed is held for retry. Reading stops, so the
		// queue absorbs what it can and Emit drops the rest.
		in := e.queue
		if len(batch) >= batchSize {
			in = nil
		}
		select {
		case <-ctx.Done():
			e.flush(batch)
			return
		case rec := <-in:
			batch = append(batch, rec)
			if len(batch) < batchSize {
				continue
			}
		case <-ticker.C:
		}
		if len(batch) == 0 || time.Now().Before(retryAt) {
			continue
		}

		if err := e.send(ctx, batch); err != nil {
			e.failedCalls.Add(1)
			if backoff == 0 {
				e.logger.Warn("collector unreachable; requests keep flowing, telemetry is dropped once the queue fills", "error", err)
			}
			backoff = min(max(2*backoff, minBackoff), maxBackoff)
			retryAt = time.Now().Add(backoff)
			continue
		}
		if backoff > 0 {
			e.logger.Info("collector reachable again")
			backoff, retryAt = 0, time.Time{}
		}
		e.exported.Add(uint64(len(batch)))
		clear(batch)
		batch = batch[:0]
	}
}

func (e *Exporter) send(ctx context.Context, batch []*telemetrypb.RequestRecord) error {
	ctx, cancel := context.WithTimeout(ctx, callTimeout)
	defer cancel()
	_, err := e.client.Export(ctx, &telemetrypb.ExportRequest{GatewayId: e.gatewayID, Records: batch})
	return err
}

// flush sends what is still queued at shutdown, giving up on the first error.
func (e *Exporter) flush(batch []*telemetrypb.RequestRecord) {
	ctx, cancel := context.WithTimeout(context.Background(), callTimeout)
	defer cancel()
	for {
		for len(batch) < batchSize {
			select {
			case rec := <-e.queue:
				batch = append(batch, rec)
				continue
			default:
			}
			break
		}
		if len(batch) == 0 {
			return
		}
		if err := e.send(ctx, batch); err != nil {
			e.dropped.Add(uint64(len(batch) + len(e.queue)))
			return
		}
		e.exported.Add(uint64(len(batch)))
		batch = batch[:0]
	}
}

// Discard is the sink used when no collector is configured.
type Discard struct{}

func (Discard) Emit(*telemetrypb.RequestRecord) {}
