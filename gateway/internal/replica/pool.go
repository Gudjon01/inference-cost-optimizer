// Package replica tracks the engine replicas the gateway balances across.
package replica

import (
	"fmt"
	"net/url"
	"sync/atomic"

	"github.com/Gudjon01/inference-cost-optimizer/gateway/internal/config"
)

type Replica struct {
	Name     string
	URL      *url.URL
	Host     string
	GPUModel string
	GPUCount int

	models   map[string]struct{}
	drained  atomic.Bool
	inflight atomic.Int64
}

func (r *Replica) Drained() bool   { return r.drained.Load() }
func (r *Replica) Inflight() int64 { return r.inflight.Load() }

// Release ends a request started by Pool.Acquire.
func (r *Replica) Release() { r.inflight.Add(-1) }

// serves reports whether the replica takes requests for model. A replica
// with no model list takes everything, and a request that names no model can
// go anywhere.
func (r *Replica) serves(model string) bool {
	if len(r.models) == 0 || model == "" {
		return true
	}
	_, ok := r.models[model]
	return ok
}

type Pool struct {
	replicas []*Replica
	byName   map[string]*Replica
	next     atomic.Uint64
}

func NewPool(cfg []config.Replica) (*Pool, error) {
	p := &Pool{byName: make(map[string]*Replica, len(cfg))}
	for _, c := range cfg {
		u, err := url.Parse(c.URL)
		if err != nil {
			return nil, fmt.Errorf("replica %q: %w", c.Name, err)
		}
		r := &Replica{Name: c.Name, URL: u, Host: c.Host, GPUModel: c.GPUModel, GPUCount: c.GPUCount}
		if len(c.Models) > 0 {
			r.models = make(map[string]struct{}, len(c.Models))
			for _, m := range c.Models {
				r.models[m] = struct{}{}
			}
		}
		p.replicas = append(p.replicas, r)
		p.byName[r.Name] = r
	}
	return p, nil
}

func (p *Pool) All() []*Replica { return p.replicas }

func (p *Pool) Get(name string) (*Replica, bool) {
	r, ok := p.byName[name]
	return r, ok
}

// SetDrained flips a replica's drain flag and reports whether it exists.
func (p *Pool) SetDrained(name string, drained bool) (*Replica, bool) {
	r, ok := p.byName[name]
	if ok {
		r.drained.Store(drained)
	}
	return r, ok
}

// Acquire returns the next replica in round-robin order that is not drained,
// serves model and is not in skip, counted as having one more request in
// flight. The caller must Release it. It returns nil when none qualifies.
func (p *Pool) Acquire(model string, skip []*Replica) *Replica {
	n := uint64(len(p.replicas))
	start := p.next.Add(1) - 1
	for i := uint64(0); i < n; i++ {
		r := p.replicas[(start+i)%n]
		if r.Drained() || !r.serves(model) || contains(skip, r) {
			continue
		}
		// Count first, then look at the flag again: a drain that lands
		// between the check above and here either sees this request in
		// flight or makes it back off, so an idle drained replica stays idle.
		r.inflight.Add(1)
		if r.Drained() {
			r.inflight.Add(-1)
			continue
		}
		return r
	}
	return nil
}

// Serves reports whether any replica, drained or not, takes model. It tells
// an unknown model apart from a model whose replicas are all drained.
func (p *Pool) Serves(model string) bool {
	for _, r := range p.replicas {
		if r.serves(model) {
			return true
		}
	}
	return false
}

func contains(rs []*Replica, r *Replica) bool {
	for _, x := range rs {
		if x == r {
			return true
		}
	}
	return false
}
