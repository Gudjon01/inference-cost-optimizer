// Package config loads the part of the shared YAML file the gateway uses.
// The collector and the replay lab read the same file; keys the gateway does
// not know are ignored.
package config

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Replicas []Replica `yaml:"replicas"`
	Headers  Headers   `yaml:"headers"`
	Gateway  Gateway   `yaml:"gateway"`
}

type Replica struct {
	Name     string `yaml:"name"`
	URL      string `yaml:"url"`
	Host     string `yaml:"host"`
	GPUModel string `yaml:"gpu_model"`
	GPUCount int    `yaml:"gpu_count"`
	// Models restricts the replica to these model names. Empty means it
	// accepts any model.
	Models []string `yaml:"models"`
}

type Headers struct {
	Tenant  string `yaml:"tenant"`
	Feature string `yaml:"feature"`
}

type Gateway struct {
	ID          string `yaml:"id"`
	Listen      string `yaml:"listen"`
	AdminListen string `yaml:"admin_listen"`
	// Collector is the collector's gRPC address. Empty disables export.
	Collector string `yaml:"collector"`
	// StateDir holds the per-install hash salt.
	StateDir        string        `yaml:"state_dir"`
	MaxBodyBytes    int64         `yaml:"max_body_bytes"`
	ShutdownTimeout time.Duration `yaml:"shutdown_timeout"`
	PrefixHash      PrefixHash    `yaml:"prefix_hash"`
}

type PrefixHash struct {
	BlockChars int `yaml:"block_chars"`
	MaxBlocks  int `yaml:"max_blocks"`
}

func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

func Parse(raw []byte) (*Config, error) {
	var c Config
	if err := yaml.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	c.applyDefaults()
	if err := c.validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

func (c *Config) applyDefaults() {
	if c.Headers.Tenant == "" {
		c.Headers.Tenant = "X-Tenant-Id"
	}
	if c.Headers.Feature == "" {
		c.Headers.Feature = "X-Feature"
	}
	g := &c.Gateway
	if g.ID == "" {
		g.ID, _ = os.Hostname()
	}
	if g.Listen == "" {
		g.Listen = ":8080"
	}
	if g.AdminListen == "" {
		g.AdminListen = ":8081"
	}
	if g.StateDir == "" {
		g.StateDir = "/var/lib/ico"
	}
	if g.MaxBodyBytes == 0 {
		g.MaxBodyBytes = 32 << 20
	}
	if g.ShutdownTimeout == 0 {
		g.ShutdownTimeout = 30 * time.Second
	}
	if g.PrefixHash.BlockChars == 0 {
		g.PrefixHash.BlockChars = 256
	}
	if g.PrefixHash.MaxBlocks == 0 {
		g.PrefixHash.MaxBlocks = 64
	}
}

func (c *Config) validate() error {
	if len(c.Replicas) == 0 {
		return errors.New("config: at least one replica is required")
	}
	seen := map[string]bool{}
	for i, r := range c.Replicas {
		if r.Name == "" {
			return fmt.Errorf("config: replicas[%d] has no name", i)
		}
		if seen[r.Name] {
			return fmt.Errorf("config: duplicate replica name %q", r.Name)
		}
		seen[r.Name] = true
		u, err := url.Parse(r.URL)
		if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
			return fmt.Errorf("config: replica %q needs an http(s) url, got %q", r.Name, r.URL)
		}
	}
	if c.Gateway.PrefixHash.BlockChars < 1 || c.Gateway.PrefixHash.MaxBlocks < 1 {
		return errors.New("config: gateway.prefix_hash values must be positive")
	}
	return nil
}
