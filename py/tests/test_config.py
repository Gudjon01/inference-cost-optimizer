from pathlib import Path

import pytest

from ico import config

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"


def test_example_config_loads(monkeypatch):
    monkeypatch.delenv("ICO_CLICKHOUSE_PASSWORD", raising=False)
    cfg = config.load(DEPLOY / "config.example.yaml")
    [replica] = cfg.replicas
    assert (replica.name, replica.url, replica.gpu_model, replica.gpu_count) == ("r1", "http://vllm:8000", "L4", 1)
    assert cfg.gpu_price_per_hour == {"L4": 0.50}
    model = cfg.models["Qwen/Qwen2.5-0.5B-Instruct"]
    assert (model.params, model.weight_bytes_per_param, model.kv_bytes_per_token) == (494_000_000, 2, 12_288)
    assert (cfg.slo.ttft_p95_ms, cfg.slo.itl_p95_ms) == (500, 50)
    assert cfg.collector.grpc_listen == "0.0.0.0:4317" and cfg.collector.scrape_interval_s == 5
    [exporter] = cfg.collector.gpu_exporters
    assert (exporter.kind, exporter.url, exporter.host) == ("dcgm", "http://dcgm-exporter:9400/metrics", "gpu-1")
    assert (cfg.clickhouse.host, cfg.clickhouse.username, cfg.clickhouse.database) == ("clickhouse", "ico", "ico")
    assert cfg.clickhouse.password == ""


def test_mock_config_loads():
    cfg = config.load(DEPLOY / "config.mock.yaml")
    assert [r.name for r in cfg.replicas] == ["mock-1", "mock-2"]
    assert cfg.collector.gpu_exporters == ()


def test_password_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ICO_CLICKHOUSE_PASSWORD", "from-env")
    assert config.load(DEPLOY / "config.example.yaml").clickhouse.password == "from-env"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({}, "at least one replica"),
        ({"replicas": [{"name": "a", "url": "http://x"}, {"name": "a", "url": "http://y"}]}, "unique"),
        ({"replicas": [{"url": "http://x"}]}, r"replicas\[0\]"),
        (
            {"replicas": [{"name": "a", "url": "http://x"}], "collector": {"gpu_exporters": [{"kind": "amd", "url": "u", "host": "h"}]}},
            "unknown kind 'amd'",
        ),
    ],
)
def test_bad_configs_are_rejected_with_the_section_named(raw, message):
    with pytest.raises(config.ConfigError, match=message):
        config.parse(raw)
