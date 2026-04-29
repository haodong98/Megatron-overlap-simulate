from __future__ import annotations

from pathlib import Path

from kernel_simulator.config import parse_config
from kernel_simulator.replay import _build_verdict


def test_parse_old_single_case_still_works() -> None:
    config = parse_config(
        {
            "version": 1,
            "name": "old_single",
            "kernel": {
                "family": "gemm",
                "implementation": "torch_matmul",
                "shape": {"m": 8, "k": 8, "n": 8},
            },
        }
    )

    assert config.name == "old_single"
    assert config.overlap.dependency_shape == "concurrent"
    assert not config.overlap.serialization_warn_threshold_explicit
    assert config.card_id is None
    assert len(config.selected_kernels()) == 1


def test_parse_card_metadata_and_null_empty_shape() -> None:
    config = parse_config(
        {
            "version": 1,
            "name": "card",
            "card_id": "DP-1",
            "parallel_source": "dp",
            "physical_path": "nccl_sm_declared",
            "risk": ["sm_contention"],
            "sim_caveat": "synthetic",
            "requires": {"distributed": True, "min_gpus": 2},
            "expected": {"overlap_pct": [0.1, 0.9]},
            "kernels": [
                {
                    "family": "null",
                    "implementation": "empty_range",
                    "phase": "forward",
                },
                {
                    "family": "gemm",
                    "shape": {"m": 8, "k": 8, "n": 8},
                },
            ],
            "overlap": {"dependency_shape": "prefetch", "serialization_warn_threshold": 0.42},
        }
    )

    assert config.card_id == "DP-1"
    assert config.parallel_source == "dp"
    assert config.physical_path == "nccl_sm_declared"
    assert config.risk == ("sm_contention",)
    assert config.requires["min_gpus"] == 2
    assert config.expected["overlap_pct"] == [0.1, 0.9]
    assert config.overlap.dependency_shape == "prefetch"
    assert config.overlap.serialization_warn_threshold == 0.42
    assert config.overlap.serialization_warn_threshold_explicit


def test_parse_optional_cudnn_policy() -> None:
    config = parse_config(
        {
            "version": 1,
            "name": "cudnn_policy",
            "cudnn_deterministic": "true",
            "cudnn_benchmark": "false",
            "kernel": {"family": "null", "implementation": "empty_range"},
        }
    )

    assert config.cudnn_deterministic is True
    assert config.cudnn_benchmark is False


def test_parse_stage1_offset_runtime_and_cache_policy() -> None:
    config = parse_config(
        {
            "version": 1,
            "name": "stage1_offset",
            "runtime": {
                "comm_concurrency_mode": "separate_process_groups",
                "nccl_profile": "controlled_ring_simple",
            },
            "cache_policy": "flushed",
            "kernels": [
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
            ],
            "overlap": {
                "dependency_shape": "offset_sweep",
                "offset_sweep": {
                    "offsets_pct": [0.0, 0.5, 0.9],
                    "direction": "b_leads_a",
                    "delay_policy": "device_spin",
                    "reference": "lead_single_median",
                },
            },
        }
    )

    assert config.overlap.dependency_shape == "offset_sweep"
    assert config.overlap.offset_sweep.offsets_pct == (0.0, 0.5, 0.9)
    assert config.overlap.offset_sweep.direction == "b_leads_a"
    assert config.runtime.comm_concurrency_mode == "separate_process_groups"
    assert config.runtime.nccl_profile == "controlled_ring_simple"
    assert config.cache_policy == "flushed"


def test_parse_cube_cards() -> None:
    root = Path("kernel_simulator/cases/cube")
    for path in root.rglob("*.yaml"):
        config = parse_config(__import__("yaml").safe_load(path.read_text()))
        assert config.selected_kernels(), path


def test_verdict_pass_warn_fail() -> None:
    base = {
        "name": "case",
        "version": 1,
        "case_hash": "abc",
        "mode": "concurrent",
        "result": {
            "concurrent": {
                "per_kernel": [
                    {"slowdown_vs_single": 1.05, "overlap_pct": {"median": 0.7}},
                    {"slowdown_vs_single": 1.02, "overlap_pct": {"median": 0.8}},
                ],
                "pairwise_overlap_pct": [
                    [{"median": 0.0}, {"median": 0.7}],
                    [{"median": 0.8}, {"median": 0.0}],
                ],
            }
        },
    }

    pass_config = parse_config(
        {
            "version": 1,
            "name": "pass",
            "expected": {
                "slowdown_vs_single": [1.0, 1.1],
                "overlap_pct": [0.6, 0.9],
                "pairwise_overlap_min": 0.6,
                "pairwise_overlap_pct_1_0": [0.7, 0.9],
            },
            "kernels": [
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
            ],
        }
    )
    assert _build_verdict(pass_config, base)["status"] == "pass"

    fail_config = parse_config(
        {
            "version": 1,
            "name": "fail",
            "expected": {"slowdown_vs_single": [1.0, 1.01]},
            "kernels": [
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
            ],
        }
    )
    assert _build_verdict(fail_config, base)["status"] == "fail"

    warn_config = parse_config(
        {
            "version": 1,
            "name": "warn",
            "kernels": [
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
            ],
        }
    )
    assert _build_verdict(warn_config, base)["status"] == "warn"


def test_verdict_lower_is_better_single_value_direction() -> None:
    base = {
        "mode": "bucket_tail_hiding",
        "result": {
            "dependency_shape": {
                "per_kernel": [],
                "metrics": {
                    "comm_exposed_tail_ms": {"median_ms": 0.25},
                    "comm_hidden_pct": {"median": 0.80},
                },
            }
        },
    }

    pass_config = parse_config(
        {
            "version": 1,
            "name": "tail_pass",
            "expected": {
                "comm_exposed_tail_ms": 0.5,
                "comm_hidden_pct": 0.75,
            },
            "kernels": [
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
            ],
        }
    )
    assert _build_verdict(pass_config, base)["status"] == "pass"

    fail_config = parse_config(
        {
            "version": 1,
            "name": "tail_fail",
            "expected": {"comm_exposed_tail_ms": 0.1},
            "kernels": [
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
            ],
        }
    )
    assert _build_verdict(fail_config, base)["status"] == "fail"


def test_verdict_structured_expected_schema() -> None:
    base = {
        "mode": "bucket_tail_hiding",
        "result": {
            "dependency_shape": {
                "per_kernel": [],
                "metrics": {
                    "consumer_exposed_tail_ms": {"median_ms": 0.25},
                    "consumer_hidden_by_later_producer_pct": {"median": 0.80},
                },
            }
        },
    }

    config = parse_config(
        {
            "version": 1,
            "name": "structured",
            "expected": {
                "consumer_exposed_tail_ms": {"op": "<=", "value": 0.5},
                "consumer_hidden_by_later_producer_pct": {"range": [0.7, 0.9]},
            },
            "kernels": [
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
            ],
        }
    )
    assert _build_verdict(config, base)["status"] == "pass"

    config = parse_config(
        {
            "version": 1,
            "name": "structured_mode",
            "expected": {
                "consumer_exposed_tail_ms": {"mode": "lower", "value": 0.1},
            },
            "kernels": [
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
            ],
        }
    )
    assert _build_verdict(config, base)["status"] == "fail"


def test_verdict_offset_sweep_metrics() -> None:
    base = {
        "mode": "offset_sweep",
        "result": {
            "offset_sweep": {
                "offset_grid": [
                    {
                        "pairwise_overlap_pct_a": {"median": 0.7},
                        "pairwise_overlap_pct_b": {"median": 0.8},
                        "spin_corrected_tax_ms": {"median_ms": 0.05},
                    },
                    {
                        "pairwise_overlap_pct_a": {"median": 0.6},
                        "pairwise_overlap_pct_b": {"median": 0.65},
                        "spin_corrected_tax_ms": {"median_ms": 0.08},
                    },
                ],
                "per_kernel": [],
            },
            "derived": {},
        },
    }
    config = parse_config(
        {
            "version": 1,
            "name": "offset_verdict",
            "expected": {"overlap_pct": 0.6, "spin_corrected_tax_ms": 0.1},
            "kernels": [
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
                {"family": "gemm", "shape": {"m": 8, "k": 8, "n": 8}},
            ],
            "overlap": {"dependency_shape": "offset_sweep"},
        }
    )

    assert _build_verdict(config, base)["status"] == "pass"
