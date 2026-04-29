from __future__ import annotations

import pytest
import torch

from kernel_simulator.config import parse_config
from kernel_simulator.offset import classify_offset_entry, offset_sample_metrics
from kernel_simulator.replay import run_replay
from kernel_simulator.runner import (
    _adaptive_overlap_threshold,
    _build_dependency_ops,
    _dependency_metrics,
    _dependency_shape_plan,
    _pairwise_overlap,
    _profiler_observed_names,
    _summarize_metric,
    _serialized_warning,
)


class _FakeKernel:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeEvent:
    def __init__(
        self,
        name: str,
        *,
        device_type: str = "DeviceType.CPU",
        cuda_time_total: float = 0.0,
        kernels: list[_FakeKernel] | None = None,
    ) -> None:
        self.name = name
        self.device_type = device_type
        self.cuda_time_total = cuda_time_total
        self.kernels = kernels or []


class _FakeProfiler:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self._events = events
        self.profiler = None

    def events(self) -> list[_FakeEvent]:
        return self._events


def test_pairwise_overlap_matrix_2_3_4() -> None:
    assert _pairwise_overlap([0.0, 2.0], [4.0, 6.0]) == [
        [0.0, 2.0],
        [2.0, 0.0],
    ]

    assert _pairwise_overlap([0.0, 2.0, 5.0], [4.0, 6.0, 7.0]) == [
        [0.0, 2.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]

    assert _pairwise_overlap([0.0, 1.0, 2.0, 8.0], [5.0, 3.0, 7.0, 9.0]) == [
        [0.0, 2.0, 3.0, 0.0],
        [2.0, 0.0, 1.0, 0.0],
        [3.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]


def test_adaptive_overlap_threshold() -> None:
    assert _adaptive_overlap_threshold(0.099) == 0.10
    assert _adaptive_overlap_threshold(0.1) == 0.25
    assert _adaptive_overlap_threshold(0.999) == 0.25
    assert _adaptive_overlap_threshold(1.0) == 0.40


def test_serialized_warning_uses_adaptive_thresholds() -> None:
    warning = _serialized_warning(
        [[0.09], [0.20], [0.39]],
        threshold=0.30,
        names=["small", "medium", "large"],
        durations_ms=[[0.05], [0.5], [2.0]],
    )
    assert warning is not None
    assert "adaptive" in warning

    no_warning = _serialized_warning(
        [[0.11], [0.20], [0.39]],
        threshold=0.30,
        names=["small", "medium", "large"],
        durations_ms=[[0.05], [0.5], [2.0]],
    )
    assert no_warning is None

    explicit_warning = _serialized_warning(
        [[0.20]],
        threshold=0.30,
        names=["explicit"],
        durations_ms=[[0.05]],
        threshold_explicit=True,
    )
    assert explicit_warning is not None


def test_bucket_tail_hiding_uses_k_to_one_topology() -> None:
    config = parse_config(
        {
            "version": 1,
            "name": "bucket_shape",
            "grain": {"buckets": 2, "producers_per_bucket": 3},
            "kernels": [
                {"family": "null", "implementation": "tiny_sm"},
                {"family": "null", "implementation": "tiny_sm"},
            ],
            "overlap": {"dependency_shape": "bucket_tail_hiding"},
        }
    )

    plan = _dependency_shape_plan(config, "bucket_tail_hiding")
    ops = _build_dependency_ops("bucket_tail_hiding", plan)

    assert plan["steps"] == 6
    assert [op.label for op in ops] == [
        "producer",
        "producer",
        "producer",
        "consumer",
        "producer",
        "producer",
        "producer",
        "consumer",
    ]
    assert ops[3].waits == (2,)
    assert ops[7].waits == (6,)


def test_producer_consumer_ring_uses_one_to_one_topology() -> None:
    config = parse_config(
        {
            "version": 1,
            "name": "ring_shape",
            "grain": {"blocks": 3},
            "kernels": [
                {"family": "null", "implementation": "tiny_sm"},
                {"family": "null", "implementation": "tiny_sm"},
            ],
            "overlap": {"dependency_shape": "producer_consumer_ring"},
        }
    )

    plan = _dependency_shape_plan(config, "producer_consumer_ring")
    ops = _build_dependency_ops("producer_consumer_ring", plan)

    assert plan["steps"] == 3
    assert [op.label for op in ops] == [
        "producer",
        "consumer",
        "producer",
        "consumer",
        "producer",
        "consumer",
    ]
    assert ops[1].waits == (0,)
    assert ops[3].waits == (2,)


def test_profiler_validation_separates_kernel_and_framework_names() -> None:
    observed = _profiler_observed_names(
        _FakeProfiler(
            [
                _FakeEvent(
                    "aten::matmul",
                    cuda_time_total=12.0,
                    kernels=[_FakeKernel("cutlass_gemm_kernel")],
                ),
                _FakeEvent("cudaLaunchKernel"),
                _FakeEvent("ncclDevKernel_AllReduce", device_type="DeviceType.CUDA"),
            ]
        )
    )

    assert observed["cuda_kernel_names"] == [
        "cutlass_gemm_kernel",
        "ncclDevKernel_AllReduce",
    ]
    assert observed["cuda_runtime_names"] == ["cudaLaunchKernel"]
    assert observed["framework_op_names"] == ["aten::matmul"]


def test_fraction_metrics_do_not_report_ms_suffixes() -> None:
    summary = _summarize_metric("comm_hidden_pct", [0.25, 0.75])

    assert summary["median"] == 0.5
    assert "median_ms" not in summary

    timing = _summarize_metric("comm_exposed_tail_ms", [1.0, 3.0])
    assert timing["median_ms"] == 2.0


def test_dependency_metrics_emit_generic_consumer_aliases() -> None:
    metrics = _dependency_metrics(
        [
            {
                "label": "producer",
                "group": 0,
                "start_ms": 0.0,
                "end_ms": 4.0,
                "duration_ms": 4.0,
            },
            {
                "label": "consumer",
                "group": 0,
                "start_ms": 4.0,
                "end_ms": 8.0,
                "duration_ms": 4.0,
            },
            {
                "label": "producer",
                "group": 1,
                "start_ms": 5.0,
                "end_ms": 9.0,
                "duration_ms": 4.0,
            },
            {
                "label": "consumer",
                "group": 1,
                "start_ms": 9.0,
                "end_ms": 11.0,
                "duration_ms": 2.0,
            },
        ]
    )

    assert metrics["consumer_hidden_by_later_producer_ms"] == 3.0
    assert metrics["consumer_exposed_ms"] == 3.0
    assert metrics["consumer_hidden_by_later_producer_pct"] == 0.5
    assert metrics["comm_hidden_ms"] == metrics["consumer_hidden_by_later_producer_ms"]
    assert metrics["comm_hidden_pct"] == metrics["consumer_hidden_by_later_producer_pct"]


def test_prefetch_metrics_report_both_denominators() -> None:
    metrics = _dependency_metrics(
        [
            {
                "label": "prefetch",
                "start_ms": 0.0,
                "end_ms": 10.0,
                "duration_ms": 10.0,
            },
            {
                "label": "compute",
                "start_ms": 2.0,
                "end_ms": 6.0,
                "duration_ms": 4.0,
            },
        ]
    )

    assert metrics["prefetch_hidden_ms"] == 4.0
    assert metrics["prefetch_hidden_pct"] == 0.4
    assert metrics["compute_covered_by_prefetch_pct"] == 1.0
    assert metrics["prefetch_exposed_tail_ms"] == 4.0


def test_offset_sample_metrics_positive_and_negative_offset() -> None:
    positive = offset_sample_metrics(
        {
            "start_a_ms": 0.0,
            "end_a_ms": 10.0,
            "start_b_ms": 5.0,
            "end_b_ms": 13.0,
            "duration_a_ms": 10.0,
            "duration_b_ms": 8.0,
        },
        single_a_ms=10.0,
        single_b_ms=8.0,
        serial_event_ms=18.5,
        target_offset_ms=5.0,
        spin_contamination_ms=0.2,
    )
    assert positive["actual_offset_ms"] == 5.0
    assert positive["actual_overlap_ms"] == 5.0
    assert positive["benefit_vs_serial_event_chain"] == 5.5
    assert positive["tax_vs_ideal_offset_overlap"] == 0.0
    assert positive["spin_corrected_tax_ms"] == -0.2

    negative = offset_sample_metrics(
        {
            "start_a_ms": 4.0,
            "end_a_ms": 14.0,
            "start_b_ms": 0.0,
            "end_b_ms": 8.0,
            "duration_a_ms": 10.0,
            "duration_b_ms": 8.0,
        },
        single_a_ms=10.0,
        single_b_ms=8.0,
        serial_event_ms=18.5,
        target_offset_ms=-4.0,
        spin_contamination_ms=0.0,
    )
    assert negative["actual_offset_ms"] == -4.0
    assert negative["actual_overlap_ms"] == 4.0
    assert negative["tax_vs_ideal_offset_overlap"] == 0.0


def test_offset_classification_uses_noise_tolerance() -> None:
    base = {
        "confidence": "high",
        "eps_ms": 0.1,
        "benefit_vs_serial_event_chain": {"median_ms": 1.0},
        "spin_corrected_tax_ms": {"median_ms": 0.05},
    }
    assert classify_offset_entry(base) == "clean_overlap"

    contended = dict(base)
    contended["spin_corrected_tax_ms"] = {"median_ms": 0.5}
    assert classify_offset_entry(contended) == "beneficial_but_contended"

    harmful = dict(base)
    harmful["benefit_vs_serial_event_chain"] = {"median_ms": -0.2}
    assert classify_offset_entry(harmful) == "harmful_overlap"

    invalid = dict(base)
    invalid["confidence"] = "invalid"
    assert classify_offset_entry(invalid) == "invalid_or_low_overlap"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_null_empty_range_single_and_concurrent_smoke() -> None:
    single_config = parse_config(
        {
            "version": 1,
            "name": "null_single",
            "warmup_iters": 1,
            "measure_iters": 1,
            "kernel": {"family": "null", "implementation": "empty_range"},
        }
    )
    assert run_replay(single_config, "single")["result"]["kernel"]["family"] == "null"

    concurrent_config = parse_config(
        {
            "version": 1,
            "name": "null_concurrent",
            "warmup_iters": 1,
            "measure_iters": 1,
            "kernels": [
                {"family": "null", "implementation": "empty_range"},
                {"family": "null", "implementation": "tiny_sm"},
            ],
        }
    )
    result = run_replay(concurrent_config, "concurrent")["result"]["concurrent"]
    assert "pairwise_overlap_ms" in result
    assert len(result["pairwise_overlap_ms"]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dependency_shape_prefetch_smoke() -> None:
    config = parse_config(
        {
            "version": 1,
            "name": "prefetch_smoke",
            "warmup_iters": 1,
            "measure_iters": 1,
            "kernels": [
                {"family": "null", "implementation": "tiny_sm"},
                {"family": "null", "implementation": "tiny_sm"},
            ],
            "overlap": {"dependency_shape": "prefetch"},
        }
    )
    summary = run_replay(config, "prefetch")
    assert "prefetch_hidden_pct" in summary["result"]["dependency_shape"]["metrics"]
