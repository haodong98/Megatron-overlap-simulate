from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kernel_simulator.config import parse_config
from kernel_simulator.stage1 import (
    builtin_primitives,
    environment_match,
    freshness_status,
    generate_stage1_manifest,
)


def test_stage1_manifest_generation_and_phase_rules(tmp_path) -> None:
    manifest = generate_stage1_manifest(tmp_path)

    assert manifest["entry_count"] == len(manifest["entries"])
    assert (tmp_path / "manifest.json").exists()
    assert manifest["entries"]

    first = manifest["entries"][0]
    config = parse_config(first["case"])
    assert config.overlap.dependency_shape == "offset_sweep"
    assert config.selected_kernels()

    assert any(
        entry["a_phase"] == "forward" and entry["b_phase"] == "backward"
        for entry in manifest["entries"]
    )
    assert any(entry["a_phase"] == "backward" for entry in manifest["entries"])
    assert not any(
        entry["a_phase"] == "backward" and entry["b_phase"] == "backward"
        for entry in manifest["entries"]
    )


def test_stage1_manifest_has_comm_group_modes(tmp_path) -> None:
    manifest = generate_stage1_manifest(tmp_path)
    comm_comm = [
        entry
        for entry in manifest["entries"]
        if entry["a_primitive"] == "all_reduce" and entry["b_primitive"] == "all_gather"
    ]

    modes = {entry["runtime"]["comm_concurrency_mode"] for entry in comm_comm}
    profiles = {entry["runtime"]["nccl_profile"] for entry in comm_comm}
    assert {"same_process_group", "separate_process_groups"}.issubset(modes)
    assert "controlled_ring_simple" in profiles
    assert "controlled_ring_ll128" in profiles


def test_stage1_manifest_marks_proxy_and_sentinel_entries(tmp_path) -> None:
    manifest = generate_stage1_manifest(tmp_path)

    fp8_entry = next(
        entry for entry in manifest["entries"] if entry["a_primitive"] == "fp8_dense_gemm"
    )
    assert fp8_entry["case"]["stage1"]["a_fidelity"] == "proxy"
    assert fp8_entry["case"]["stage1"]["confidence_cap"] == "low"

    assert manifest["sentinel_interval_seconds"] == 30 * 60
    assert {entry["section"] for entry in manifest["sentinels"]} == {
        "stage1_sentinel_compute",
        "stage1_sentinel_comm",
        "stage1_sentinel_comm_comm_optional",
    }


def test_stage1b_has_memory_comm_non_diagonal_subset(tmp_path) -> None:
    manifest = generate_stage1_manifest(tmp_path)
    pairs = {
        (entry["a_primitive"], entry["b_primitive"], entry["a_shape_class"], entry["b_shape_class"])
        for entry in manifest["entries"]
        if entry["section"] == "stage1b_memory_comm"
    }

    assert ("d2d_copy_contiguous", "all_to_all", "large", "medium") in pairs
    assert ("d2d_copy_noncontiguous", "all_to_all", "large", "medium") in pairs
    assert not any(pair[0] == "memset_or_fill" for pair in pairs)


def test_builtin_primitive_count_and_required_set() -> None:
    primitives = builtin_primitives()
    assert len(primitives) == 18
    assert "all_to_all" in primitives
    assert "p2p_send_recv" in primitives
    assert primitives["optimizer_adam"].sim_caveat


def test_environment_match_and_freshness() -> None:
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)
    measured = (now - timedelta(days=10)).isoformat()
    env = {
        "device_name": "H100",
        "device_capability": (9, 0),
        "cuda_runtime": "12.4",
        "torch": "2.6.0",
        "nccl_version": (2, 21, 5),
        "cuda_device_max_connections": "8",
        "nccl_algo": "Ring",
        "nccl_proto": "Simple",
    }

    assert freshness_status(measured, now=now) == "warning"
    match = environment_match(env, dict(env), measured_at=measured, now=now)
    assert match["status"] == "match"
    assert match["freshness"] == "warning"

    changed = dict(env)
    changed["nccl_proto"] = "LL128"
    mismatch = environment_match(env, changed, measured_at=measured, now=now)
    assert mismatch["status"] == "mismatch"
    assert mismatch["mismatches"][0]["key"] == "nccl_proto"
