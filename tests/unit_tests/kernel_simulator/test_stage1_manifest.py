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

    # Stage-1 v1 deliberately excludes fwd x bwd in the canonical entry.
    assert not any(
        entry["a_phase"] == "forward" and entry["b_phase"] == "backward"
        for entry in manifest["entries"]
    )
    assert any(entry["a_phase"] == "backward" for entry in manifest["entries"])


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
