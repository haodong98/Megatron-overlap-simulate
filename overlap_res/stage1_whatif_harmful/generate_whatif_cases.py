#!/usr/bin/env python3
"""Generate fixed-model what-if cases for harmful GEMM/NCCL overlap.

The generated cases intentionally keep DeepSeek model-derived dimensions fixed:
M = MBS * SEQ_LEN / TP = 2048, K = hidden = 7168, N = ffn_hidden / TP = 4096.
The optional fused SwiGLU proxy only changes N to 8192, matching a fused
gate/up projection for the same model.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


M = 2048
K = 7168
N_BASE = 4096
N_FUSED_SWIGLU = 8192
COMM_BYTES = 2048 * 7168 * 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--include-fused", action="store_true")
    parser.add_argument("--variants", default="")
    parser.add_argument("--timing-warmup-iters", type=int, default=20)
    parser.add_argument("--timing-measure-iters", type=int, default=200)
    parser.add_argument("--nsys-warmup-iters", type=int, default=3)
    parser.add_argument("--nsys-measure-iters", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cases_dir = out_dir / "cases"
    manifest_dir = out_dir / "manifests"
    cases_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    variants = _variant_specs(include_fused=args.include_fused)
    if args.variants:
        keep = {item.strip() for item in args.variants.split(",") if item.strip()}
        variants = [variant for variant in variants if variant["id"] in keep]
        missing = sorted(keep - {variant["id"] for variant in variants})
        if missing:
            raise SystemExit(f"Unknown WHATIF_VARIANTS entries: {', '.join(missing)}")

    materialized = []
    for variant in variants:
        timing_case = _case_for_variant(
            variant,
            warmup_iters=args.timing_warmup_iters,
            measure_iters=args.timing_measure_iters,
            nsys=False,
        )
        nsys_case = _case_for_variant(
            variant,
            warmup_iters=args.nsys_warmup_iters,
            measure_iters=args.nsys_measure_iters,
            nsys=True,
        )
        case_path = cases_dir / f"{variant['id']}.json"
        nsys_case_path = cases_dir / f"{variant['id']}_nsys.json"
        manifest_path = manifest_dir / f"{variant['id']}.json"
        _write_json(case_path, timing_case)
        _write_json(nsys_case_path, nsys_case)
        _write_json(manifest_path, _manifest_for_case(variant, timing_case))

        item = dict(variant)
        item.update(
            {
                "case": str(case_path),
                "nsys_case": str(nsys_case_path),
                "manifest": str(manifest_path),
            }
        )
        materialized.append(item)

    _write_json(out_dir / "variants.json", {"variants": materialized})
    with (out_dir / "run_list.tsv").open("w", encoding="utf-8") as handle:
        for variant in materialized:
            handle.write(
                "\t".join(
                    [
                        variant["id"],
                        variant["manifest"],
                        variant["nsys_case"],
                        variant["nccl_algo"],
                        variant["nccl_proto"],
                        str(variant.get("nccl_min_nchannels") or ""),
                    ]
                )
            )
            handle.write("\n")
    print(f"generated_variants={len(materialized)}")
    print(f"variants_json={out_dir / 'variants.json'}")
    print(f"run_list={out_dir / 'run_list.tsv'}")


def _variant_specs(include_fused: bool) -> List[Dict[str, Any]]:
    variants = [
        _variant("anchor_ar_simple", "all_reduce", "controlled_ring_simple"),
        _variant("ar_simple_ch4", "all_reduce", "controlled_ring_simple", min_channels=4),
        _variant("ar_simple_ch8", "all_reduce", "controlled_ring_simple", min_channels=8),
        _variant("ar_simple_ch16", "all_reduce", "controlled_ring_simple", min_channels=16),
        _variant("ar_ll128", "all_reduce", "controlled_ring_ll128"),
        _variant("ar_ll", "all_reduce", "controlled_ring_ll"),
        _variant("rs_simple", "reduce_scatter", "controlled_ring_simple"),
        _variant("rs_simple_ch8", "reduce_scatter", "controlled_ring_simple", min_channels=8),
        _variant("ag_simple", "all_gather", "controlled_ring_simple"),
        _variant("ag_simple_ch8", "all_gather", "controlled_ring_simple", min_channels=8),
    ]
    if include_fused:
        variants.extend(
            [
                _variant("fused_ar_simple", "all_reduce", "controlled_ring_simple", gemm_n=N_FUSED_SWIGLU),
                _variant(
                    "fused_ar_simple_ch8",
                    "all_reduce",
                    "controlled_ring_simple",
                    min_channels=8,
                    gemm_n=N_FUSED_SWIGLU,
                ),
                _variant("fused_rs_simple", "reduce_scatter", "controlled_ring_simple", gemm_n=N_FUSED_SWIGLU),
                _variant("fused_ag_simple", "all_gather", "controlled_ring_simple", gemm_n=N_FUSED_SWIGLU),
            ]
        )
    return variants


def _variant(
    variant_id,
    collective,
    nccl_profile,
    *,
    min_channels=None,
    gemm_n=N_BASE,
) -> Dict[str, Any]:
    proto = {
        "controlled_ring_simple": "Simple",
        "controlled_ring_ll128": "LL128",
        "controlled_ring_ll": "LL",
    }[nccl_profile]
    return {
        "id": variant_id,
        "collective": collective,
        "gemm_n": gemm_n,
        "fused_swiglu_proxy": gemm_n == N_FUSED_SWIGLU,
        "nccl_profile": nccl_profile,
        "nccl_algo": "Ring",
        "nccl_proto": proto,
        "nccl_min_nchannels": min_channels,
    }


def _case_for_variant(variant, *, warmup_iters, measure_iters, nsys):
    suffix = "_nsys" if nsys else ""
    case_id = f"whatif_harmful_{variant['id']}{suffix}"
    gemm_name = (
        "deepseek_swiglu_fused_gate_up_gemm_tp4"
        if variant["fused_swiglu_proxy"]
        else "deepseek_mlp_projection_gemm_tp4"
    )
    collective = variant["collective"]
    return {
        "version": 1,
        "name": case_id,
        "card_id": case_id,
        "device": "cuda:0",
        "dtype": "bf16",
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "parallel": {"distributed": True},
        "runtime": {
            "comm_concurrency_mode": "none",
            "nccl_profile": variant["nccl_profile"],
        },
        "cache_policy": "steady_warm",
        "requires": {"distributed": True, "min_gpus": 4},
        "physical_path": "comm",
        "sim_caveat": (
            "Fixed DeepSeek-derived TP GEMM proxy. Optional fused variants use the same model "
            "with a fused SwiGLU gate/up projection proxy."
        ),
        "kernels": [
            {
                "name": gemm_name,
                "family": "gemm",
                "implementation": "torch_matmul",
                "phase": "forward",
                "dtype": "bf16",
                "shape": {"m": M, "k": K, "n": int(variant["gemm_n"])},
            },
            {
                "name": f"tp_hidden_{collective}_28mib",
                "family": collective,
                "implementation": "torch_distributed",
                "phase": "forward",
                "dtype": "bf16",
                "shape": {"bytes": COMM_BYTES},
            },
        ],
        "overlap": {
            "dependency_shape": "offset_sweep",
            "offset_sweep": {
                "offsets_pct": [0.0],
                "direction": "a_leads_b",
                "delay_policy": "device_spin",
                "reference": "single_a_median",
            },
        },
        "stage1": {
            "section": "whatif_harmful_fixed_model",
            "a_primitive": "bf16_dense_gemm",
            "b_primitive": collective,
            "a_shape_class": "deepseek_fixed",
            "b_shape_class": "tp_activation_28mib",
            "declared_nccl_algo": variant["nccl_algo"],
            "declared_nccl_proto": variant["nccl_proto"],
            "nccl_min_nchannels": variant.get("nccl_min_nchannels"),
            "fused_swiglu_proxy": variant["fused_swiglu_proxy"],
        },
    }


def _manifest_for_case(variant, case):
    entry = {
        "pair_id": variant["id"],
        "section": "whatif_harmful_fixed_model",
        "mode": "offset_sweep",
        "a_primitive": "bf16_dense_gemm",
        "b_primitive": variant["collective"],
        "a_phase": "forward",
        "b_phase": "forward",
        "a_shape_class": "deepseek_fixed",
        "b_shape_class": "tp_activation_28mib",
        "runtime": case["runtime"],
        "requires": case["requires"],
        "case": case,
    }
    return {
        "version": 1,
        "kind": "stage1_offset_sweep_manifest",
        "world_size": 4,
        "node_scope": "1-node/4-GPU",
        "entry_count": 1,
        "entries": [entry],
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
