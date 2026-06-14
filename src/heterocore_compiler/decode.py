"""Compile explainable placement decisions for memory-bound LLM decoding."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class DecodeModel:
    name: str = "Qwen2.5-1.5B-Instruct"
    parameter_count: int = 1_543_714_816
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_key_value_heads: int = 2
    head_dim: int = 128


@dataclass(frozen=True)
class DecodePolicy:
    weight_bits: int = 4
    kv_bits: int = 4
    block_size: int = 32
    selected_block_fraction: float = 0.25
    summary_dimensions: int = 16
    summary_bits: int = 8
    local_sram_bytes: int = 8 * 1024 * 1024


def _kv_capacity_bytes(
    model: DecodeModel,
    context_tokens: int,
    batch_size: int,
    bits: int,
) -> int:
    scalars_per_token = (
        2 * model.num_hidden_layers * model.num_key_value_heads * model.head_dim
    )
    return math.ceil(scalars_per_token * bits / 8) * context_tokens * batch_size


def compile_decode_plan(
    model: DecodeModel | None = None,
    policy: DecodePolicy | None = None,
    context_tokens: int = 8192,
    batch_size: int = 1,
) -> dict:
    model = model or DecodeModel()
    policy = policy or DecodePolicy()
    if context_tokens <= 0 or batch_size <= 0:
        raise ValueError("context and batch must be positive")
    if policy.weight_bits not in {4, 8, 16} or policy.kv_bits not in {4, 8, 16}:
        raise ValueError("weight and KV precision must be 4, 8, or 16 bits")
    if not 0 < policy.selected_block_fraction <= 1:
        raise ValueError("selected block fraction must be within (0, 1]")
    if policy.local_sram_bytes <= 0:
        raise ValueError("local SRAM must be positive")

    weight_bytes = math.ceil(model.parameter_count * policy.weight_bits / 8)
    kv_bytes = _kv_capacity_bytes(model, context_tokens, batch_size, policy.kv_bits)
    block_count = math.ceil(context_tokens / policy.block_size)
    summary_bytes = (
        model.num_hidden_layers
        * model.num_key_value_heads
        * block_count
        * policy.summary_dimensions
        * policy.summary_bits
        // 8
    )
    vector_bytes = batch_size * model.hidden_size * 2
    weight_tile_bytes = min(policy.local_sram_bytes // 2, 2 * 1024 * 1024)
    summary_location = (
        "local_sram"
        if summary_bytes <= policy.local_sram_bytes // 4
        else "external_memory"
    )

    decisions = [
        {
            "object": "model_weights",
            "kind": "state",
            "target": "external_memory",
            "bytes": weight_bytes,
            "reason": "Full packed weights exceed local SRAM and stream by operator.",
        },
        {
            "object": "active_weight_tiles",
            "kind": "state",
            "target": "local_sram",
            "bytes": weight_tile_bytes,
            "reason": "Double-buffered tiles overlap external reads with execution.",
        },
        {
            "object": "kv_cache",
            "kind": "state",
            "target": "paged_external_memory",
            "bytes": kv_bytes,
            "reason": "Context-scale KV state exceeds local capacity.",
        },
        {
            "object": "kv_block_summaries",
            "kind": "state",
            "target": summary_location,
            "bytes": summary_bytes,
            "reason": "Compact summaries are prioritized because they gate full KV reads.",
        },
        {
            "object": "decode_vectors",
            "kind": "state",
            "target": "local_sram",
            "bytes": vector_bytes * 4,
            "reason": "Residual, normalized, query, and reduction vectors are immediately reused.",
        },
        {
            "object": "qk_block_scoring",
            "kind": "operator",
            "target": "quantized_selector",
            "bytes": summary_bytes,
            "reason": "Low-precision summary scoring happens before expensive KV gather.",
        },
        {
            "object": "topk_block_selection",
            "kind": "operator",
            "target": "streaming_topk",
            "bytes": 0,
            "reason": "Streaming insertion top-k avoids a complete score sort.",
        },
        {
            "object": "selected_kv_gather",
            "kind": "operator",
            "target": "dma_gather_engine",
            "bytes": round(kv_bytes * policy.selected_block_fraction),
            "reason": "Only selected blocks move into the exact attention datapath.",
        },
        {
            "object": "residual_rmsnorm",
            "kind": "operator",
            "target": "fused_vector_engine",
            "bytes": vector_bytes * 3,
            "reason": "Fusion removes an intermediate external-memory round trip.",
        },
        {
            "object": "projections_and_mlp",
            "kind": "operator",
            "target": "digital_matrix_vector_engine",
            "bytes": weight_bytes,
            "reason": "Packed matrix-vector execution remains digital until analog error is validated.",
        },
    ]

    return {
        "schema_version": "heterocore.decode_plan.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": asdict(model),
        "workload": {
            "context_tokens": context_tokens,
            "batch_size": batch_size,
        },
        "policy": asdict(policy),
        "summary": {
            "weight_bytes": weight_bytes,
            "kv_capacity_bytes": kv_bytes,
            "summary_bytes": summary_bytes,
            "selected_kv_bytes": round(kv_bytes * policy.selected_block_fraction),
            "decision_count": len(decisions),
        },
        "decisions": decisions,
        "objective": "minimize logical bytes moved per output token subject to local capacity",
        "claim_scope": {
            "classification": "analytical placement",
            "measured_hardware": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--weight-bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--kv-bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--selected-block-fraction", type=float, default=0.25)
    parser.add_argument("--local-sram-mib", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qwen_decode.plan.json"),
    )
    args = parser.parse_args(argv)

    plan = compile_decode_plan(
        policy=DecodePolicy(
            weight_bits=args.weight_bits,
            kv_bits=args.kv_bits,
            selected_block_fraction=args.selected_block_fraction,
            local_sram_bytes=args.local_sram_mib * 1024 * 1024,
        ),
        context_tokens=args.context,
        batch_size=args.batch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(plan['decisions'])} decode placements to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
