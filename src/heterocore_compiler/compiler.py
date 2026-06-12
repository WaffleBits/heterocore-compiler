from __future__ import annotations

from datetime import datetime, timezone

from .config import HardwareConfig, PartitionPolicy
from .cost_model import estimate_operator
from .model import ModelGraph, Operator

SCHEMA_VERSION = "heterocore.execution_plan.v1"


def select_target(op: Operator, policy: PartitionPolicy) -> tuple[str, str]:
    if op.op_type not in policy.supported_analog_ops:
        return "digital", "operator is not supported by the analog backend"
    if op.accuracy_sensitive:
        return "digital", "operator is marked accuracy-sensitive"
    if op.weight_bits > policy.maximum_analog_weight_bits:
        return "digital", "weight precision exceeds analog policy"
    if op.macs < policy.minimum_analog_macs:
        return "digital", "operation is below the analog utilization threshold"
    return "analog", "matrix operation satisfies analog mapping policy"


def compile_graph(
    graph: ModelGraph,
    hardware: HardwareConfig | None = None,
    policy: PartitionPolicy | None = None,
) -> dict:
    hardware = hardware or HardwareConfig()
    policy = policy or PartitionPolicy()
    compiled_operators = []

    for sequence, op in enumerate(graph.operators):
        target, reason = select_target(op, policy)
        compiled_operators.append(
            {
                "sequence": sequence,
                "id": op.operator_id,
                "type": op.op_type,
                "dimensions": op.dimensions(),
                "weight_bits": op.weight_bits,
                "macs": op.macs,
                "target": target,
                "reason": reason,
                "estimate": estimate_operator(op, target, hardware),
            }
        )

    analog_macs = sum(op["macs"] for op in compiled_operators if op["target"] == "analog")
    total_macs = sum(op["macs"] for op in compiled_operators)
    total_cycles = sum(op["estimate"]["cycles"] for op in compiled_operators)
    total_energy_pj = sum(op["estimate"]["total_energy_pj"] for op in compiled_operators)
    digital_baseline_energy_pj = sum(
        estimate_operator(op, "digital", hardware)["total_energy_pj"]
        for op in graph.operators
    )
    digital_baseline_traffic = sum(
        estimate_operator(op, "digital", hardware)["memory_traffic_bytes"]
        for op in graph.operators
    )
    mapped_traffic = sum(op["estimate"]["memory_traffic_bytes"] for op in compiled_operators)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"name": graph.name, "operator_count": len(graph.operators)},
        "hardware": hardware.to_dict(),
        "policy": policy.to_dict(),
        "summary": {
            "total_macs": total_macs,
            "analog_macs": analog_macs,
            "analog_mac_fraction": round(analog_macs / total_macs, 6) if total_macs else 0.0,
            "estimated_cycles": total_cycles,
            "estimated_latency_us": round(total_cycles / hardware.clock_mhz, 6),
            "estimated_energy_uj": round(total_energy_pj / 1_000_000.0, 6),
            "digital_baseline_energy_uj": round(
                digital_baseline_energy_pj / 1_000_000.0, 6
            ),
            "projected_energy_reduction": round(
                1.0 - total_energy_pj / digital_baseline_energy_pj, 6
            )
            if digital_baseline_energy_pj
            else 0.0,
            "mapped_memory_traffic_bytes": mapped_traffic,
            "digital_baseline_memory_traffic_bytes": digital_baseline_traffic,
            "projected_memory_traffic_reduction": round(
                1.0 - mapped_traffic / digital_baseline_traffic, 6
            )
            if digital_baseline_traffic
            else 0.0,
        },
        "operators": compiled_operators,
        "claim_scope": {
            "classification": "analytical projection",
            "measured_hardware": False,
            "notes": "Values come from the configured cost model, not fabricated silicon.",
        },
    }

