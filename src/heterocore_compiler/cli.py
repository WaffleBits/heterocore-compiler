from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compiler import compile_graph
from .config import HardwareConfig, PartitionPolicy
from .model import ModelGraph
from .onnx_import import import_onnx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Partition an AI graph across analog and digital targets."
    )
    parser.add_argument("graph", type=Path, help="Input graph JSON or ONNX model")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--array-rows", type=int, default=128)
    parser.add_argument("--array-cols", type=int, default=128)
    parser.add_argument("--digital-macs", type=int, default=256)
    parser.add_argument("--clock-mhz", type=int, default=500)
    parser.add_argument("--minimum-analog-macs", type=int, default=65_536)
    parser.add_argument("--maximum-analog-weight-bits", type=int, default=8)
    parser.add_argument(
        "--onnx-weight-bits",
        type=int,
        default=8,
        help="Assumed quantized precision for ONNX initializer-backed matrix operations",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.graph.suffix.lower() == ".onnx":
        graph = import_onnx(args.graph, weight_bits=args.onnx_weight_bits)
    else:
        graph = ModelGraph.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    hardware = HardwareConfig(
        analog_array_rows=args.array_rows,
        analog_array_cols=args.array_cols,
        digital_mac_units=args.digital_macs,
        clock_mhz=args.clock_mhz,
    )
    policy = PartitionPolicy(
        minimum_analog_macs=args.minimum_analog_macs,
        maximum_analog_weight_bits=args.maximum_analog_weight_bits,
    )
    plan = compile_graph(graph, hardware, policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    summary = plan["summary"]
    print(
        f"{graph.name}: analog_macs={summary['analog_mac_fraction']:.1%}, "
        f"projected_energy_reduction={summary['projected_energy_reduction']:.1%}, "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
