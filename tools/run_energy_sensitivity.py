from __future__ import annotations

import argparse
import json
from pathlib import Path

from heterocore_compiler import HardwareConfig, PartitionPolicy, compile_graph, import_onnx


SCENARIOS = {
    "optimistic_peripherals": HardwareConfig(
        dac_conversion_energy_pj=1.0,
        adc_conversion_energy_pj=10.0,
        analog_control_energy_pj_per_tile=200.0,
        analog_calibration_energy_pj_per_array=1_000.0,
        interconnect_byte_energy_pj=0.1,
    ),
    "nominal_peripherals": HardwareConfig(),
    "conservative_peripherals": HardwareConfig(
        dac_conversion_energy_pj=8.0,
        adc_conversion_energy_pj=120.0,
        analog_accumulation_energy_pj=3.0,
        analog_control_energy_pj_per_tile=1_200.0,
        analog_calibration_energy_pj_per_array=20_000.0,
        interconnect_byte_energy_pj=1.0,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show how analog energy conclusions change with peripheral assumptions."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--minimum-analog-macs", type=int, default=16_000)
    parser.add_argument("--weight-bits", type=int, default=4)
    args = parser.parse_args()

    graph = import_onnx(args.model, weight_bits=args.weight_bits)
    policy = PartitionPolicy(
        minimum_analog_macs=args.minimum_analog_macs,
        maximum_analog_weight_bits=args.weight_bits,
    )
    scenarios = []
    for name, hardware in SCENARIOS.items():
        plan = compile_graph(graph, hardware, policy)
        scenarios.append(
            {
                "name": name,
                "hardware": hardware.to_dict(),
                "analog_mac_fraction": plan["summary"]["analog_mac_fraction"],
                "estimated_energy_uj": plan["summary"]["estimated_energy_uj"],
                "digital_baseline_energy_uj": plan["summary"][
                    "digital_baseline_energy_uj"
                ],
                "projected_energy_reduction": plan["summary"][
                    "projected_energy_reduction"
                ],
                "energy_breakdown_pj": plan["summary"]["energy_breakdown_pj"],
            }
        )

    report = {
        "schema_version": "heterocore.energy_sensitivity.v1",
        "model": graph.name,
        "source": str(args.model),
        "scenarios": scenarios,
        "claim_scope": {
            "classification": "analytical sensitivity analysis",
            "measured_hardware": False,
            "notes": "The range is assumption-driven and must be calibrated against hardware.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for scenario in scenarios:
        print(
            f"{scenario['name']}: "
            f"projected_energy_reduction={scenario['projected_energy_reduction']:.1%}"
        )


if __name__ == "__main__":
    main()
