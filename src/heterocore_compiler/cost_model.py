from __future__ import annotations

from math import ceil

from .config import HardwareConfig
from .model import Operator


def estimate_operator(op: Operator, target: str, hardware: HardwareConfig) -> dict:
    if op.macs == 0:
        cycles = max(1, ceil((op.m * op.n) / hardware.digital_mac_units))
        compute_energy_pj = float(op.m * op.n) * hardware.digital_mac_energy_pj
        peripheral_energy_pj = cycles * hardware.digital_control_energy_pj_per_cycle
        weight_bytes = 0
    elif target == "analog":
        row_tiles = ceil(op.k / hardware.analog_array_rows)
        column_tiles = ceil(op.n / hardware.analog_array_cols)
        cycles = op.m * row_tiles * column_tiles * hardware.analog_adc_cycles
        tile_activations = op.m * row_tiles * column_tiles
        dac_conversions = op.m * op.k * column_tiles
        adc_conversions = op.m * op.n * row_tiles
        partial_sum_accumulations = op.m * op.n * max(0, row_tiles - 1)
        compute_energy_pj = op.macs * hardware.analog_mac_energy_pj
        peripheral_energy_pj = (
            dac_conversions * hardware.dac_conversion_energy_pj
            + adc_conversions * hardware.adc_conversion_energy_pj
            + partial_sum_accumulations * hardware.analog_accumulation_energy_pj
            + tile_activations * hardware.analog_control_energy_pj_per_tile
            + row_tiles
            * column_tiles
            * hardware.analog_calibration_energy_pj_per_array
        )
        weight_bytes = ceil(op.k * op.n * op.weight_bits / 8)
    else:
        cycles = ceil(op.macs / hardware.digital_mac_units)
        compute_energy_pj = op.macs * hardware.digital_mac_energy_pj
        peripheral_energy_pj = cycles * hardware.digital_control_energy_pj_per_cycle
        weight_bytes = ceil(op.k * op.n * op.weight_bits / 8)

    activation_bytes = (op.m * op.k + op.m * op.n) * 2
    memory_traffic_bytes = activation_bytes if target == "analog" else activation_bytes + weight_bytes
    memory_energy_pj = memory_traffic_bytes * hardware.sram_byte_energy_pj
    interconnect_energy_pj = (
        memory_traffic_bytes * hardware.interconnect_byte_energy_pj
    )
    total_energy_pj = (
        compute_energy_pj
        + peripheral_energy_pj
        + memory_energy_pj
        + interconnect_energy_pj
    )

    return {
        "cycles": cycles,
        "latency_us": round(cycles / hardware.clock_mhz, 6),
        "compute_energy_pj": round(compute_energy_pj, 3),
        "peripheral_energy_pj": round(peripheral_energy_pj, 3),
        "memory_energy_pj": round(memory_energy_pj, 3),
        "interconnect_energy_pj": round(interconnect_energy_pj, 3),
        "total_energy_pj": round(total_energy_pj, 3),
        "weight_bytes": weight_bytes,
        "memory_traffic_bytes": memory_traffic_bytes,
    }
