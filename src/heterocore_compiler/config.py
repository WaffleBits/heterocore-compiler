from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareConfig:
    analog_array_rows: int = 128
    analog_array_cols: int = 128
    analog_adc_cycles: int = 4
    digital_mac_units: int = 256
    clock_mhz: int = 500
    analog_mac_energy_pj: float = 0.12
    digital_mac_energy_pj: float = 1.8
    sram_byte_energy_pj: float = 5.0
    dram_byte_energy_pj: float = 120.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PartitionPolicy:
    minimum_analog_macs: int = 65_536
    maximum_analog_weight_bits: int = 8
    supported_analog_ops: tuple[str, ...] = ("matmul", "linear")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["supported_analog_ops"] = list(self.supported_analog_ops)
        return result

