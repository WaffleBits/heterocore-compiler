from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Operator:
    operator_id: str
    op_type: str
    m: int = 1
    k: int = 1
    n: int = 1
    weight_bits: int = 16
    accuracy_sensitive: bool = False
    weight_name: str | None = None
    source_node: str | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    @property
    def macs(self) -> int:
        if self.op_type not in {"matmul", "linear"}:
            return 0
        return self.m * self.k * self.n

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Operator":
        dimensions = value.get("dimensions", {})
        return cls(
            operator_id=str(value["id"]),
            op_type=str(value["type"]).lower(),
            m=int(dimensions.get("m", 1)),
            k=int(dimensions.get("k", 1)),
            n=int(dimensions.get("n", 1)),
            weight_bits=int(value.get("weight_bits", 16)),
            accuracy_sensitive=bool(value.get("accuracy_sensitive", False)),
            weight_name=value.get("weight_name"),
            source_node=value.get("source_node"),
            inputs=tuple(value.get("inputs", ())),
            outputs=tuple(value.get("outputs", ())),
        )

    def dimensions(self) -> dict[str, int]:
        return {"m": self.m, "k": self.k, "n": self.n}


@dataclass(frozen=True)
class ModelGraph:
    name: str
    operators: tuple[Operator, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelGraph":
        return cls(
            name=str(value["model"]["name"]),
            operators=tuple(Operator.from_dict(op) for op in value["operators"]),
        )
