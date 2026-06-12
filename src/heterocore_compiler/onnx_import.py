from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any

from .model import ModelGraph, Operator


def _dimension_value(dimension: Any) -> int:
    if dimension.HasField("dim_value") and dimension.dim_value > 0:
        return int(dimension.dim_value)
    return 1


def _collect_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    values = [
        *model.graph.input,
        *model.graph.value_info,
        *model.graph.output,
    ]
    shapes = {}
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shapes[value.name] = tuple(
            _dimension_value(dimension) for dimension in tensor_type.shape.dim
        )
    for initializer in model.graph.initializer:
        shapes[initializer.name] = tuple(int(dimension) for dimension in initializer.dims)
    return shapes


def _matrix_dimensions(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[int, int, int]:
    if len(left_shape) < 2 or len(right_shape) < 2:
        return 1, 1, 1
    m = prod(left_shape[:-1])
    k = left_shape[-1]
    n = right_shape[-1]
    return max(1, m), max(1, k), max(1, n)


def import_onnx(path: str | Path, weight_bits: int = 8) -> ModelGraph:
    try:
        import onnx
        from onnx import shape_inference
    except ImportError as error:
        raise RuntimeError(
            "ONNX import requires the optional dependency: pip install -e '.[onnx]'"
        ) from error

    model = shape_inference.infer_shapes(onnx.load(str(path)), strict_mode=False)
    shapes = _collect_shapes(model)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    operators = []

    type_map = {
        "MatMul": "matmul",
        "Gemm": "linear",
        "Softmax": "softmax",
        "LayerNormalization": "layer_norm",
        "Relu": "relu",
        "Gelu": "gelu",
        "Gather": "embedding",
    }

    for index, node in enumerate(model.graph.node):
        op_type = type_map.get(node.op_type)
        if op_type is None:
            continue

        node_name = node.name or f"{node.op_type.lower()}_{index}"
        left_shape = shapes.get(node.input[0], ()) if node.input else ()
        right_shape = shapes.get(node.input[1], ()) if len(node.input) > 1 else ()
        if node.op_type in {"MatMul", "Gemm"}:
            m, k, n = _matrix_dimensions(left_shape, right_shape)
        else:
            flattened = prod(left_shape[:-1]) if len(left_shape) > 1 else 1
            width = left_shape[-1] if left_shape else 1
            m, k, n = max(1, flattened), max(1, width), max(1, width)

        weight_name = (
            node.input[1]
            if len(node.input) > 1 and node.input[1] in initializer_names
            else None
        )
        matrix_without_resident_weights = (
            node.op_type in {"MatMul", "Gemm"} and weight_name is None
        )
        operators.append(
            Operator(
                operator_id=node_name,
                op_type=op_type,
                m=m,
                k=k,
                n=n,
                weight_bits=weight_bits if weight_name else 16,
                accuracy_sensitive=node.op_type
                in {"Softmax", "LayerNormalization"}
                or matrix_without_resident_weights,
                weight_name=weight_name,
                source_node=node.op_type,
                inputs=tuple(node.input),
                outputs=tuple(node.output),
            )
        )

    if not operators:
        raise ValueError(f"{path} contains no supported ONNX operators")

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    model_name = metadata.get("model_name") or model.graph.name or Path(path).stem
    return ModelGraph(name=model_name, operators=tuple(operators))
