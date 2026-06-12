import tempfile
import unittest
from pathlib import Path

from heterocore_compiler.onnx_import import import_onnx


class OnnxImportTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("importlib").util.find_spec("onnx"),
        "onnx optional dependency is not installed",
    )
    def test_imports_initializer_backed_and_dynamic_matmul(self):
        import onnx
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [
                helper.make_node(
                    "MatMul",
                    ["input", "weight"],
                    ["projection"],
                    name="projection",
                ),
                helper.make_node(
                    "MatMul",
                    ["projection", "dynamic_rhs"],
                    ["scores"],
                    name="attention_scores",
                ),
            ],
            "onnx-import-test",
            [
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, [1, 8, 16]
                ),
                helper.make_tensor_value_info(
                    "dynamic_rhs", TensorProto.FLOAT, [1, 16, 8]
                ),
            ],
            [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 8, 8])],
            [
                helper.make_tensor(
                    "weight",
                    TensorProto.FLOAT,
                    [16, 16],
                    [0.0] * (16 * 16),
                )
            ],
        )
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            onnx.save(model, path)
            imported = import_onnx(path, weight_bits=4)

        self.assertEqual(imported.operators[0].weight_name, "weight")
        self.assertEqual(imported.operators[0].weight_bits, 4)
        self.assertFalse(imported.operators[0].accuracy_sensitive)
        self.assertTrue(imported.operators[1].accuracy_sensitive)
