import unittest

from heterocore_compiler import HardwareConfig, PartitionPolicy, compile_graph
from heterocore_compiler.model import ModelGraph, Operator


class CompilerTests(unittest.TestCase):
    def test_partitions_supported_large_matrix(self):
        graph = ModelGraph(
            "test",
            (
                Operator("large", "linear", 32, 128, 128, 4),
                Operator("norm", "layer_norm", 32, 128, 128, 16, True),
            ),
        )
        plan = compile_graph(
            graph,
            HardwareConfig(),
            PartitionPolicy(minimum_analog_macs=1_000),
        )
        self.assertEqual(plan["operators"][0]["target"], "analog")
        self.assertEqual(plan["operators"][1]["target"], "digital")
        self.assertGreater(plan["summary"]["analog_mac_fraction"], 0)

    def test_precision_policy_keeps_high_precision_op_digital(self):
        graph = ModelGraph("test", (Operator("matmul", "matmul", 64, 64, 64, 16),))
        plan = compile_graph(graph)
        self.assertEqual(plan["operators"][0]["target"], "digital")
        self.assertIn("precision", plan["operators"][0]["reason"])

    def test_analog_mapping_reduces_weight_traffic(self):
        graph = ModelGraph("test", (Operator("linear", "linear", 64, 256, 256, 4),))
        plan = compile_graph(graph)
        summary = plan["summary"]
        self.assertGreater(summary["projected_memory_traffic_reduction"], 0)
        self.assertGreater(summary["projected_energy_reduction"], 0)

    def test_peripheral_energy_is_reported(self):
        graph = ModelGraph("test", (Operator("linear", "linear", 64, 256, 256, 4),))
        plan = compile_graph(graph)
        estimate = plan["operators"][0]["estimate"]
        self.assertGreater(estimate["peripheral_energy_pj"], 0)
        self.assertIn("energy_breakdown_pj", plan["summary"])

    def test_expensive_adc_can_erase_analog_advantage(self):
        graph = ModelGraph("test", (Operator("linear", "linear", 64, 256, 256, 4),))
        plan = compile_graph(
            graph,
            HardwareConfig(adc_conversion_energy_pj=500.0),
        )
        self.assertLess(plan["summary"]["projected_energy_reduction"], 0)


if __name__ == "__main__":
    unittest.main()
