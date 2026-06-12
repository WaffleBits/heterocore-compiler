# HeteroCore Integration

The compiler execution plan is the shared contract for the project:

1. `heterocore-compiler` partitions a graph and emits
   `heterocore.execution_plan.v1`.
2. `heterocore-analog-sim` sweeps non-idealities for analog-mapped operators.
3. `heterocore-memory` models SRAM traffic and generates OpenRAM configs.
4. `heterocore-rtl` encodes the plan into the controller instruction format.
5. `heterocore-fpga` loads the same instruction format into initialized memory.

With all repositories cloned as siblings:

```bash
cd heterocore-compiler
pip install -e .
heterocore-compile examples/tiny_transformer.json \
  -o results/tiny_transformer.plan.json

cd ../heterocore-analog-sim
pip install -e .
heterocore-analog-sim \
  ../heterocore-compiler/results/tiny_transformer.plan.json \
  -o results/tiny_transformer_sweep.json \
  --csv results/tiny_transformer_sweep.csv

cd ../heterocore-memory
pip install -e .
heterocore-memory \
  ../heterocore-compiler/results/tiny_transformer.plan.json \
  -o results/tiny_transformer_memory.json \
  --openram-dir generated/openram

cd ../heterocore-rtl
python tools/generate_schedule.py \
  ../heterocore-compiler/results/tiny_transformer.plan.json \
  -o results/tiny_transformer_schedule.hex \
  --manifest results/tiny_transformer_schedule.json
```

Each result carries an explicit simulated or analytical claim scope.

