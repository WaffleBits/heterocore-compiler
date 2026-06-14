# HeteroCore Compiler

[![CI](https://github.com/WaffleBits/heterocore-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/WaffleBits/heterocore-compiler/actions/workflows/ci.yml)

Compiler and analytical cost model for mixed analog-digital AI inference. It
partitions a model graph, explains every placement decision, and emits the
versioned execution plan consumed by the other HeteroCore repositories.

> Status: architecture simulation and digital implementation prototype. All
> performance and energy values are projections, not measured silicon results.

## System Context

```mermaid
flowchart LR
    C[heterocore-compiler] --> A[heterocore-analog-sim]
    C --> M[heterocore-memory]
    C --> R[heterocore-rtl]
    R --> F[heterocore-fpga]
```

## What It Demonstrates

- Framework-neutral JSON graph ingestion and direct ONNX import.
- Explainable analog-versus-digital partitioning.
- Configurable array, digital MAC, clock, and energy assumptions.
- Per-operator cycle, energy, and memory-traffic estimates.
- Explicit DAC, ADC, accumulation, calibration, control, and interconnect energy.
- A JSON Schema-backed cross-repository execution plan.
- Automated tests and a reproducible sample workload.
- Explainable memory-centric decode placement for model weights, KV state,
  summaries, gathers, normalization, and projection/MLP execution.

## Checked-In Evidence

`results/tiny_transformer.plan.json` is generated from the sample transformer
block with default hardware assumptions:

| Metric | Result |
| --- | ---: |
| operators | 11 |
| analog-mapped operators | 6 |
| analog MAC fraction | 96.0% |
| projected energy reduction vs. all-digital model | 68.30% |
| projected memory-traffic reduction | 15.79% |

These are analytical cost-model outputs. The input, assumptions, placement
reasons, and per-operator estimates are all present in the result file.

The ONNX transformer result is deliberately less favorable: 81.7% of MACs map
to analog, but the nominal peripheral-aware model projects only a 7.4% energy
reduction. `results/energy_sensitivity.json` shows how that conclusion changes
under optimistic, nominal, and conservative converter/control assumptions.

`results/qwen_decode.plan.json` adds the memory-centric path. For an 8,192-token
context at batch 1, it places 229,376 bytes of block summaries in local SRAM,
pages the 58,720,256-byte INT4 KV cache externally, allocates a 2 MiB active
weight tile, and routes 14,680,064 selected KV bytes through the gather engine.
All ten decisions include a human-readable reason.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
heterocore-compile examples/tiny_transformer.json \
  -o results/tiny_transformer.plan.json
```

On PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Compile the checked-in ONNX transformer:

```bash
pip install -e ".[onnx]"
heterocore-compile examples/tiny_char_transformer.onnx \
  -o results/tiny_char_transformer.plan.json \
  --onnx-weight-bits 4 \
  --minimum-analog-macs 16000
```

The command prints the analog MAC fraction and projected energy reduction. The
full assumptions and per-operator decisions are saved in the output plan.

Compile the checked Qwen2.5-1.5B-Instruct decode placement:

```bash
heterocore-compile-decode \
  --context 8192 \
  --batch 1 \
  --weight-bits 4 \
  --kv-bits 4 \
  --selected-block-fraction 0.25 \
  --output results/qwen_decode.plan.json
```

The decode plan explains where weights, KV pages, block summaries, active
vectors, selection, gather, fused normalization, and matrix-vector work are
placed. It optimizes logical bytes per output token under a local-SRAM
constraint; it is an analytical plan rather than measured hardware behavior.

## Input Format

The minimal graph format is intentionally small:

```json
{
  "model": {"name": "example"},
  "operators": [
    {
      "id": "projection",
      "type": "linear",
      "dimensions": {"m": 128, "k": 256, "n": 256},
      "weight_bits": 4
    }
  ]
}
```

ONNX models are imported directly using ONNX shape inference. Initializer-backed
matrix operations preserve their weight names in the execution plan so the
analog simulator can replace the exact selected tensors.

## Reproduce the Evidence

```bash
python -m unittest discover -s tests
heterocore-compile examples/tiny_transformer.json \
  -o results/tiny_transformer.plan.json
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for partitioning and cost-model details.
See [docs/INTEGRATION.md](docs/INTEGRATION.md) for the five-repository workflow.
See [docs/BRANDING.md](docs/BRANDING.md) for the temporary project-name policy.

## Claim Boundaries

This project can test compiler behavior and compare architecture assumptions.
It does not establish fabrication yield, physical analog noise, thermal
behavior, ADC/DAC energy, or real tokens per watt.
