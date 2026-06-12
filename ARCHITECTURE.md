# Architecture

HeteroCore Compiler turns a framework-neutral graph description into
`heterocore.execution_plan.v1`. The plan is the contract between the five
HeteroCore repositories.

```mermaid
flowchart LR
    A[Model graph] --> B[Graph normalization]
    B --> C[Partition policy]
    C --> D[Analytical cost model]
    D --> E[Execution plan JSON]
    E --> F[Analog simulator]
    E --> G[Memory model]
    E --> H[RTL schedule generator]
    H --> I[FPGA shell]
```

## Partitioning

The current policy maps `linear` and `matmul` operators when they:

1. exceed a configurable MAC threshold;
2. use no more than the configured weight precision; and
3. are not explicitly marked accuracy-sensitive.

This rule-based policy is deliberately reviewable. It is a baseline for later
search-based or learned partitioners, not a claim of optimal placement.

## Cost model

The cost model estimates cycles, compute energy, and memory traffic from
explicit hardware parameters. Analog weights are treated as resident in the
crossbar, so mapped operations avoid weight reads in the traffic estimate.

All generated values are analytical projections. The schema carries
`claim_scope.measured_hardware=false` so downstream dashboards cannot silently
present them as lab measurements.

