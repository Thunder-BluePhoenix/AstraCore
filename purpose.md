# AstraCore Purpose

## Why AstraCore Exists
AstraCore is built to make **quantum circuit simulation and analysis practical for everyday engineering workflows**.  
Most quantum tools are either:
- focused only on academic experimentation, or
- hard to integrate into production-style developer tooling.

AstraCore focuses on a middle path:
- fast local simulation,
- clear CLI and library APIs,
- optimization + analysis in one runtime,
- developer-facing interfaces (TUI, HTML report, local server).

## Why You Need AstraCore
Use AstraCore when you want to:
- prototype and validate quantum circuits quickly on local machines,
- benchmark/optimize gate sequences before larger experiments,
- generate readable reports for review and collaboration,
- embed simulation into Rust systems without external service dependencies.

In short, AstraCore is useful when you need a **developer-first quantum runtime** rather than only a research notebook workflow.

## Core Value Proposition
- One engine for **run + optimize + analyze + visualize**.
- Local-first workflow with no mandatory cloud dependency.
- Rust-native performance and composability.
- Multiple output surfaces: terminal, TUI, HTML, and HTTP dashboard.

## Pros
- High-performance Rust implementation.
- CLI is practical and script-friendly (`run`, `opt`, `analyze`, `report`, `serve`).
- Supports both interactive and automated workflows.
- Can be used as a library (`astracore::core::Simulator`) inside Rust projects.
- Includes optimization and analysis pipeline, not just raw simulation.

## Cons
- Early-stage project (API/CLI may evolve).
- Simulator-based approach still bounded by classical hardware limits.
- Team ecosystem and third-party integrations are still growing.
- Requires users to learn AQL instruction model and conventions.

## Best Fit
- Quantum algorithm prototyping teams.
- Engineering teams building internal quantum tooling.
- Developers needing reproducible local simulation + reporting.

## Not Ideal For
- Massive distributed quantum simulation workloads out of the box.
- Teams expecting long-term stable enterprise API guarantees from day one.
