# How To Use AstraCore

## 1. Build and Run

From project root:

```bash
cargo build
cargo run
```

`cargo run` (without command) runs built-in demos.

## 2. CLI Commands

```bash
astracore demo
astracore run <file.aql>
astracore opt <file.aql>
astracore analyze <file.aql>
astracore dash <file.aql>
astracore report <file.aql> [output.html]
astracore serve <file.aql> [port]
astracore help
```

If `astracore` is not in PATH yet, use:

```bash
cargo run -- <command> ...
```

Example:

```bash
cargo run -- run examples/bell.aql
cargo run -- analyze examples/bell.aql
cargo run -- report examples/bell.aql bell_report.html
cargo run -- serve examples/bell.aql 8080
```

## 3. Minimal AQL Example

Create `bell.aql`:

```txt
QREG 2
H 0
CNOT 0 1
MEASURE_ALL
```

Run it:

```bash
astracore run bell.aql
```

## 4. Use AstraCore as a Rust Library

```rust
use astracore::core::Simulator;

fn main() {
    let mut sim = Simulator::new(2);
    sim.h(0).cnot(0, 1);
    let probs = sim.probabilities();
    println!("P(|00>) = {:.4}", probs[0]);
    println!("P(|11>) = {:.4}", probs[3]);
}
```

## 5. Generate Visual Outputs

### TUI Dashboard

```bash
astracore dash bell.aql
```

### HTML Report

```bash
astracore report bell.aql bell_report.html
```

### Local HTTP Dashboard

```bash
astracore serve bell.aql 8080
```

Then open `http://localhost:8080`.

## 6. Optimization and Analysis Workflow

Recommended flow for real projects:

1. `analyze` to inspect circuit shape/depth.
2. `opt` to reduce gate count.
3. `run` to validate behavior.
4. `report` for sharing and review.

## 7. Benchmarking

Run included benchmark:

```bash
cargo bench
```

## 8. Troubleshooting

- If a file fails to load, confirm correct path and extension (`.aql`).
- If a command fails, run `astracore help` for expected arguments.
- If TUI/serve fails, check terminal/network port availability.
