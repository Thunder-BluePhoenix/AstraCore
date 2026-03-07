# AstraCore Plugin Marketplace — Design Specification

## Overview

The Plugin Marketplace is a future registry for community-contributed AstraCore plugins:
gate libraries, optimizer passes, and simulation backends. This document defines the registry
format, submission process, and planned CLI integration.

---

## Plugin Types

| Type        | Interface                     | Description                                          |
|-------------|-------------------------------|------------------------------------------------------|
| `gate`      | `GatePlugin` trait            | New quantum gate definitions (matrix + decomposition)|
| `optimizer` | `OptimizerPlugin` trait       | Peephole / circuit-level optimization passes         |
| `backend`   | `BackendPlugin` trait         | Alternative simulation engines (GPU, distributed, …) |

These traits are defined in `src/plugins/` and documented in `docs/aql_spec.md`.

---

## Registry Format

The marketplace is a single JSON index file hosted at a stable URL (or bundled in the binary).

```json
{
  "schema_version": 1,
  "plugins": [
    {
      "name": "qft-gate-lib",
      "version": "0.2.1",
      "type": "gate",
      "description": "QFT and IQFT gate library for AQL circuits",
      "author": "Example Author",
      "license": "MIT",
      "repository": "https://github.com/example/qft-gate-lib",
      "download_url": "https://example.com/qft-gate-lib-0.2.1.tar.gz",
      "sha256": "abc123...",
      "astracore_min_version": "0.3.0",
      "tags": ["qft", "fourier", "gates"]
    }
  ]
}
```

### Fields

| Field                    | Required | Description                                      |
|--------------------------|----------|--------------------------------------------------|
| `name`                   | Yes      | Unique plugin name (kebab-case)                  |
| `version`                | Yes      | Semantic version (MAJOR.MINOR.PATCH)             |
| `type`                   | Yes      | `gate`, `optimizer`, or `backend`                |
| `description`            | Yes      | One-sentence description (<100 chars)            |
| `author`                 | Yes      | Author name or GitHub handle                     |
| `license`                | Yes      | SPDX license identifier                          |
| `repository`             | Yes      | Source code URL                                  |
| `download_url`           | Yes      | URL to `.tar.gz` containing `plugin.toml` + Rust source |
| `sha256`                 | Yes      | SHA-256 checksum of the download archive         |
| `astracore_min_version`  | No       | Minimum AstraCore version required               |
| `tags`                   | No       | Searchable tags                                  |

---

## Plugin Package Format

A plugin is a `.tar.gz` archive with the following structure:

```
qft-gate-lib-0.2.1/
  plugin.toml          — plugin metadata (mirrors registry entry)
  src/
    lib.rs             — Rust implementation of the plugin trait
  Cargo.toml           — dependency manifest
  README.md            — usage guide
  LICENSE
```

`plugin.toml` example:

```toml
[plugin]
name = "qft-gate-lib"
version = "0.2.1"
type = "gate"
description = "QFT and IQFT gate library for AQL circuits"
author = "Example Author"
license = "MIT"

[plugin.entry]
# Rust symbol name of the plugin factory function
factory = "create_plugin"
```

---

## Planned CLI Integration

```bash
# Search the marketplace
astracore plugin search qft

# Install a plugin
astracore plugin install qft-gate-lib

# List installed plugins
astracore plugin list

# Remove a plugin
astracore plugin remove qft-gate-lib

# Show plugin details
astracore plugin info qft-gate-lib
```

Plugins are installed to `~/.astracore/plugins/<name>-<version>/`.
At startup, AstraCore scans this directory and loads all valid plugins via the dynamic
plugin registry (`src/plugins/registry.rs`).

---

## Submission Process

1. Create your plugin following the package format above.
2. Test with `cargo test` and `astracore plugin test ./my-plugin/`.
3. Publish source to a public Git repository.
4. Submit a Pull Request to the marketplace registry repository adding your entry to `index.json`.
5. Automated CI validates:
   - `sha256` checksum matches the download
   - Plugin compiles against the minimum declared AstraCore version
   - `cargo test` passes with no failures
6. Maintainer review and merge.

---

## Security Model

- All plugins are **compiled from source** — no pre-compiled blobs are distributed.
- Checksums are verified before compilation.
- Plugins run **in-process** — the user accepts responsibility for third-party code.
- Future: sandboxed execution via WebAssembly component model.

---

## Roadmap

| Milestone | Description                                             |
|-----------|---------------------------------------------------------|
| v0.4.0    | `astracore plugin install` (compile + install flow)     |
| v0.5.0    | Hosted index.json at `plugins.astracore.dev`            |
| v0.6.0    | Web UI for marketplace browsing                         |
| v1.0.0    | WASM plugin sandbox + signed packages                   |
