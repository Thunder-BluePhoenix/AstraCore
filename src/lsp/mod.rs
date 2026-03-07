/// AQL Language Server Protocol (LSP) and Debug Adapter Protocol (DAP) support.
///
/// Enabled via `--features lsp`. Entry points:
///   `astracore lsp` → LSP server on stdin/stdout (used by VS Code extension)
///   `astracore dap` → DAP server on stdin/stdout (used by VS Code debugger)

pub mod completion;
pub mod diagnostics;
pub mod goto;
pub mod hover;

#[cfg(feature = "lsp")]
pub mod debug_adapter;
#[cfg(feature = "lsp")]
pub mod server;

// ── Public entry points ───────────────────────────────────────────────────

/// Start the AQL Language Server on stdin/stdout (blocks until shutdown).
#[cfg(feature = "lsp")]
pub async fn run_lsp() {
    server::run().await;
}

/// Start the AQL Debug Adapter on stdin/stdout (blocks until disconnected).
#[cfg(feature = "lsp")]
pub async fn run_dap() {
    debug_adapter::serve().await;
}
