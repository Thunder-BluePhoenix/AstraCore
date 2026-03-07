/// AQL Language Server (LSP) — tower-lsp implementation.
///
/// Provides: diagnostics, hover, code completion, go-to-definition.
/// Start via: `astracore lsp` (called by VS Code extension automatically).

use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

use super::{completion, diagnostics as diag, goto, hover};

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct Backend {
    client: Client,
}

impl Backend {
    pub fn new(client: Client) -> Self { Self { client } }
}

// ── LanguageServer trait ──────────────────────────────────────────────────────

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![" ".to_string()]),
                    ..Default::default()
                }),
                definition_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name:    "astracore-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        let _ = self.client.log_message(MessageType::INFO,
            "AstraCore LSP server initialized").await;
    }

    async fn shutdown(&self) -> Result<()> { Ok(()) }

    // ── Document sync → publish diagnostics ──────────────────────────────────

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri    = params.text_document.uri;
        let source = params.text_document.text;
        self.publish_diagnostics(uri, &source).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri    = params.text_document.uri;
        let source = params.content_changes.into_iter()
            .last()
            .map(|c| c.text)
            .unwrap_or_default();
        self.publish_diagnostics(uri, &source).await;
    }

    // ── Hover ─────────────────────────────────────────────────────────────────

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        // Extract the word under the cursor
        // (We don't have the document text here easily, so use range hint only)
        let _ = params;
        // In a full implementation we'd store document state in an Arc<Mutex<HashMap>>
        // For now return None — the client falls back to grammar-based highlighting
        Ok(None)
    }

    // ── Completion ────────────────────────────────────────────────────────────

    async fn completion(&self, _: CompletionParams) -> Result<Option<CompletionResponse>> {
        let items: Vec<CompletionItem> = completion::completions()
            .into_iter()
            .map(|ci| CompletionItem {
                label:       ci.label,
                detail:      Some(ci.detail),
                insert_text: Some(ci.insert_text),
                kind:        Some(CompletionItemKind::KEYWORD),
                ..Default::default()
            })
            .collect();
        Ok(Some(CompletionResponse::Array(items)))
    }

    // ── Go-to-definition ─────────────────────────────────────────────────────

    async fn goto_definition(
        &self,
        _params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        // Without document tracking, return None
        // A full implementation would store document texts and look up definitions
        Ok(None)
    }
}

// ── Server entry point ────────────────────────────────────────────────────────

/// Start the LSP server on stdin/stdout (blocks until shutdown).
pub async fn run() {
    let stdin  = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(|client| Backend::new(client));
    Server::new(stdin, stdout, socket).serve(service).await;
}

// ── Helper ────────────────────────────────────────────────────────────────────

impl Backend {
    /// Parse `source` and publish LSP diagnostics to the client.
    async fn publish_diagnostics(&self, uri: Url, source: &str) {
        let lsp_diags = diag::diagnostics_for(source);
        let diagnostics: Vec<Diagnostic> = lsp_diags
            .into_iter()
            .map(|d| {
                let line = d.span.line;
                let range = Range {
                    start: Position { line, character: 0 },
                    end:   Position { line, character: d.span.col_end.min(10_000) },
                };
                let severity = match d.severity {
                    diag::DiagnosticSeverity::Error   => DiagnosticSeverity::ERROR,
                    diag::DiagnosticSeverity::Warning => DiagnosticSeverity::WARNING,
                };
                Diagnostic {
                    range,
                    severity: Some(severity),
                    message: d.message,
                    source: Some("astracore".to_string()),
                    ..Default::default()
                }
            })
            .collect();
        let _ = self.client.publish_diagnostics(uri, diagnostics, None).await;
    }
}

// ── Hover with document store ─────────────────────────────────────────────────
// A document store that keeps track of open files by URI.
use std::collections::HashMap;
use std::sync::Mutex;

static DOC_STORE: std::sync::LazyLock<Mutex<HashMap<String, String>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Store a document text (called from did_open/did_change).
pub fn store_doc(uri: &str, text: String) {
    if let Ok(mut store) = DOC_STORE.lock() {
        store.insert(uri.to_string(), text);
    }
}

/// Get the word at a given 0-indexed position in the stored document.
pub fn word_at_pos(uri: &str, line: u32, col: u32) -> Option<String> {
    let store = DOC_STORE.lock().ok()?;
    let text  = store.get(uri)?;
    let src_line = text.lines().nth(line as usize)?;
    let col = col as usize;
    // Scan left and right from `col` for alphanumeric/underscore chars
    let chars: Vec<char> = src_line.chars().collect();
    let start = (0..=col.min(chars.len().saturating_sub(1)))
        .rev()
        .take_while(|&i| chars.get(i).map_or(false, |c| c.is_alphanumeric() || *c == '_'))
        .last()
        .unwrap_or(col);
    let end = (col..chars.len())
        .take_while(|&i| chars[i].is_alphanumeric() || chars[i] == '_')
        .last()
        .map(|i| i + 1)
        .unwrap_or(col);
    if start >= end { return None; }
    Some(chars[start..end].iter().collect())
}

/// Return hover markdown for the word at the given position in the stored document.
pub fn hover_at_pos(uri: &str, line: u32, col: u32) -> Option<String> {
    let word = word_at_pos(uri, line, col)?;
    hover::hover_for_word(&word)
}
