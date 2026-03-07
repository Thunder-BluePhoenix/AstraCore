import * as vscode from 'vscode';
import * as cp from 'child_process';

/**
 * WebviewPanel that renders an SVG circuit diagram for the active AQL file.
 * Refreshes automatically on save.
 */
export class CircuitPanel {
    public static readonly viewType = 'astracore.circuit';

    private static current: CircuitPanel | undefined;

    private readonly panel: vscode.WebviewPanel;
    private readonly serverBin: string;
    private disposables: vscode.Disposable[] = [];

    private constructor(
        panel: vscode.WebviewPanel,
        serverBin: string,
        context: vscode.ExtensionContext,
    ) {
        this.panel = panel;
        this.serverBin = serverBin;

        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);

        // Refresh on save
        this.disposables.push(
            vscode.workspace.onDidSaveTextDocument((doc) => {
                if (doc.languageId === 'aql') {
                    this.update(doc.uri.fsPath);
                }
            }),
        );

        // Initial load from active editor
        const editor = vscode.window.activeTextEditor;
        if (editor && editor.document.languageId === 'aql') {
            this.update(editor.document.uri.fsPath);
        } else {
            this.panel.webview.html = this.getHtml('<p>Open an .aql file and save to render circuit.</p>');
        }
    }

    public static show(context: vscode.ExtensionContext, serverBin: string): void {
        if (CircuitPanel.current) {
            CircuitPanel.current.panel.reveal(vscode.ViewColumn.Beside);
            return;
        }
        const panel = vscode.window.createWebviewPanel(
            CircuitPanel.viewType,
            'AQL Circuit Diagram',
            vscode.ViewColumn.Beside,
            { enableScripts: true, retainContextWhenHidden: true },
        );
        CircuitPanel.current = new CircuitPanel(panel, serverBin, context);
    }

    private update(filePath: string): void {
        // Run `astracore report` in JSON mode (piped) or extract SVG via serve
        // For simplicity: use `astracore run --format json` if available,
        // otherwise fall back to a placeholder message.
        const proc = cp.spawnSync(this.serverBin, ['run', filePath, '--format', 'json'], {
            encoding: 'utf8',
            timeout: 10_000,
        });

        if (proc.error || proc.status !== 0) {
            // Try to get SVG from plain run output
            this.panel.webview.html = this.getHtml(
                `<pre>${escapeHtml(proc.stderr || proc.stdout || 'Run astracore to generate circuit diagram.')}</pre>`,
            );
            return;
        }

        try {
            const data = JSON.parse(proc.stdout);
            const svg: string = data.circuit_svg || '';
            this.panel.webview.html = this.getHtml(svg || '<p>No circuit SVG in output.</p>');
        } catch {
            this.panel.webview.html = this.getHtml(
                `<pre>${escapeHtml(proc.stdout)}</pre>`,
            );
        }
    }

    private getHtml(body: string): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AQL Circuit Diagram</title>
  <style>
    body { background: var(--vscode-editor-background); color: var(--vscode-editor-foreground);
           font-family: var(--vscode-editor-font-family); padding: 16px; }
    svg { max-width: 100%; height: auto; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>${body}</body>
</html>`;
    }

    private dispose(): void {
        CircuitPanel.current = undefined;
        this.panel.dispose();
        this.disposables.forEach((d) => d.dispose());
        this.disposables = [];
    }
}

function escapeHtml(s: string): string {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
