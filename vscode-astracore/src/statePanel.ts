import * as vscode from 'vscode';
import * as cp from 'child_process';

interface RunResult {
    final_probabilities?: number[];
    final_amplitudes?: Array<{ re: number; im: number }>;
    measurements?: Array<{ qubit: number; outcome: number }>;
    num_qubits?: number;
}

/**
 * WebviewPanel that renders a probability bar chart and amplitude table
 * for the active AQL file. Refreshes automatically on save.
 */
export class StatePanel {
    public static readonly viewType = 'astracore.state';

    private static current: StatePanel | undefined;

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

        this.disposables.push(
            vscode.workspace.onDidSaveTextDocument((doc) => {
                if (doc.languageId === 'aql') {
                    this.update(doc.uri.fsPath);
                }
            }),
        );

        const editor = vscode.window.activeTextEditor;
        if (editor && editor.document.languageId === 'aql') {
            this.update(editor.document.uri.fsPath);
        } else {
            this.panel.webview.html = this.getHtml(null);
        }
    }

    public static show(context: vscode.ExtensionContext, serverBin: string): void {
        if (StatePanel.current) {
            StatePanel.current.panel.reveal(vscode.ViewColumn.Beside);
            return;
        }
        const panel = vscode.window.createWebviewPanel(
            StatePanel.viewType,
            'AQL State Inspector',
            vscode.ViewColumn.Beside,
            { enableScripts: true, retainContextWhenHidden: true },
        );
        StatePanel.current = new StatePanel(panel, serverBin, context);
    }

    private update(filePath: string): void {
        const proc = cp.spawnSync(this.serverBin, ['run', filePath, '--format', 'json'], {
            encoding: 'utf8',
            timeout: 10_000,
        });

        if (proc.error || proc.status !== 0) {
            this.panel.webview.html = this.getHtml(null,
                proc.stderr || proc.stdout || 'Run astracore to inspect state.');
            return;
        }

        try {
            const data: RunResult = JSON.parse(proc.stdout);
            this.panel.webview.html = this.getHtml(data);
        } catch {
            this.panel.webview.html = this.getHtml(null, proc.stdout);
        }
    }

    private getHtml(data: RunResult | null, errorText?: string): string {
        let body: string;

        if (errorText) {
            body = `<pre>${escapeHtml(errorText)}</pre>`;
        } else if (!data) {
            body = '<p>Open an .aql file and save to inspect quantum state.</p>';
        } else {
            const probs = data.final_probabilities ?? [];
            const amps  = data.final_amplitudes  ?? [];
            const nq    = data.num_qubits ?? Math.round(Math.log2(probs.length || 1));

            const bars = probs.map((p, i) => {
                const label = i.toString(2).padStart(nq, '0');
                const pct   = (p * 100).toFixed(2);
                return `<tr>
  <td>|${label}⟩</td>
  <td class="prob-cell">
    <div class="bar" style="width:${(p * 100).toFixed(1)}%"></div>
    <span>${pct}%</span>
  </td>
</tr>`;
            }).join('\n');

            const ampRows = amps.map((a, i) => {
                const label = i.toString(2).padStart(nq, '0');
                const re    = a.re.toFixed(6);
                const im    = a.im >= 0 ? `+${a.im.toFixed(6)}i` : `${a.im.toFixed(6)}i`;
                return `<tr><td>|${label}⟩</td><td>${re} ${im}</td></tr>`;
            }).join('\n');

            const measRows = (data.measurements ?? []).map(m =>
                `<tr><td>q${m.qubit}</td><td>${m.outcome}</td></tr>`
            ).join('\n');

            body = `
<h2>Probability Distribution</h2>
<table class="prob-table">
  <thead><tr><th>State</th><th>Probability</th></tr></thead>
  <tbody>${bars}</tbody>
</table>

${ampRows ? `<h2>Amplitudes</h2>
<table>
  <thead><tr><th>State</th><th>Amplitude</th></tr></thead>
  <tbody>${ampRows}</tbody>
</table>` : ''}

${measRows ? `<h2>Measurements</h2>
<table>
  <thead><tr><th>Qubit</th><th>Outcome</th></tr></thead>
  <tbody>${measRows}</tbody>
</table>` : ''}`;
        }

        return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AQL State Inspector</title>
  <style>
    body { background: var(--vscode-editor-background); color: var(--vscode-editor-foreground);
           font-family: var(--vscode-editor-font-family); padding: 16px; font-size: 13px; }
    h2 { color: var(--vscode-textLink-foreground); margin-top: 24px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
    th, td { padding: 4px 8px; border: 1px solid var(--vscode-editorGroup-border); }
    th { background: var(--vscode-editorGroupHeader-tabsBackground); }
    .prob-cell { position: relative; min-width: 200px; }
    .bar { height: 16px; background: var(--vscode-progressBar-background, #007acc);
           border-radius: 2px; display: inline-block; vertical-align: middle; }
    .prob-cell span { margin-left: 8px; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>${body}</body>
</html>`;
    }

    private dispose(): void {
        StatePanel.current = undefined;
        this.panel.dispose();
        this.disposables.forEach((d) => d.dispose());
        this.disposables = [];
    }
}

function escapeHtml(s: string): string {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
