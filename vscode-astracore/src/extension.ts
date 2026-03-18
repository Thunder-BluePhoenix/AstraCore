import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';
import { CircuitPanel } from './circuitPanel';
import { StatePanel } from './statePanel';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext): void {
    // ── LSP: start astracore lsp on stdio ──────────────────────────────────
    const serverBin: string =
        vscode.workspace.getConfiguration('astracore').get('serverPath', 'astracore');

    const serverOptions: ServerOptions = {
        command: serverBin,
        args: ['lsp'],
        transport: TransportKind.stdio,
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'aql' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.aql'),
        },
    };

    client = new LanguageClient(
        'aql-language-server',
        'AQL Language Server',
        serverOptions,
        clientOptions,
    );

    client.start();
    context.subscriptions.push(client);

    // ── Commands ─────────────────────────────────────────────────────────────
    context.subscriptions.push(
        vscode.commands.registerCommand('astracore.showCircuit', () => {
            CircuitPanel.show(context, serverBin);
        }),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('astracore.showState', () => {
            StatePanel.show(context, serverBin);
        }),
    );
}

export async function deactivate(): Promise<void> {
    if (client) {
        await client.stop();
    }
}
