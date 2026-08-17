import * as vscode from "vscode";
import { requestCompletion } from "../api/client";
import { readSettings } from "../settings/settings";

function delay(ms: number, token: vscode.CancellationToken): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    token.onCancellationRequested(() => {
      clearTimeout(timer);
      resolve();
    });
  });
}

function toAbortSignal(token: vscode.CancellationToken): AbortSignal {
  const controller = new AbortController();
  token.onCancellationRequested(() => controller.abort());
  return controller.signal;
}

export class DockerfileCompletionProvider implements vscode.InlineCompletionItemProvider {
  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionItem[] | undefined> {
    const settings = readSettings();
    if (!settings.enabled) {
      return undefined;
    }

    await delay(settings.debounceMs, token);
    if (token.isCancellationRequested) {
      return undefined;
    }

    const result = await requestCompletion(
      settings.backendUrl,
      { fileContent: document.getText(), cursorOffset: document.offsetAt(position) },
      settings.requestTimeoutMs,
      toAbortSignal(token)
    );

    if (token.isCancellationRequested || !result.ok || result.completion.trim().length === 0) {
      return undefined;
    }

    return [new vscode.InlineCompletionItem(result.completion, new vscode.Range(position, position))];
  }
}
