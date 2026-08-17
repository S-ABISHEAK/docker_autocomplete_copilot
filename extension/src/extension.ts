import * as vscode from "vscode";
import { DockerfileCompletionProvider } from "./providers/completionProvider";

export function activate(context: vscode.ExtensionContext): void {
  const provider = new DockerfileCompletionProvider();
  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider({ language: "dockerfile" }, provider)
  );
}

export function deactivate(): void {}
