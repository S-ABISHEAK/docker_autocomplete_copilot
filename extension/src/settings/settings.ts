import * as vscode from "vscode";

export interface Settings {
  backendUrl: string;
  requestTimeoutMs: number;
  debounceMs: number;
  enabled: boolean;
}

export function readSettings(): Settings {
  const config = vscode.workspace.getConfiguration("dockerfileAutocomplete");
  return {
    backendUrl: config.get<string>("backendUrl", "http://127.0.0.1:8123"),
    requestTimeoutMs: config.get<number>("requestTimeoutMs", 5000),
    debounceMs: config.get<number>("debounceMs", 300),
    enabled: config.get<boolean>("enabled", true),
  };
}
