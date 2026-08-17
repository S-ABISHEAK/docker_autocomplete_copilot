export interface CompletionRequest {
  fileContent: string;
  cursorOffset: number;
}

export type CompletionResult =
  | { ok: true; completion: string }
  | { ok: false; error: string };

export async function requestCompletion(
  baseUrl: string,
  request: CompletionRequest,
  timeoutMs: number,
  externalSignal: AbortSignal
): Promise<CompletionResult> {
  const controller = new AbortController();
  const onExternalAbort = () => controller.abort();
  externalSignal.addEventListener("abort", onExternalAbort);
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${baseUrl}/complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_content: request.fileContent,
        cursor_offset: request.cursorOffset,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = (await response.json()) as { error: string; reason: string };
      return { ok: false, error: body.reason ?? body.error };
    }

    const body = (await response.json()) as { completion: string };
    return { ok: true, completion: body.completion };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "unknown error" };
  } finally {
    clearTimeout(timeout);
    externalSignal.removeEventListener("abort", onExternalAbort);
  }
}
