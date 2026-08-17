from dataclasses import dataclass

MAX_PREFIX_CHARS = 4000


@dataclass(frozen=True)
class EditorContext:
    prefix: str
    filename: str | None


def collect_context(file_content: str, cursor_offset: int, filename: str | None = None) -> EditorContext:
    prefix = file_content[:cursor_offset]
    if len(prefix) > MAX_PREFIX_CHARS:
        prefix = prefix[-MAX_PREFIX_CHARS:]
    return EditorContext(prefix=prefix, filename=filename)
