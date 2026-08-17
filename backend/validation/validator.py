class ValidationError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def validate_complete_request(file_content: str, cursor_offset: int) -> None:
    if cursor_offset < 0 or cursor_offset > len(file_content):
        raise ValidationError("cursor_offset is out of bounds for file_content")
    if not file_content[:cursor_offset].strip():
        raise ValidationError("no content before the cursor to complete from")
