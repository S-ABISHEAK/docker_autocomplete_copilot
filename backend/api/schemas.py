from pydantic import BaseModel


class CompleteRequest(BaseModel):
    file_content: str
    cursor_offset: int
    filename: str | None = None


class CompleteResponse(BaseModel):
    completion: str
