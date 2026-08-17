from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.api.schemas import CompleteRequest, CompleteResponse
from backend.config import load_settings
from backend.context.collector import collect_context
from backend.inference.manager import InferenceManager
from backend.inference.model_service import ModelService
from backend.prompt.builder import build_prompt
from backend.validation.validator import ValidationError, validate_complete_request

settings = load_settings()
model_service = ModelService(
    onnx_path=settings.onnx_path,
    block_size=settings.block_size,
    max_new_tokens=settings.max_new_tokens,
    eot_token_id=settings.eot_token_id,
)
inference_manager = InferenceManager(generate_fn=model_service.generate)

app = FastAPI(title="docker-autocomplete-backend")


@app.get("/health")
def health() -> dict:
    return {"status": "ready"}


@app.post("/complete", response_model=CompleteResponse)
def complete(request: CompleteRequest):
    try:
        validate_complete_request(request.file_content, request.cursor_offset)
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"error": "invalid_request", "reason": exc.reason})

    context = collect_context(request.file_content, request.cursor_offset, request.filename)
    prompt_tokens = build_prompt(context.prefix, settings.context_window_tokens)

    try:
        completion = inference_manager.submit(prompt_tokens, timeout=settings.request_timeout_seconds)
    except TimeoutError as exc:
        return JSONResponse(status_code=504, content={"error": "timeout", "reason": str(exc)})

    return CompleteResponse(completion=completion)
