import queue
import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class InferenceJob:
    prompt_tokens: list[int]
    result: "queue.Queue[str | Exception]" = field(default_factory=queue.Queue)


class InferenceManager:
    """Serializes access to the single persistent model instance through one worker
    thread, so a blocking ONNX call never blocks the API's async event loop."""

    def __init__(self, generate_fn: Callable[[list[int]], str]):
        self._generate_fn = generate_fn
        self._queue: "queue.Queue[InferenceJob]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, prompt_tokens: list[int], timeout: float) -> str:
        job = InferenceJob(prompt_tokens=prompt_tokens)
        self._queue.put(job)
        try:
            outcome = job.result.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("inference did not complete within the request timeout")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                job.result.put(self._generate_fn(job.prompt_tokens))
            except Exception as exc:
                job.result.put(exc)
