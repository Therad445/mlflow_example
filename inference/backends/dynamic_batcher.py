from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np

from inference.backends.onnx_backend import OnnxEmbedder
from inference.config import settings
from inference.pooling import l2_normalize_numpy


@dataclass(slots=True)
class QueueItem:
    texts: list[str]
    prompt: str | None
    normalize: bool | None
    future: Future


class DynamicBatchingService:
    def __init__(self) -> None:
        self.embedder = OnnxEmbedder()
        self.queue: queue.Queue[QueueItem] = queue.Queue()
        self.max_batch_size = settings.batch_max_size
        self.timeout_ms = settings.batch_timeout_ms
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.backend_name = f"onnxruntime-dynamic-batching(max={self.max_batch_size},timeout_ms={self.timeout_ms})"

    def stop(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=1.0)

    def submit(self, texts: list[str], prompt: str | None, normalize: bool | None) -> Future:
        future: Future = Future()
        self.queue.put(QueueItem(texts=texts, prompt=prompt, normalize=normalize, future=future))
        return future

    def _collect_batch(self, first_item: QueueItem) -> list[QueueItem]:
        batch = [first_item]
        batch_items = len(first_item.texts)
        deadline = time.perf_counter() + self.timeout_ms / 1000.0

        while batch_items < self.max_batch_size:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                item = self.queue.get(timeout=timeout)
            except queue.Empty:
                break

            projected_size = batch_items + len(item.texts)
            if projected_size > self.max_batch_size:
                self.queue.put(item)
                break

            batch.append(item)
            batch_items = projected_size

        return batch

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                first_item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            items = self._collect_batch(first_item)
            prepared_texts: list[str] = []
            lengths: list[int] = []

            for item in items:
                prefix = settings.default_prompt if item.prompt is None else item.prompt
                current = [f"{prefix}{text}" if prefix else text for text in item.texts]
                prepared_texts.extend(current)
                lengths.append(len(item.texts))

            try:
                embeddings = np.asarray(self.embedder._run_prepared(prepared_texts), dtype=np.float32)
                offset = 0
                for item, length in zip(items, lengths, strict=True):
                    current = embeddings[offset: offset + length].copy()
                    do_normalize = settings.normalize_default if item.normalize is None else item.normalize
                    if do_normalize:
                        current = l2_normalize_numpy(current)
                    item.future.set_result(current.astype(np.float32).tolist())
                    offset += length
            except Exception as exc:
                for item in items:
                    item.future.set_exception(exc)
