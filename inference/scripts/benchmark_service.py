from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

import httpx
import psutil


def make_payload(texts_per_request: int, repeat_words: int) -> dict:
    text = " ".join(["тестовый"] * repeat_words)
    texts = [f"{text} {idx}" for idx in range(texts_per_request)]
    return {"texts": texts, "normalize": True}


async def wait_until_ready(base_url: str, timeout_sec: float = 180.0) -> dict:
    deadline = time.perf_counter() + timeout_sec
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.perf_counter() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                response.raise_for_status()
                return response.json()
            except Exception:
                await asyncio.sleep(1.0)
    raise TimeoutError(f"Service {base_url} did not become ready in time")


async def run_load(base_url: str, total_requests: int, concurrency: int, payload: dict, warmup_requests: int) -> dict:
    latencies_ms: list[float] = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=120.0) as client:
        for _ in range(warmup_requests):
            await client.post(f"{base_url}/embed", json=payload)

        started_at = time.perf_counter()

        async def one_call() -> None:
            async with sem:
                t0 = time.perf_counter()
                response = await client.post(f"{base_url}/embed", json=payload)
                response.raise_for_status()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(latency_ms)

        await asyncio.gather(*[one_call() for _ in range(total_requests)])
        elapsed = time.perf_counter() - started_at

    latencies_ms.sort()

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        idx = int(round((len(values) - 1) * q))
        return values[idx]

    return {
        "requests_total": total_requests,
        "concurrency": concurrency,
        "elapsed_sec": elapsed,
        "throughput_rps": total_requests / elapsed if elapsed > 0 else 0.0,
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
            "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
            "min": min(latencies_ms) if latencies_ms else 0.0,
            "max": max(latencies_ms) if latencies_ms else 0.0,
        },
    }


class ResourceMonitor:
    def __init__(self, pid: int, sample_interval: float = 0.2) -> None:
        self.process = psutil.Process(pid)
        self.sample_interval = sample_interval
        self.stop_event = Event()
        self.peak_rss_mb = 0.0
        self.thread = Thread(target=self._run, daemon=True)
        self.cpu_time_before = self._cpu_time()
        self.cpu_time_after = self.cpu_time_before

    def _cpu_time(self) -> float:
        cpu_times = self.process.cpu_times()
        return float(cpu_times.user + cpu_times.system)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                rss_mb = self.process.memory_info().rss / (1024 * 1024)
                self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            except psutil.Error:
                break
            time.sleep(self.sample_interval)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.cpu_time_after = self._cpu_time()

    def summary(self, elapsed_sec: float) -> dict:
        cpu_time_delta = self.cpu_time_after - self.cpu_time_before
        cpu_one_core_percent = (cpu_time_delta / elapsed_sec) * 100.0 if elapsed_sec > 0 else 0.0
        cpu_all_cores_percent = cpu_one_core_percent / max(psutil.cpu_count(logical=True), 1)
        return {
            "peak_rss_mb": self.peak_rss_mb,
            "cpu_time_sec": cpu_time_delta,
            "cpu_percent_one_core": cpu_one_core_percent,
            "cpu_percent_all_cores": cpu_all_cores_percent,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True, help="Например inference.apps.baseline_app:app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--texts-per-request", type=int, default=1)
    parser.add_argument("--repeat-words", type=int, default=24)
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument("--output", default="artifacts/benchmark/result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            args.app,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    )

    base_url = f"http://{args.host}:{args.port}"

    try:
        health = asyncio.run(wait_until_ready(base_url))
        monitor = ResourceMonitor(pid=health.get("pid", process.pid))
        payload = make_payload(args.texts_per_request, args.repeat_words)

        monitor.start()
        result = asyncio.run(
            run_load(
                base_url=base_url,
                total_requests=args.requests,
                concurrency=args.concurrency,
                payload=payload,
                warmup_requests=args.warmup_requests,
            )
        )
        monitor.stop()

        result.update(
            {
                "service": health,
                "payload": {
                    "texts_per_request": args.texts_per_request,
                    "repeat_words": args.repeat_words,
                },
                "resources": monitor.summary(result["elapsed_sec"]),
            }
        )

        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
