# Inference HW: rubert-mini-frida

Отдельный модуль для ДЗ по оптимизации inference pipeline.

## Что внутри

- `apps/baseline_app.py` — базовый CPU inference через `transformers`
- `apps/onnx_app.py` — inference через `onnxruntime`
- `apps/batching_app.py` — inference через `onnxruntime` + dynamic batching
- `scripts/export_to_onnx.py` — экспорт модели в ONNX
- `scripts/benchmark_service.py` — автоматический запуск сервиса и бенчмарк
- `report_inference.md` — итоговый отчёт по бенчмаркам

## Установка

```bash
pip install -r inference/requirements.txt
```

## Часть 1. Базовый сервис

```bash
python -m uvicorn inference.apps.baseline_app:app --host 0.0.0.0 --port 8011
```

Проверка:

```bash
curl -X POST http://127.0.0.1:8011/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Привет, мир", "Как дела?"], "normalize": true}'
```

## Часть 2. Экспорт в ONNX и сервис

Сначала экспорт:

```bash
python -m inference.scripts.export_to_onnx
```

Потом запуск сервиса:

```bash
python -m uvicorn inference.apps.onnx_app:app --host 0.0.0.0 --port 8012
```

## Часть 3. Dynamic batching

```bash
python -m uvicorn inference.apps.batching_app:app --host 0.0.0.0 --port 8013
```

По умолчанию:

- `DYN_BATCH_MAX_SIZE=32`
- `DYN_BATCH_TIMEOUT_MS=8`

Их можно менять через переменные окружения.

## Бенчмарки

### Baseline

```bash
python -m inference.scripts.benchmark_service \
  --app inference.apps.baseline_app:app \
  --port 8011 \
  --requests 300 \
  --concurrency 16 \
  --output artifacts/benchmark/baseline.json
```

### ONNX

```bash
python -m inference.scripts.benchmark_service \
  --app inference.apps.onnx_app:app \
  --port 8012 \
  --requests 300 \
  --concurrency 16 \
  --output artifacts/benchmark/onnx.json
```

### Dynamic batching

```bash
python -m inference.scripts.benchmark_service \
  --app inference.apps.batching_app:app \
  --port 8013 \
  --requests 300 \
  --concurrency 16 \
  --output artifacts/benchmark/batching.json
```

## Какие метрики использовать

- `latency p50/p95/p99` — чтобы видеть не только среднее, но и хвосты
- `throughput (requests/sec)` — сколько запросов система выдерживает
- `peak_rss_mb` — пиковое потребление памяти процессом
- `cpu_percent_one_core` или `cpu_time_sec` — нагрузка на CPU

## Почему именно такие метрики

Это сервис инференса эмбеддингов на CPU, поэтому важны:

1. задержка одного запроса;
2. способность держать поток запросов;
3. цена по ресурсам.

Именно по этим трём осям удобно сравнивать baseline, ONNX и batching.

Скрипт `benchmark_service.py` сам пытается поднять uvicorn на указанном порту.
Поэтому перед запуском бенчмарка лучше либо:
- не держать сервис уже запущенным на этом порту,
- либо использовать результаты существующего процесса, если сервис уже поднят вручную.

Если порт уже занят, в логах может появиться `address already in use`, но при наличии уже работающего сервиса бенчмарк всё равно может успешно завершиться.