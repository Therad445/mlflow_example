# Model Registry (from scratch)

Небольшой сервис model registry: хранит **метаданные** моделей/версий в PostgreSQL и **артефакты** в S3‑совместимом хранилище (MinIO). Поддерживает версии, стадии (draft/staging/production/archived), аудит переходов и выдачу presigned URL для загрузки/скачивания артефактов.

---

## 1) Запуск

### 1.1. Подготовка окружения

В папке `registry/`:

```bash
cp .env.example .env
```

При необходимости поменяй порты/секреты в `.env`.

### 1.2. Поднять сервисы

```bash
docker compose up --build
```

После старта:

* API: `http://localhost:8009`
* Swagger UI: `http://localhost:8009/docs`
* MinIO Console: `http://localhost:9010` (логин/пароль: `minioadmin/minioadmin`)

Проверка здоровья:

```bash
curl -s http://localhost:8009/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

---

## 2) Быстрый сценарий (curl)

Ниже — демонстрация полного флоу:

1. создать модель
2. получить presigned URL
3. загрузить артефакт в MinIO
4. зафиксировать артефакт в БД (commit)
5. создать версию модели
6. промоутнуть draft → staging → production
7. получить текущую production‑версию

> Для удобства рекомендуется установить `jq`.

### 2.1. Создать модель

```bash
curl -s -X POST "http://localhost:8009/v1/models" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"fraud_detector",
    "description":"baseline",
    "owner":"mlds_180",
    "tags":["fraud","tabular"]
  }' | jq
```

Сохрани `id` из ответа в переменную:

```bash
MODEL_ID="<PASTE_MODEL_ID>"
```

### 2.2. Получить presigned URL для загрузки

Задаём путь объекта в бакете:

```bash
MODEL_OBJ="models/fraud_detector/v1/model.bin"

curl -s -X POST "http://localhost:8009/v1/artifacts/presign-upload" \
  -H "Content-Type: application/json" \
  -d "{\"object_name\":\"$MODEL_OBJ\",\"content_type\":\"application/octet-stream\"}" | jq
```

Из ответа возьми `url` и положи в переменную:

```bash
PUT_URL="<PASTE_URL_FROM_RESPONSE>"
```

### 2.3. Загрузить артефакт (PUT)

Создадим тестовый файл:

```bash
echo "dummy-model-bytes" > /tmp/model.bin
```

Зальём его:

```bash
curl -s -X PUT "$PUT_URL" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@/tmp/model.bin"
```

### 2.4. Commit артефакта (запись в БД)

Посчитаем `sha256` и размер:

```bash
SHA=$(sha256sum /tmp/model.bin | awk '{print $1}')
SIZE=$(stat -c%s /tmp/model.bin)
```

Зафиксируем:

```bash
curl -s -X POST "http://localhost:8009/v1/artifacts/commit" \
  -H "Content-Type: application/json" \
  -d "{\"object_name\":\"$MODEL_OBJ\",\"sha256\":\"$SHA\",\"size_bytes\":$SIZE}" | jq
```

Сохрани `id` артефакта:

```bash
ARTIFACT_ID="<PASTE_ARTIFACT_ID>"
```

### 2.5. Создать версию модели

```bash
curl -s -X POST "http://localhost:8009/v1/models/$MODEL_ID/versions" \
  -H "Content-Type: application/json" \
  -d "{
    \"metrics\": {\"auc\": 0.91},
    \"params\": {\"lr\": 0.01, \"epochs\": 3},
    \"env\": {\"python\": \"3.12\"},
    \"dataset_ref\": \"s3://datasets/fraud/2026-03-01/split_v1\",
    \"code_ref\": \"git:abcdef1234\",
    \"artifact_id\": \"$ARTIFACT_ID\"
  }" | jq
```

Версия создастся как `draft` и получит `version=1` (или следующий номер).

### 2.6. Promote: draft → staging → production

Сначала в `staging`:

```bash
curl -s -X POST "http://localhost:8009/v1/models/$MODEL_ID/versions/1/promote" \
  -H "Content-Type: application/json" \
  -d '{"stage":"staging","comment":"offline tests ok"}' | jq
```

Потом в `production`:

```bash
curl -s -X POST "http://localhost:8009/v1/models/$MODEL_ID/versions/1/promote" \
  -H "Content-Type: application/json" \
  -d '{"stage":"production","comment":"approved"}' | jq
```

### 2.7. Получить текущую production‑версию

```bash
curl -s "http://localhost:8009/v1/models/$MODEL_ID/production" | jq
```

---

## 3) Интеграция с MLflow пайплайном (идея)

В проекте уже используется MLflow для трекинга экспериментов (params/metrics/artifacts/runs).

**Разделение ответственности:**

* **MLflow** — история экспериментов: как обучали, какие метрики, какие артефакты.
* **Registry** — “контракт для использования”: какая версия модели считается актуальной для staging/production, где лежит бинарник, кто промоутил.

Типовой шаг после `evaluate()`:

1. упаковать/сохранить артефакт модели (например `model.joblib`)
2. получить presigned URL: `POST /v1/artifacts/presign-upload`
3. залить файл в MinIO по URL
4. commit в БД: `POST /v1/artifacts/commit`
5. создать версию модели с метаданными из MLflow: `POST /v1/models/{id}/versions`
6. при соблюдении критериев качества — promote в `staging`, затем ручной approve в `production`

---

## 4) Полезные ссылки

* Swagger UI: `http://localhost:8009/docs`
* Healthcheck: `http://localhost:8009/health`
* MinIO Console: `http://localhost:9010`

---

## 5) Примечание

Это учебный проект. Для боевого варианта позже можно добавить:

* полноценную авторизацию (OAuth2/JWT + роли)
* валидации для promotion (например, нельзя в production без артефакта и без набора тестов)
* отдельный сервис/процесс импорта старых папок (`models/mlds_*/...`) в registry
* мониторинг (метрики/логи/трейсы)
