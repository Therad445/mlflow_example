import boto3
from botocore.client import Config
from app.core.config import settings


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        use_ssl=bool(settings.s3_secure),
        config=Config(s3={"addressing_style": "path"}),
    )


def ensure_bucket():
    client = s3_client()
    buckets = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if settings.s3_bucket not in buckets:
        client.create_bucket(Bucket=settings.s3_bucket)


def presign_put(object_name: str, content_type: str | None = None, expires_sec: int = 3600) -> tuple[str, dict]:
    ensure_bucket()
    client = s3_client()
    params = {"Bucket": settings.s3_bucket, "Key": object_name}
    if content_type:
        params["ContentType"] = content_type
    url = client.generate_presigned_url(
        "put_object",
        Params=params,
        ExpiresIn=expires_sec,
    )
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    return url, headers


def presign_get(object_name: str, expires_sec: int = 3600) -> str:
    ensure_bucket()
    client = s3_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": object_name},
        ExpiresIn=expires_sec,
    )
    return url


def build_s3_uri(object_name: str) -> str:
    return f"s3://{settings.s3_bucket}/{object_name}"