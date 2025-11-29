# s3_blobstore.py
import boto3
from typing import Tuple, Optional

class S3BlobStore:
    def __init__(self, bucket: str, aws_profile: Optional[str] = None, region: Optional[str] = None):
        # boto3 will pick credentials from env/iam/profile
        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        self.s3 = session.client("s3", region_name=region)
        self.bucket = bucket

    def put(self, key: str, data: bytes, content_type: str = "text/plain"):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def get(self, key: str) -> bytes:
        resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def url(self, key: str, expires_in: int = 3600) -> str:
        return self.s3.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_in)
