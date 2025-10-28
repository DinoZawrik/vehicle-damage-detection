"""
MinIO storage integration for image and result storage.
"""

from minio import Minio
from minio.error import S3Error
import os
from io import BytesIO
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StorageClient:
    """
    Client for interacting with MinIO object storage.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: bool = False
    ):
        """
        Initialize MinIO client.

        Args:
            endpoint: MinIO server endpoint
            access_key: Access key for authentication
            secret_key: Secret key for authentication
            secure: Whether to use HTTPS
        """
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.secure = secure

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

        # Bucket names
        self.images_bucket = "vehicle-images"
        self.results_bucket = "analysis-results"

        self._ensure_buckets()

    def _ensure_buckets(self):
        """Create buckets if they don't exist."""
        for bucket in [self.images_bucket, self.results_bucket]:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
            except S3Error as e:
                logger.error(f"Error creating bucket {bucket}: {e}")

    def upload_image(
        self,
        file_data: bytes,
        filename: str,
        content_type: str = "image/jpeg"
    ) -> str:
        """
        Upload an image to MinIO.

        Args:
            file_data: Image data as bytes
            filename: Name for the stored file
            content_type: MIME type of the image

        Returns:
            URL to the uploaded file
        """
        try:
            data_stream = BytesIO(file_data)
            self.client.put_object(
                self.images_bucket,
                filename,
                data_stream,
                length=len(file_data),
                content_type=content_type
            )

            url = self._get_object_url(self.images_bucket, filename)
            logger.info(f"Uploaded image: {filename}")
            return url

        except S3Error as e:
            logger.error(f"Error uploading image {filename}: {e}")
            raise

    def upload_result(
        self,
        file_data: bytes,
        filename: str,
        content_type: str = "image/jpeg"
    ) -> str:
        """
        Upload an analysis result (visualized image) to MinIO.

        Args:
            file_data: Result data as bytes
            filename: Name for the stored file
            content_type: MIME type

        Returns:
            URL to the uploaded file
        """
        try:
            data_stream = BytesIO(file_data)
            self.client.put_object(
                self.results_bucket,
                filename,
                data_stream,
                length=len(file_data),
                content_type=content_type
            )

            url = self._get_object_url(self.results_bucket, filename)
            logger.info(f"Uploaded result: {filename}")
            return url

        except S3Error as e:
            logger.error(f"Error uploading result {filename}: {e}")
            raise

    def get_object(self, bucket: str, filename: str) -> bytes:
        """
        Download an object from MinIO.

        Args:
            bucket: Bucket name
            filename: Object name

        Returns:
            Object data as bytes
        """
        try:
            response = self.client.get_object(bucket, filename)
            data = response.read()
            response.close()
            response.release_conn()
            return data

        except S3Error as e:
            logger.error(f"Error downloading {filename} from {bucket}: {e}")
            raise

    def delete_object(self, bucket: str, filename: str):
        """
        Delete an object from MinIO.

        Args:
            bucket: Bucket name
            filename: Object name
        """
        try:
            self.client.remove_object(bucket, filename)
            logger.info(f"Deleted {filename} from {bucket}")

        except S3Error as e:
            logger.error(f"Error deleting {filename} from {bucket}: {e}")
            raise

    def _get_object_url(self, bucket: str, filename: str) -> str:
        """
        Generate URL for an object.

        Args:
            bucket: Bucket name
            filename: Object name

        Returns:
            HTTP URL to the object
        """
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.endpoint}/{bucket}/{filename}"

    def list_objects(self, bucket: str, prefix: str = "") -> list:
        """
        List objects in a bucket.

        Args:
            bucket: Bucket name
            prefix: Filter by prefix

        Returns:
            List of object names
        """
        try:
            objects = self.client.list_objects(bucket, prefix=prefix)
            return [obj.object_name for obj in objects]

        except S3Error as e:
            logger.error(f"Error listing objects in {bucket}: {e}")
            return []
