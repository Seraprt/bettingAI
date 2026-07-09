import boto3
import os
import logging
from botocore.exceptions import ClientError, NoCredentialsError
from .config import Config

# Set up logging (it will inherit the root logger)
logger = logging.getLogger(__name__)

def get_s3_client():
    """
    Return an S3 client configured for Blackbase or AWS.
    Uses environment variables from Config.
    """
    return boto3.client(
        's3',
        endpoint_url=Config.AWS_S3_ENDPOINT_URL or None,  # None means use default AWS endpoint
        region_name=Config.AWS_REGION or 'us-east-1',
        aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
        config=boto3.session.Config(signature_version='s3v4')
    )

def check_cloud_connection():
    """
    Verify that cloud storage credentials work and the bucket is accessible.
    Returns (success, message).
    """
    # Check if all required configs are present
    if not all([Config.AWS_ACCESS_KEY_ID, Config.AWS_SECRET_ACCESS_KEY, Config.AWS_S3_BUCKET]):
        return False, "❌ Cloud storage not configured: missing credentials or bucket name. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_S3_BUCKET."

    try:
        s3 = get_s3_client()
        # Try to list objects (just the first page) to verify access
        s3.list_objects_v2(Bucket=Config.AWS_S3_BUCKET, MaxKeys=1)
        return True, f"✅ Cloud storage ready: bucket '{Config.AWS_S3_BUCKET}' is accessible."
    except NoCredentialsError:
        return False, "❌ AWS credentials not found. Check your .env variables."
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            return False, f"❌ Bucket '{Config.AWS_S3_BUCKET}' does not exist or you don't have access."
        elif error_code == 'AccessDenied':
            return False, "❌ Access denied to bucket. Check your permissions."
        else:
            return False, f"❌ Cloud storage error: {e}"
    except Exception as e:
        return False, f"❌ Unexpected error checking cloud connection: {e}"

def upload_file(local_path, s3_key):
    """
    Upload a file to S3 bucket.
    Returns True on success, False on failure.
    """
    if not Config.AWS_S3_BUCKET:
        logger.warning("❌ AWS_S3_BUCKET not set – skipping upload.")
        return False
    try:
        s3 = get_s3_client()
        s3.upload_file(local_path, Config.AWS_S3_BUCKET, s3_key)
        logger.info(f"✅ Uploaded {local_path} to s3://{Config.AWS_S3_BUCKET}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"❌ Upload failed for {local_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during upload: {e}")
        return False

def download_file(s3_key, local_path):
    """
    Download a file from S3 bucket.
    Returns True on success, False on failure.
    """
    if not Config.AWS_S3_BUCKET:
        logger.warning("❌ AWS_S3_BUCKET not set – skipping download.")
        return False
    try:
        # --- FIX: create the parent directory if it doesn't exist ---
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3 = get_s3_client()
        s3.download_file(Config.AWS_S3_BUCKET, s3_key, local_path)
        logger.info(f"✅ Downloaded s3://{Config.AWS_S3_BUCKET}/{s3_key} to {local_path}")
        return True
    except ClientError as e:
        logger.error(f"❌ Download failed for {s3_key}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during download: {e}")
        return False

def list_models():
    """
    List available model files in the bucket under the 'models/' prefix.
    Returns a list of S3 keys (strings) or an empty list on failure.
    """
    if not Config.AWS_S3_BUCKET:
        return []
    try:
        s3 = get_s3_client()
        response = s3.list_objects_v2(Bucket=Config.AWS_S3_BUCKET, Prefix='models/')
        if 'Contents' not in response:
            return []
        return [obj['Key'] for obj in response['Contents']]
    except ClientError as e:
        logger.error(f"❌ Failed to list models: {e}")
        return []