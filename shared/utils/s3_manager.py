import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import os
from typing import Optional, Union, Dict, Any
from io import BytesIO
import mimetypes

from .logging_utils import get_module_logger 

logger = get_module_logger("s3_manager") # 로깅 설정

# S3 관리자 클래스 (원본 이미지, 음성 S3 업로드)
class S3Manager:
    def __init__(self):
        """
        S3 클라이언트 초기화 (AWS 자격 증명 및 region 설정)
        
        보안상 자격 증명 정보는 로그에 기록하지 않습니다.
        """
        self.region_name = os.getenv("AWS_REGION", "ap-northeast-2") # AWS region 이름
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID") # AWS 액세스 키 ID
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY") # AWS 비밀 액세스 키
        
        self.s3_client = None
        self.s3_resource = None
        self._initialize_clients()
        
    
    def _initialize_clients(self) -> None:
        """S3 클라이언트 초기화 (내부 메서드)"""
        try:
            session_params = {
                'region_name': self.region_name
            }
            
            # 자격 증명이 제공된 경우에만 추가
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_params.update({
                    'aws_access_key_id': self.aws_access_key_id,
                    'aws_secret_access_key': self.aws_secret_access_key
                })
                logger.info("S3Manager initialized with explicit credentials")
            else:
                logger.info("S3Manager initialized with default credential chain")

            session = boto3.Session(**session_params)
            self.s3_client = session.client("s3")
            self.s3_resource = session.resource("s3")
            
            # 연결 테스트 (단순히 클라이언트 생성이 아닌 실제 연결 확인)
            try:
                self.s3_client.list_buckets()
                logger.info("S3 connection verified successfully")
            except Exception as e:
                logger.warning(f"S3 connection test failed: {type(e).__name__}")
                
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"AWS credentials error: {type(e).__name__}. Please configure credentials.")
            raise
        except Exception as e:
            logger.error(f"Error initializing S3 client: {type(e).__name__}: {str(e)}")
            raise
    
    def _get_content_type(self, file_path: str) -> str:
        """파일 확장자에 따른 Content type 결정"""
        content_type, _ = mimetypes.guess_type(file_path)
        return content_type or 'application/octet-stream'  # 기본값 설정

    def _validate_upload_params(self, bucket_name: str, object_key: str) -> bool:
        """업로드 파라미터 검증"""
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot upload file.")
            return False

        if not bucket_name or not isinstance(bucket_name, str):
            logger.error("Invalid bucket name provided.")
            return False
            
        if not object_key or not isinstance(object_key, str):
            logger.error("Invalid object key provided.")
            return False
            
        return True

    def upload_file(
        self,
        file_source: Union[str, BytesIO],
        bucket_name: str,
        object_key: str,
        content_type: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        파일 또는 BytesIO 객체를 S3 버킷에 업로드

        Args:
            file_source: 로컬 파일 경로 (str) 또는 파일류 객체 (BytesIO)
            bucket_name: S3 버킷 이름
            object_key: S3 객체 키 (예: 'path/to/your/file.jpg')
            content_type: 선택적. 파일의 콘텐츠 타입
            extra_args: 선택적. S3 업로드 함수에 전달할 추가 인수

        Returns:
            성공 시 S3 객체 URL (s3://bucket/key), 실패 시 None
        """
        # 파라미터 검증
        if not self._validate_upload_params(bucket_name, object_key):
            return None
        
        effective_extra_args = extra_args.copy() if extra_args else {}

        try:
            if isinstance(file_source, str):
                return self._upload_from_file_path(
                    file_source, bucket_name, object_key, content_type, effective_extra_args
                )
            elif isinstance(file_source, BytesIO):
                return self._upload_from_bytes_io(
                    file_source, bucket_name, object_key, content_type, effective_extra_args
                )
            else:
                logger.error(f"Invalid file_source type: {type(file_source)}. Must be str or BytesIO.")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload to {bucket_name}/{object_key}: {type(e).__name__}: {str(e)}")
            return None

    def _upload_from_file_path(
        self, 
        file_path: str, 
        bucket_name: str, 
        object_key: str, 
        content_type: Optional[str], 
        extra_args: Dict[str, Any]
    ) -> Optional[str]:
        """로컬 파일 경로에서 업로드"""
        if not os.path.exists(file_path):
            logger.error(f"Local file not found: {file_path}")
            return None
        
        if not os.path.isfile(file_path):
            logger.error(f"Path is not a file: {file_path}")
            return None

        final_content_type = content_type or self._get_content_type(file_path)
        if 'ContentType' not in extra_args:
            extra_args['ContentType'] = final_content_type

        try:
            with open(file_path, 'rb') as f:
                self.s3_client.upload_fileobj(f, bucket_name, object_key, ExtraArgs=extra_args)
            
            s3_object_url = f"s3://{bucket_name}/{object_key}"
            logger.info(f"Successfully uploaded file to {s3_object_url}")
            return s3_object_url
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"S3 ClientError during file upload: {error_code}")
            return None

    def _upload_from_bytes_io(
        self, 
        bytes_io: BytesIO, 
        bucket_name: str, 
        object_key: str, 
        content_type: Optional[str], 
        extra_args: Dict[str, Any]
    ) -> Optional[str]:
        """BytesIO 객체에서 업로드"""
        bytes_io.seek(0)  # 파일 포인터를 처음으로 이동
        
        final_content_type = content_type or 'application/octet-stream'
        if 'ContentType' not in extra_args:
            extra_args['ContentType'] = final_content_type

        try:
            self.s3_client.upload_fileobj(bytes_io, bucket_name, object_key, ExtraArgs=extra_args)
            
            s3_object_url = f"s3://{bucket_name}/{object_key}"
            logger.info(f"Successfully uploaded BytesIO to {s3_object_url}")
            return s3_object_url
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"S3 ClientError during BytesIO upload: {error_code}")
            return None

    def get_presigned_url(self, bucket_name: str, object_key: str, expiration: int = 3600) -> Optional[str]:
        """
        S3 객체에 접근하기 위한 presigned URL 생성

        Args:
            bucket_name: S3 버킷 이름
            object_key: S3 객체 키
            expiration: URL 유효 시간(초) (기본값: 1시간)

        Returns:
            성공 시 presigned URL, 실패 시 None
        """
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot generate presigned URL.")
            return None
            
        if not bucket_name or not object_key:
            logger.error("Bucket name and object key are required for presigned URL.")
            return None
            
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            logger.info(f"Generated presigned URL for {bucket_name}/{object_key}")
            return response
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"S3 ClientError generating presigned URL: {error_code}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL: {type(e).__name__}: {str(e)}")
            return None

    def check_bucket_exists(self, bucket_name: str) -> bool:
        """버킷 존재 여부 확인"""
        if not self.s3_client:
            return False
            
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                logger.warning(f"Bucket {bucket_name} does not exist")
            else:
                logger.error(f"Error checking bucket {bucket_name}: {error_code}")
            return False
        except Exception:
            return False

    def is_healthy(self) -> bool:
        """S3 연결 상태 확인"""
        if not self.s3_client:
            return False
            
        try:
            self.s3_client.list_buckets()
            return True
        except Exception:
            return False

# Example Usage (for direct testing of this module):
if __name__ == '__main__':
    # Ensure AWS credentials and region are configured in your environment,
    # or pass them directly to S3Manager for testing.
    # For this example, we rely on environment configuration.
    
    s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME") # Set this env var for testing
    if not s3_bucket_name:
        print("AWS_S3_BUCKET_NAME environment variable not set. Skipping S3Manager tests.")
    else:
        print(f"Testing S3Manager with bucket: {s3_bucket_name}")
        manager = S3Manager()

        if not manager.s3_client:
            print("S3 client failed to initialize. Check credentials and AWS configuration.")
        else:
            # Test 1: Upload a dummy text file
            dummy_file_name = "s3_test_upload.txt"
            dummy_object_key = f"tests/{dummy_file_name}"
            try:
                with open(dummy_file_name, "w") as f:
                    f.write("Hello from S3Manager test!\n")
                
                print(f"Attempting to upload {dummy_file_name} to {s3_bucket_name}/{dummy_object_key}...")
                s3_url = manager.upload_file(dummy_file_name, s3_bucket_name, dummy_object_key)
                if s3_url:
                    print(f"Successfully uploaded. S3 URL: {s3_url}")
                    
                    # Test 2: Get a presigned URL for the uploaded file
                    print(f"Attempting to get presigned URL for {dummy_object_key}...")
                    presigned_url = manager.get_presigned_url(s3_bucket_name, dummy_object_key)
                    if presigned_url:
                        print(f"Successfully generated presigned URL: {presigned_url}")
                        print(f"You can try opening this URL in a browser (valid for 1 hour).")
                    else:
                        print("Failed to generate presigned URL.")
                else:
                    print(f"Failed to upload {dummy_file_name}.")
            except Exception as e:
                print(f"Error during S3Manager test: {e}")
            finally:
                if os.path.exists(dummy_file_name):
                    os.remove(dummy_file_name)
            
            # Test 3: Upload BytesIO object
            bytes_content = BytesIO(b"This is some binary data from BytesIO.")
            bytes_object_key = "tests/s3_test_bytes_upload.bin"
            print(f"Attempting to upload BytesIO object to {s3_bucket_name}/{bytes_object_key}...")
            s3_bytes_url = manager.upload_file(bytes_content, s3_bucket_name, bytes_object_key, content_type="application/octet-stream")
            if s3_bytes_url:
                print(f"Successfully uploaded BytesIO. S3 URL: {s3_bytes_url}")
            else:
                print("Failed to upload BytesIO object.")

            print("S3Manager tests complete.") 