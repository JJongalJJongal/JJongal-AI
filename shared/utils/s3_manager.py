import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import os
from typing import Optional, Union
from io import BytesIO
import mimetypes

from .logging_utils import get_module_logger 

logger = get_module_logger("s3_manager") # 로깅 설정

# S3 관리자 클래스 (원본 이미지, 음성 S3 업로드)
class S3Manager:
    def __init__(self):
        """
        S3 클라이언트 초기화 (AWS 자격 증명 및 region 설정)
        
        Args:
            region_name: AWS region 이름 (기본값: 환경 변수에서 로드)
        """
        self.region_name = os.getenv("AWS_REGION") # AWS region 이름
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID") # AWS 액세스 키 ID
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY") # AWS 비밀 액세스 키
        
        self.s3_client = None # S3 Client
        self.s3_resource = None # S3 Resource
        try:
            session_params = {} # Session 파라미터
            if self.region_name: # region 이름이 있으면 설정
                session_params['region_name'] = self.region_name # region 이름 설정
            if self.aws_access_key_id and self.aws_secret_access_key: # 액세스 키 id와 비밀 액세스 키가 있으면 설정
                session_params['aws_access_key_id'] = self.aws_access_key_id # 액세스 key id 설정
                session_params['aws_secret_access_key'] = self.aws_secret_access_key # 비밀 액세스 키 설정

            session = boto3.Session(**session_params) # Session 생성
            self.s3_client = session.client("s3") # S3 Client 생성
            self.s3_resource = session.resource("s3") # S3 Resource 생성
            logger.info(f"S3Manager initialized. Region: {session.region_name or 'default'}") # 로깅
        except (NoCredentialsError, PartialCredentialsError): # 자격 증명 오류 처리
            logger.error("AWS credentials not found. Please configure credentials.") # 로깅
        except Exception as e: # 예외 처리
            logger.error(f"Error initializing S3 client: {e}", exc_info=True) # 로깅
            raise # 예외 발생
    
    # 파일 확장자에 따른 Content type 결정
    def _get_content_type(self, file_path: str) -> str:
        """Determines the content type of a file based on its extension.""" 
        content_type, _ = mimetypes.guess_type(file_path) # 파일 확장자에 따른 Content type 결정
        return content_type 

    # 파일 업로드
    def upload_file(
        self,
        file_source: Union[str, BytesIO], # 파일 소스 (로컬 파일 경로 또는 BytesIO 객체)
        bucket_name: str, # S3 버킷 이름
        object_key: str, # S3 객체 키 (버킷 내 경로)
        content_type: Optional[str] = None, # 콘텐츠 타입 (기본값: None)
        extra_args: Optional[dict] = None # 추가 인수 (기본값: None)
    ) -> Optional[str]:
        """
        Uploads a file or BytesIO object to an S3 bucket.

        Args:
            file_source: Path to the local file (str) or a file-like object (BytesIO).
            bucket_name: S3 버킷 이름.
            object_key: S3 객체 키 (예: 'path/to/your/file.jpg').
            content_type: 선택적. 파일의 콘텐츠 타입. 파일 소스가 경로인 경우 None이면 추측됩니다.
                          it will be guessed. For BytesIO, it defaults to application/octet-stream if not set.
            extra_args: 선택적. S3 업로드 함수에 전달할 추가 인수 (예: {'ACL': 'public-read'} 또는 Metadata).
                        (e.g., {'ACL': 'public-read'} or Metadata).

        Returns:
            S3 객체 URL (s3://bucket/key) 성공 시, None 반환
        """
        if not self.s3_client: # S3 Client가 초기화되지 않은 경우
            logger.error("S3 client not initialized. Cannot upload file.") # 로깅
            return None # 아무것도 반환하지 않음.

        if not bucket_name: # 버킷 이름이 없는 경우
            logger.error("Bucket name not provided. Cannot upload file.") # 로깅
            return None # 아무것도 반환하지 않음.
        
        effective_extra_args = extra_args or {} # 추가 인수 설정
        final_content_type = content_type # 콘텐츠 타입 설정

        try:
            if isinstance(file_source, str): # 로컬 파일 경로
                if not os.path.exists(file_source): # 파일이 존재하지 않는 경우
                    logger.error(f"Local file not found: {file_source}") # 로깅
                    return None # 아무것도 반환하지 않음.
                
                final_content_type = content_type or self._get_content_type(file_source) # Content 타입 결정
                if 'ContentType' not in effective_extra_args: # ContentType이 없는 경우
                    effective_extra_args['ContentType'] = final_content_type # ContentType 설정

                with open(file_source, 'rb') as f: # 파일 열기
                    self.s3_client.upload_fileobj(f, bucket_name, object_key, ExtraArgs=effective_extra_args) # 파일 업로드
            
            elif isinstance(file_source, BytesIO): # In-memory BytesIO object
                file_source.seek(0) # 파일 포인터를 처음으로 이동
                final_content_type = content_type or 'application/octet-stream' # Content 타입 설정
                if 'ContentType' not in effective_extra_args: # ContentType이 없는 경우
                    effective_extra_args['ContentType'] = final_content_type # ContentType 설정
                
                self.s3_client.upload_fileobj(file_source, bucket_name, object_key, ExtraArgs=effective_extra_args) # 파일 업로드
            else: # 파일 소스 타입이 유효하지 않은 경우 
                logger.error(f"Invalid file_source type: {type(file_source)}. Must be str or BytesIO.") # 로깅
                return None # 아무것도 반환하지 않음.

            s3_object_url = f"s3://{bucket_name}/{object_key}" # S3 객체 URL 생성
            logger.info(f"Successfully uploaded to {s3_object_url} with ContentType: {effective_extra_args.get('ContentType')}")
            return s3_object_url
        
        except FileNotFoundError: # Should be caught by os.path.exists for str paths
            logger.error(f"File not found: {file_source}")
            return None
        except NoCredentialsError:
            logger.error("AWS credentials not found during upload.")
            return None
        except PartialCredentialsError:
            logger.error("Incomplete AWS credentials found during upload.")
            return None
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            logger.error(f"S3 ClientError during upload to {bucket_name}/{object_key}: {error_code} - {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during S3 upload to {bucket_name}/{object_key}: {e}", exc_info=True)
            return None

    def get_presigned_url(self, bucket_name: str, object_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generates a presigned URL to access an S3 object.

        Args:
            bucket_name: Name of the S3 bucket.
            object_key: The S3 object key.
            expiration: Time in seconds for the presigned URL to remain valid (default is 1 hour).

        Returns:
            The presigned URL if successful, None otherwise.
        """
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot generate presigned URL.")
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
            logger.error(f"Error generating presigned URL for {bucket_name}/{object_key}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred generating presigned URL for {bucket_name}/{object_key}: {e}", exc_info=True)
            return None

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