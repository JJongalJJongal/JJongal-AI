import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import os
from typing import Optional, Union, Dict, Any
from io import BytesIO
import mimetypes

from ...shared.utils.logging import get_module_logger 

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

    def upload_temp_files_to_s3(
        self,
        temp_dir: str,
        bucket_name: str,
        story_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        temp 폴더의 파일들을 S3에 업로드 (타입별 폴더 분류)
        
        Args:
            temp_dir: temp 폴더 경로
            bucket_name: S3 버킷 이름
            story_id: 특정 스토리 ID의 파일만 업로드 (선택적)
        
        Returns:
            업로드 결과 딕셔너리
        """
        if not self._validate_upload_params(bucket_name, "temp"):
            return {"success": False, "error": "Invalid parameters"}
        
        if not os.path.exists(temp_dir):
            logger.error(f"Temp directory not found: {temp_dir}")
            return {"success": False, "error": "Temp directory not found"}
        
        import glob
        
        # 파일 스캔
        pattern = os.path.join(temp_dir, "**", "*")
        all_files = glob.glob(pattern, recursive=True)
        
        results = {
            "success": True,
            "uploaded_files": [],
            "failed_files": [],
            "stats": {"images": 0, "audio": 0, "other": 0}
        }
        
        for file_path in all_files:
            try:
                # 디렉토리는 건너뛰기
                if os.path.isdir(file_path):
                    continue
                
                # 숨김 파일 건너뛰기
                if os.path.basename(file_path).startswith('.'):
                    continue
                
                # 특정 스토리 ID만 처리하는 경우
                if story_id and story_id not in os.path.basename(file_path):
                    continue
                
                # 파일 타입 결정
                file_ext = os.path.splitext(file_path)[1].lower()
                filename = os.path.basename(file_path)
                
                if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    s3_folder = "images"
                    file_type = "image"
                elif file_ext in ['.mp3', '.wav', '.m4a', '.ogg']:
                    s3_folder = "audio"
                    file_type = "audio"
                else:
                    s3_folder = "other"
                    file_type = "other"
                
                # S3 객체 키 생성
                object_key = f"{s3_folder}/{filename}"
                
                # S3에 업로드
                s3_url = self.upload_file(
                    file_source=file_path,
                    bucket_name=bucket_name,
                    object_key=object_key
                )
                
                if s3_url:
                    results["uploaded_files"].append({
                        "local_path": file_path,
                        "s3_url": s3_url,
                        "type": file_type,
                        "size": os.path.getsize(file_path)
                    })
                    results["stats"][file_type] += 1
                    logger.info(f"Uploaded {filename} to {s3_url}")
                else:
                    results["failed_files"].append({
                        "local_path": file_path,
                        "error": "Upload failed"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                results["failed_files"].append({
                    "local_path": file_path,
                    "error": str(e)
                })
        
        logger.info(f"Batch upload completed. Uploaded: {len(results['uploaded_files'])}, Failed: {len(results['failed_files'])}")
        return results

    def list_s3_files(
        self,
        bucket_name: str,
        prefix: str = "",
        max_keys: int = 1000
    ) -> Dict[str, Any]:
        """
        S3 버킷의 파일 목록 조회
        
        Args:
            bucket_name: S3 버킷 이름
            prefix: 검색할 접두사 (예: "images/", "audio/")
            max_keys: 최대 반환 파일 수
        
        Returns:
            파일 목록 딕셔너리
        """
        if not self.s3_client:
            return {"success": False, "error": "S3 client not initialized"}
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    file_info = {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "etag": obj["ETag"].strip('"')
                    }
                    
                    # 파일 타입 결정
                    file_ext = os.path.splitext(obj["Key"])[1].lower()
                    if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                        file_info["type"] = "image"
                    elif file_ext in ['.mp3', '.wav', '.m4a', '.ogg']:
                        file_info["type"] = "audio"
                    else:
                        file_info["type"] = "other"
                    
                    files.append(file_info)
            
            # 타입별 통계
            stats = {
                "images": len([f for f in files if f["type"] == "image"]),
                "audio": len([f for f in files if f["type"] == "audio"]),
                "other": len([f for f in files if f["type"] == "other"])
            }
            
            return {
                "success": True,
                "files": files,
                "count": len(files),
                "stats": stats,
                "is_truncated": response.get("IsTruncated", False)
            }
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"S3 ClientError listing files: {error_code}")
            return {"success": False, "error": f"S3 error: {error_code}"}
        except Exception as e:
            logger.error(f"Error listing S3 files: {e}")
            return {"success": False, "error": str(e)}

    def upload_story_files_to_s3(
        self,
        temp_dir: str,
        bucket_name: str,
        story_id: str
    ) -> Dict[str, Any]:
        """
        특정 스토리의 모든 파일을 S3에 업로드
        
        Args:
            temp_dir: temp 폴더 경로
            bucket_name: S3 버킷 이름
            story_id: 스토리 ID
        
        Returns:
            업로드 결과
        """
        logger.info(f"Starting S3 upload for story: {story_id}")
        
        result = self.upload_temp_files_to_s3(
            temp_dir=temp_dir,
            bucket_name=bucket_name,
            story_id=story_id
        )
        
        if result["success"]:
            # 스토리별 메타데이터 생성
            metadata = {
                "story_id": story_id,
                "upload_timestamp": __import__('datetime').datetime.now().isoformat(),
                "total_files": len(result["uploaded_files"]),
                "stats": result["stats"]
            }
            
            # 메타데이터도 S3에 업로드
            metadata_key = f"metadata/story_{story_id}_metadata.json"
            import json
            from io import BytesIO
            
            metadata_bytes = BytesIO(json.dumps(metadata, ensure_ascii=False, indent=2).encode('utf-8'))
            metadata_url = self.upload_file(
                file_source=metadata_bytes,
                bucket_name=bucket_name,
                object_key=metadata_key,
                content_type="application/json"
            )
            
            if metadata_url:
                result["metadata_url"] = metadata_url
                logger.info(f"Uploaded metadata for story {story_id}: {metadata_url}")
        
        return result

    def get_story_files_from_s3(
        self,
        bucket_name: str,
        story_id: str
    ) -> Dict[str, Any]:
        """
        S3에서 특정 스토리의 파일들 조회
        
        Args:
            bucket_name: S3 버킷 이름
            story_id: 스토리 ID
        
        Returns:
            스토리 파일 목록
        """
        try:
            # 이미지 파일들 조회
            images_result = self.list_s3_files(bucket_name, f"images/", 1000)
            audio_result = self.list_s3_files(bucket_name, f"audio/", 1000)
            
            if not images_result["success"] or not audio_result["success"]:
                return {"success": False, "error": "Failed to list S3 files"}
            
            # 스토리 ID가 포함된 파일들만 필터링
            story_images = [f for f in images_result["files"] if story_id in f["key"]]
            story_audio = [f for f in audio_result["files"] if story_id in f["key"]]
            
            # presigned URL 생성
            for file_info in story_images + story_audio:
                file_info["download_url"] = self.get_presigned_url(
                    bucket_name, file_info["key"], expiration=3600
                )
            
            return {
                "success": True,
                "story_id": story_id,
                "images": story_images,
                "audio": story_audio,
                "total_files": len(story_images) + len(story_audio),
                "stats": {
                    "images": len(story_images),
                    "audio": len(story_audio)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting story files from S3: {e}")
            return {"success": False, "error": str(e)}

    def sync_temp_to_s3(
        self,
        temp_dir: str,
        bucket_name: str,
        delete_after_upload: bool = False
    ) -> Dict[str, Any]:
        """
        temp 폴더의 모든 파일을 S3와 동기화
        
        Args:
            temp_dir: temp 폴더 경로
            bucket_name: S3 버킷 이름
            delete_after_upload: 업로드 후 로컬 파일 삭제 여부
        
        Returns:
            동기화 결과
        """
        logger.info(f"Starting temp to S3 sync. Delete after upload: {delete_after_upload}")
        
        # 모든 파일 업로드
        result = self.upload_temp_files_to_s3(temp_dir, bucket_name)
        
        if result["success"] and delete_after_upload:
            deleted_files = []
            delete_errors = []
            
            for file_info in result["uploaded_files"]:
                try:
                    local_path = file_info["local_path"]
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        deleted_files.append(local_path)
                        logger.info(f"Deleted local file: {local_path}")
                except Exception as e:
                    delete_errors.append({
                        "file": file_info["local_path"],
                        "error": str(e)
                    })
                    logger.error(f"Failed to delete {file_info['local_path']}: {e}")
            
            result["deleted_files"] = deleted_files
            result["delete_errors"] = delete_errors
            result["cleanup_completed"] = len(delete_errors) == 0
        
        return result

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