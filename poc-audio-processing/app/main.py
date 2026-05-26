import os
import uuid
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.transcribe import process_pipeline, search_knowledge_base, delete_source_from_qdrant, generate_quiz

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio/Video Ingestion & Vector Store Service (Sprint 2 & 3)",
    description="Service to extract audio, transcribe, run diarization, semantically chunk, filter, and index vectors in Qdrant.",
    version="2.0.0"
)

# Configuration & Security constants
ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a"}
ALLOWED_MODEL_SIZES = {"tiny", "base", "small"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit
UPLOAD_DIR = "/tmp/poc_uploads"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChunkSegment(BaseModel):
    chunk_index: int
    ts_start: float
    ts_end: float
    speaker: str
    content: str
    category: str
    confidence_score: float
    topic_summary: str
    qdrant_vector_id: str


class PipelineResponse(BaseModel):
    text: str
    chunks: List[ChunkSegment]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    course_id: Optional[str] = None
    limit: Optional[int] = Field(5, ge=1, le=20)


class SearchHitPayload(BaseModel):
    source_id: str
    course_id: str
    content: str
    topic_summary: str
    speaker: str
    confidence_score: float
    ts_start: float
    ts_end: float
    category: str


class SearchHit(BaseModel):
    id: str
    score: float
    payload: SearchHitPayload


class SearchResponse(BaseModel):
    hits: List[SearchHit]


class QuizGenerateRequest(BaseModel):
    course_id: str
    num_questions: Optional[int] = Field(5, ge=1, le=20)
    bloom_level: Optional[str] = Field("comprender", pattern="^(recordar|comprender|aplicar|analizar)$")


class QuizQuestionOptions(BaseModel):
    a: str
    b: str
    c: str
    d: str


class QuizQuestion(BaseModel):
    question: str
    options: QuizQuestionOptions
    correct_option: str = Field(..., pattern="^[a-d]$")
    bloom_level: str
    justification: str
    chunk_id: str
    validation_score: float
    validation_reason: str


class QuizGenerateResponse(BaseModel):
    questions: List[QuizQuestion]



def validate_uuid(uuid_to_test: str) -> bool:
    """
    Validates that a string is a valid UUID.
    """
    try:
        uuid.UUID(str(uuid_to_test))
        return True
    except ValueError:
        return False


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/pipeline/transcribe", response_model=PipelineResponse)
async def transcribe(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    source_id: Optional[str] = Form(None),
    course_id: Optional[str] = Form(None)
):
    """
    Ingests audio/video, runs full pipeline, and indexes semantic chunks in Qdrant.
    """
    logger.info(f"Received pipeline request for file: {file.filename}")
    
    # 1. Input Validation: Validate model_size
    if model_size not in ALLOWED_MODEL_SIZES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model size. Choose from: {list(ALLOWED_MODEL_SIZES)}"
        )
        
    # 2. Input Validation: Validate UUIDs if provided, otherwise generate secure random ones
    if source_id:
        if not validate_uuid(source_id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid source_id UUID format.")
    else:
        source_id = str(uuid.uuid4())
        
    if course_id:
        if not validate_uuid(course_id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid course_id UUID format.")
    else:
        course_id = str(uuid.uuid4())
        
    # 3. Input Validation: Validate file extension
    original_filename = file.filename
    _, ext = os.path.splitext(original_filename)
    ext = ext.lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"File upload rejected: invalid extension '{ext}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed formats: {list(ALLOWED_EXTENSIONS)}"
        )

    # 4. Path Security: Generate a safe, random UUID filename (prevents directory traversal)
    safe_filename = f"{uuid.uuid4()}{ext}"
    temp_file_path = os.path.abspath(os.path.join(UPLOAD_DIR, safe_filename))
    
    # Ensure it's locked within the UPLOAD_DIR directory boundary
    resolved_upload_dir = os.path.abspath(UPLOAD_DIR)
    if not temp_file_path.startswith(resolved_upload_dir + os.path.sep):
        logger.error(f"Path traversal attempt detected. Path resolved to: {temp_file_path}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename path boundary."
        )

    # 5. Input Validation: Enforce maximum file size streamingly to prevent memory exhaustion
    size = 0
    try:
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_FILE_SIZE_BYTES:
                    buffer.close()
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    logger.warning(f"File upload rejected: exceeds max size of {MAX_FILE_SIZE_BYTES} bytes")
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES / (1024*1024):.1f} MB."
                    )
                buffer.write(chunk)
    except Exception as ex:
        if isinstance(ex, HTTPException):
            raise ex
        logger.error(f"Failed to write file to disk: {ex}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error writing uploaded file to disk."
        )

    # 6. Core Execution: Process pipeline and index
    try:
        hf_token = os.environ.get("HF_TOKEN")
        result = process_pipeline(
            input_path=temp_file_path,
            source_id=source_id,
            course_id=course_id,
            model_size=model_size,
            hf_token=hf_token
        )
        return JSONResponse(content=result)
        
    except Exception as ex:
        logger.exception("Error during pipeline processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {str(ex)}"
        )
        
    finally:
        # 7. Secure Cleanup: Ensure the temporary uploaded file is deleted
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary uploaded file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")


@app.post("/api/kb/search", response_model=SearchResponse)
def search_kb(request: SearchRequest):
    """
    Searches the knowledge base collection in Qdrant using vector embeddings.
    """
    logger.info(f"Received search request: '{request.query}' limit={request.limit}")
    
    # Input Validation: Validate course_id UUID if supplied
    if request.course_id:
        if not validate_uuid(request.course_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid course_id UUID format."
            )
            
    try:
        hits = search_knowledge_base(
            query=request.query,
            course_id=request.course_id,
            limit=request.limit
        )
        return {"hits": hits}
    except Exception as ex:
        logger.error(f"Failed to search Qdrant knowledge base: {ex}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(ex)}"
        )


@app.delete("/api/kb/source/{source_id}")
def delete_source(source_id: str):
    """
    Removes all chunks and vector points associated with the given source_id.
    """
    logger.info(f"Received request to delete source: {source_id}")
    
    # Input Validation: Validate source_id UUID
    if not validate_uuid(source_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid source_id UUID format."
        )
        
    try:
        delete_source_from_qdrant(source_id)
        return {"status": "deleted", "source_id": source_id}
    except Exception as ex:
        logger.error(f"Failed to delete source from Qdrant: {ex}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {str(ex)}"
        )


@app.post("/api/quizzes/generate", response_model=QuizGenerateResponse)
def generate_quiz_endpoint(request: QuizGenerateRequest):
    """
    RAG-based quiz generation endpoint. Retrieves chunks, generates questions via Ollama,
    validates grounding via LLM-as-a-judge, filters duplicates, and returns questions.
    """
    logger.info(f"Received quiz generation request for course: {request.course_id}")
    
    # Input Validation: Validate course_id UUID
    if not validate_uuid(request.course_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid course_id UUID format."
        )
        
    try:
        questions = generate_quiz(
            course_id=request.course_id,
            num_questions=request.num_questions,
            bloom_level=request.bloom_level
        )
        return {"questions": questions}
    except Exception as ex:
        logger.error(f"Failed to generate quiz: {ex}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz generation failed: {str(ex)}"
        )

