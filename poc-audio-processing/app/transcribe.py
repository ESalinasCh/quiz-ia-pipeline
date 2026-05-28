import os
import json
import uuid
import subprocess
import logging
import tempfile
import time
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import httpx

# Single source of truth for model names + Qdrant collection.
# Suffix "_python" distinguishes vectors stored by this service from the .NET alternative.
LLM_MODEL = "qwen2.5-coder:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "quiz_chunks_python"

TRANSCRIPTS_DIR = os.environ.get("TRANSCRIPTS_DIR", "/tmp/poc_transcripts")
QUIZZES_DIR = os.environ.get("QUIZZES_DIR", "/tmp/poc_quizzes")

# Attempt to load ML libraries inside the container
try:
    import librosa
    from sklearn.cluster import KMeans
except ImportError as e:
    logging.warning(f"Failed to import ML libraries for fallback: {e}")

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    logging.warning(f"Failed to import faster-whisper: {e}")

try:
    from pyannote.audio import Pipeline
except ImportError as e:
    logging.warning(f"Failed to import pyannote.audio: {e}")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError as e:
    logging.warning(f"Failed to import Qdrant client: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)}m {int(secs)}s ({seconds:.1f}s)"


class StageTimer:
    """Cumulative stage timer; logs each step's wall time and tracks totals."""
    def __init__(self, pipeline_name: str = "pipeline"):
        self.pipeline_name = pipeline_name
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.stages: List[Dict[str, Any]] = []

    def mark(self, stage: str):
        now = time.perf_counter()
        elapsed = now - self.last
        self.last = now
        self.stages.append({"stage": stage, "seconds": round(elapsed, 3)})
        logger.info(f"[TIMING] {stage} took {_format_elapsed(elapsed)}.")

    def total(self) -> float:
        return time.perf_counter() - self.t0


def save_transcript(input_path: str, model_size: str, aligned: List[Dict[str, Any]]) -> str:
    """Saves the aligned transcript (with speaker labels + timestamps) to disk."""
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(TRANSCRIPTS_DIR, f"{base}_{ts}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript: {base}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Model: faster-whisper-{model_size}\n")
        f.write(f"# Segments: {len(aligned)}\n\n")
        for seg in aligned:
            mm = int(seg["start"] // 60)
            ss = seg["start"] % 60
            mm_e = int(seg["end"] // 60)
            ss_e = seg["end"] % 60
            f.write(f"[{mm:02d}:{ss:06.3f} -> {mm_e:02d}:{ss_e:06.3f}] ({seg['speaker']}) {seg['text'].strip()}\n")

    logger.info(f"Transcript saved to {path}")
    return path


def save_quiz(course_id: str, bloom_level: str, questions: List[Dict[str, Any]], timings: List[Dict[str, Any]]) -> str:
    """Saves the generated quiz (with model metadata + timings) to disk."""
    os.makedirs(QUIZZES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(QUIZZES_DIR, f"quiz_{course_id}_{ts}.json")

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "course_id": course_id,
        "bloom_level": bloom_level,
        "qdrant_collection": COLLECTION_NAME,
        "models": {
            "transcript": "faster-whisper (set at ingest time)",
            "processing": LLM_MODEL,
            "embedding": EMBEDDING_MODEL,
            "llm_as_judge": LLM_MODEL,
        },
        "timings": timings,
        "questions": questions,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Quiz saved to {path}")
    return path


def extract_audio(input_path: str, output_path: str) -> str:
    """
    Extracts audio from video/audio input to 16kHz mono 16-bit PCM WAV.
    Uses list-based subprocess execution for command-injection security.
    """
    logger.info(f"Extracting audio from {input_path} to {output_path}...")
    
    # Secure command execution (list parameters, shell=False by default)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",                   # Disable video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",          # 16kHz sample rate
        "-ac", "1",              # Mono channel
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        logger.info("Audio extraction completed successfully.")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8", errors="ignore")
        logger.error(f"FFmpeg failed: {error_msg}")
        raise RuntimeError(f"FFmpeg audio extraction failed: {error_msg}")


def transcribe_audio(audio_path: str, model_size: str = "base") -> List[Dict[str, Any]]:
    """
    Transcribes audio using faster-whisper.
    """
    logger.info(f"Starting transcription with model '{model_size}' on CPU...")
    cache_dir = os.environ.get("WHISPER_CACHE_DIR", "/cache/whisper")
    
    # Initialize the model on CPU
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="float32",
        download_root=cache_dir
    )
    
    segments, info = model.transcribe(audio_path, beam_size=5)
    logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
    
    results = []
    for segment in segments:
        results.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })
        logger.debug(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
        
    logger.info(f"Transcription complete. Generated {len(results)} segments.")
    return results


def run_fallback_diarization(audio_path: str, n_speakers: int = 2) -> List[Dict[str, Any]]:
    """
    Local speaker diarization fallback using librosa and KMeans with simple VAD.
    Does not require external APIs or HuggingFace tokens.
    """
    logger.info("Running local fallback speaker diarization with VAD...")
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration < 4.0:
            logger.info("Audio is too short for clustering. Assigning single speaker.")
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "SPEAKER_0"}]
            
        segment_len = 1.0  # 1-second windows for higher resolution
        samples_per_seg = int(segment_len * sr)
        total_samples = len(y)
        
        # Calculate RMS energy for all segments to establish a VAD threshold
        all_rms = []
        for start_idx in range(0, total_samples, samples_per_seg):
            end_idx = min(start_idx + samples_per_seg, total_samples)
            if end_idx - start_idx < (sr // 2):
                continue
            chunk = y[start_idx:end_idx]
            rms = np.sqrt(np.mean(chunk**2))
            all_rms.append(rms)
            
        if not all_rms:
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "SPEAKER_0"}]
            
        max_rms = max(all_rms)
        # Threshold: 10% of maximum RMS energy or a minimum of 0.005
        vad_threshold = max(max_rms * 0.1, 0.005)
        
        features = []
        timestamps = []
        
        # Extract features for segments that pass VAD
        for start_idx in range(0, total_samples, samples_per_seg):
            end_idx = min(start_idx + samples_per_seg, total_samples)
            if end_idx - start_idx < (sr // 2):
                continue
            
            chunk = y[start_idx:end_idx]
            rms = np.sqrt(np.mean(chunk**2))
            
            if rms >= vad_threshold:
                # Extract MFCCs
                mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
                # Compute mean and std deviation to capture voice texture
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                feature_vector = np.concatenate([mfcc_mean, mfcc_std])
                
                features.append(feature_vector)
                timestamps.append((start_idx / sr, end_idx / sr))
                
        if len(features) < n_speakers:
            logger.info("Not enough speech segments for clustering. Assigning single speaker.")
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "SPEAKER_0"}]
            
        # Cluster MFCC features of active speech
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Construct speaker intervals
        intervals = []
        current_speaker = f"SPEAKER_{labels[0]}"
        start_time = timestamps[0][0]
        end_time = timestamps[0][1]
        
        for i in range(1, len(labels)):
            speaker = f"SPEAKER_{labels[i]}"
            is_consecutive = (timestamps[i][0] - timestamps[i-1][1]) < (segment_len + 0.1)
            
            if speaker == current_speaker and is_consecutive:
                end_time = timestamps[i][1]
            else:
                intervals.append({
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "speaker": current_speaker
                })
                current_speaker = speaker
                start_time = timestamps[i][0]
                end_time = timestamps[i][1]
                
        # Append final interval
        intervals.append({
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "speaker": current_speaker
        })
        
        logger.info(f"Fallback diarization complete with VAD. Intervals: {intervals}")
        return intervals
        
    except Exception as ex:
        logger.error(f"Fallback diarization failed: {ex}. Using mock speaker assignment.")
        return [{"start": 0.0, "end": 99999.0, "speaker": "SPEAKER_0"}]


def diarize_audio(audio_path: str, hf_token: str = None) -> List[Dict[str, Any]]:
    """
    Performs speaker diarization. Uses pyannote-audio if hf_token is provided,
    otherwise falls back to KMeans local clustering.
    """
    if hf_token:
        logger.info("Initializing neural diarization via pyannote-audio...")
        try:
            # Requires accepting user conditions for models on HuggingFace:
            # pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # Run pipeline
            diarization = pipeline(audio_path)
            
            intervals = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                intervals.append({
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": speaker
                })
                
            logger.info(f"Neural diarization complete. Detected {len(intervals)} segments.")
            if intervals:
                return intervals
            else:
                logger.warning("Neural diarization returned empty segments. Falling back.")
                
        except Exception as ex:
            logger.error(f"PyAnnote diarization failed: {ex}. Falling back to local clustering.")
            
    # Fallback to local KMeans clustering
    return run_fallback_diarization(audio_path)


def align_segments(transcripts: List[Dict[str, Any]], diarization: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aligns transcription text segments with speaker diarization time segments.
    Assigns each transcript segment to the speaker with the maximum overlap.
    """
    logger.info("Aligning transcripts with speaker segments...")
    aligned_results = []
    
    for seg in transcripts:
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        best_speaker = "SPEAKER_UNKNOWN"
        max_overlap = -1.0
        
        for dia in diarization:
            # Calculate overlap interval
            overlap_start = max(seg_start, dia["start"])
            overlap_end = min(seg_end, dia["end"])
            
            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = dia["speaker"]
                    
        # Fallback if no overlap is found
        if max_overlap <= 0:
            # Assign to the speaker closest to the start of this segment
            min_dist = float("inf")
            for dia in diarization:
                dist = min(abs(seg_start - dia["end"]), abs(seg_end - dia["start"]))
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = dia["speaker"]
                    
        aligned_results.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": best_speaker,
            "text": seg["text"]
        })
        
    return aligned_results


# ==========================================
# SPRINT 2: Semantic Chunking & Relevance AI
# ==========================================

def get_embedding(text: str) -> List[float]:
    """
    Generates embedding vector from host Ollama using nomic-embed-text.
    """
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{ollama_url}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text}
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embedding from Ollama: {e}")
        # Return a dummy vector of 768 dimensions if Ollama is unavailable
        return [0.0] * 768


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculates cosine similarity between two vectors.
    """
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def split_into_sentences(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups Whisper segments into sentences/phrases of at least 15 words or ending with punctuation.
    """
    sentences = []
    current_words = []
    start_time = None
    speakers = []
    
    for seg in segments:
        text = seg["text"].strip()
        if start_time is None:
            start_time = seg["start"]
        current_words.append(text)
        speakers.append(seg["speaker"])
        
        ends_with_punc = text and text[-1] in {".", "?", "!"}
        word_count = sum(len(w.split()) for w in current_words)
        
        if ends_with_punc or word_count >= 15:
            dominant_speaker = max(set(speakers), key=speakers.count) if speakers else "SPEAKER_UNKNOWN"
            sentences.append({
                "text": " ".join(current_words),
                "start": start_time,
                "end": seg["end"],
                "speaker": dominant_speaker
            })
            current_words = []
            start_time = None
            speakers = []
            
    if current_words:
        dominant_speaker = max(set(speakers), key=speakers.count) if speakers else "SPEAKER_UNKNOWN"
        sentences.append({
            "text": " ".join(current_words),
            "start": start_time if start_time is not None else segments[0]["start"],
            "end": segments[-1]["end"],
            "speaker": dominant_speaker
        })
        
    return sentences


def generate_semantic_chunks(segments: List[Dict[str, Any]], target_words: int = 200, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Computes sentence embeddings, groups them semantically using cosine similarity thresholds,
    and returns a list of semantic chunks.
    """
    logger.info("Starting semantic chunking...")
    sentences = split_into_sentences(segments)
    if not sentences:
        return []
        
    # Get embeddings for each sentence
    logger.info(f"Computing embeddings for {len(sentences)} sentences...")
    embeddings = [get_embedding(s["text"]) for s in sentences]
    
    # Compute cosine similarities between adjacent sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)
        
    chunks = []
    current_sentences = [sentences[0]]
    current_word_count = len(sentences[0]["text"].split())
    
    for i in range(len(similarities)):
        sim = similarities[i]
        next_sentence = sentences[i+1]
        next_word_count = len(next_sentence["text"].split())
        
        should_split = False
        if sim < similarity_threshold:
            if current_word_count >= 30: # Avoid splitting very small chunks
                should_split = True
        
        # Force split if current chunk gets too large (e.g. 350 words)
        if current_word_count + next_word_count > 350:
            should_split = True
            
        if should_split:
            chunks.append(current_sentences)
            current_sentences = [next_sentence]
            current_word_count = next_word_count
        else:
            current_sentences.append(next_sentence)
            current_word_count += next_word_count
            
    if current_sentences:
        chunks.append(current_sentences)
        
    # Format chunks with timestamps and speakers
    formatted_chunks = []
    for idx, chunk_sents in enumerate(chunks):
        content = " ".join([s["text"] for s in chunk_sents])
        speakers = [s["speaker"] for s in chunk_sents]
        dominant_speaker = max(set(speakers), key=speakers.count) if speakers else "SPEAKER_UNKNOWN"
        
        formatted_chunks.append({
            "chunk_index": idx,
            "ts_start": chunk_sents[0]["start"],
            "ts_end": chunk_sents[-1]["end"],
            "speaker": dominant_speaker,
            "content": content
        })
        
    logger.info(f"Generated {len(formatted_chunks)} semantic chunks.")
    return formatted_chunks


def classify_chunk(text: str, speaker: str) -> Dict[str, Any]:
    """
    Asks host Ollama (qwen3-coder) to classify the educational relevance of a chunk.
    Uses Structured JSON format option.
    """
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    prompt = f"""Eres un clasificador de contenido educativo universitario.
Clasifica el siguiente fragmento de clase grabada.

SPEAKER: {speaker}
TEXTO: "{text}"

Categorías:
- ACADEMICO: contenido del tema de la materia, explicaciones, definiciones, ejemplos académicos.
- OFF_TOPIC: conversación no relacionada al tema (clima, anécdotas personales, organización, saludos).
- ERROR_ESTUDIANTIL: respuesta incorrecta de un estudiante no corregida explícitamente.
- ANECDOTA: historia ilustrativa, puede ser útil como contexto pero no es conocimiento clave.

Responde únicamente con un objeto JSON válido con este formato:
{{"categoria": "ACADEMICO", "confidence": 0.92, "razon": "explicación sobre termodinámica"}}
"""
    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False
                }
            )
            resp.raise_for_status()
            res_json = json.loads(resp.json()["response"])
            category = res_json.get("categoria", "ACADEMICO").upper()
            confidence = float(res_json.get("confidence", 0.8))
            reason = res_json.get("razon", "")
            return {"category": category, "confidence": confidence, "reason": reason}
    except Exception as e:
        logger.error(f"Failed to classify chunk via Ollama: {e}")
        return {"category": "ACADEMICO", "confidence": 0.8, "reason": "default fallback"}


def generate_topic_summary(text: str) -> str:
    """
    Generates a 5-word topic summary using host Ollama.
    """
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    prompt = f"Proporciona un título o resumen del tema de máximo 5 palabras para el siguiente fragmento. Responde únicamente con el resumen sin comillas ni texto adicional:\n\n{text}"
    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )
            resp.raise_for_status()
            summary = resp.json()["response"].strip().replace('"', '')
            return summary
    except Exception as e:
        logger.error(f"Failed to generate topic summary: {e}")
        return "Resumen de clase"


# ==========================================
# SPRINT 3: Qdrant Vector Storage Helpers
# ==========================================

def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns QdrantClient pointing to Qdrant service.
    """
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    return QdrantClient(host=qdrant_host, port=6333)


def init_qdrant_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    """
    Ensures that the collection exists in Qdrant, creating it if necessary.
    """
    try:
        if not client.collection_exists(collection_name=collection_name):
            logger.info(f"Creating Qdrant collection: {collection_name}...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=768,  # nomic-embed-text vector dimensions
                    distance=qmodels.Distance.COSINE
                )
            )
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection: {e}")
        raise e


def index_chunks_in_qdrant(source_id: str, course_id: str, chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Generates embeddings and indexes chunks in Qdrant.
    Returns the list of point IDs.
    """
    logger.info(f"Indexing {len(chunks)} chunks in Qdrant for source {source_id}...")
    client = get_qdrant_client()
    collection_name = COLLECTION_NAME
    init_qdrant_collection(client, collection_name)
    
    point_ids = []
    points = []
    
    for chunk in chunks:
        vector = get_embedding(chunk["content"])
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)
        
        payload = {
            "source_id": source_id,
            "course_id": course_id,
            "content": chunk["content"],
            "topic_summary": chunk["topic_summary"],
            "speaker": chunk["speaker"],
            "confidence_score": chunk["confidence_score"],
            "ts_start": chunk["ts_start"],
            "ts_end": chunk["ts_end"],
            "category": chunk["category"]
        }
        
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )
        
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info("Successfully indexed points in Qdrant.")
        
    return point_ids


def delete_source_from_qdrant(source_id: str):
    """
    Deletes all points associated with the given source_id.
    """
    logger.info(f"Deleting all vectors for source_id {source_id} from Qdrant...")
    client = get_qdrant_client()
    collection_name = COLLECTION_NAME
    
    if client.collection_exists(collection_name=collection_name):
        client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source_id",
                            match=qmodels.MatchValue(value=source_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Successfully deleted vectors for source_id {source_id}.")


def search_knowledge_base(query: str, course_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Performs cosine similarity vector search on Qdrant collection.
    """
    client = get_qdrant_client()
    collection_name = COLLECTION_NAME
    
    if not client.collection_exists(collection_name=collection_name):
        return []
        
    query_vector = get_embedding(query)
    
    search_filter = None
    if course_id:
        search_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="course_id",
                    match=qmodels.MatchValue(value=course_id)
                )
            ]
        )
        
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=limit
    )
    
    hits = []
    for hit in results:
        hits.append({
            "id": hit.id,
            "score": round(hit.score, 4),
            "payload": hit.payload
        })
        
    return hits


def get_course_chunks(course_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all indexed chunks for a course_id from Qdrant.
    """
    client = get_qdrant_client()
    collection_name = COLLECTION_NAME
    if not client.collection_exists(collection_name=collection_name):
        return []
    
    # Scroll to get points matching the course_id filter
    result, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="course_id",
                    match=qmodels.MatchValue(value=course_id)
                )
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    
    chunks = []
    for point in result:
        payload = point.payload
        payload["id"] = point.id
        chunks.append(payload)
    return chunks


def generate_quiz_question(chunk_content: str, bloom_level: str) -> Dict[str, Any]:
    """
    Calls Ollama to generate a single multiple choice question grounded in chunk_content.
    """
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    prompt = f"""Eres un experto en diseño instruccional universitario.
Basándote ÚNICAMENTE en el siguiente contenido de clase, genera UNA pregunta de opción múltiple con 4 opciones (a, b, c, d).

CONTENIDO DE CLASE:
"{chunk_content}"

REQUISITOS:
- Nivel de Taxonomía de Bloom objetivo: {bloom_level}
- La respuesta correcta debe poder verificarse y justificarse de manera directa y factual con el contenido dado.
- Los 3 distractores deben ser plausibles pero claramente incorrectos según el contenido de clase.
- No generes preguntas sobre detalles triviales, saludos, o anécdotas personales.

Responde únicamente con un objeto JSON válido que siga exactamente esta estructura:
{{
  "question": "texto de la pregunta...",
  "options": {{
    "a": "opción a...",
    "b": "opción b...",
    "c": "opción c...",
    "d": "opción d..."
  }},
  "correct_option": "a",
  "bloom_level": "{bloom_level}",
  "justification": "justificación basada en el texto..."
}}
"""
    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False
                }
            )
            resp.raise_for_status()
            return json.loads(resp.json()["response"])
    except Exception as e:
        logger.error(f"Failed to generate quiz question: {e}")
        return {}


def validate_question(question_data: Dict[str, Any], chunk_content: str) -> Dict[str, Any]:
    """
    LLM-as-a-judge validation: rates factual grounding, distractor quality, and relevance.
    Returns score and reason.
    """
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    
    options = question_data.get("options", {})
    prompt = f"""Eres un evaluador de preguntas de examen universitario. Tu objetivo es juzgar la calidad y fidelidad factual de la pregunta generada a partir de un fragmento de clase.

FRAGMENTO DE CLASE:
"{chunk_content}"

PREGUNTA EVALUADA:
Pregunta: {question_data.get('question')}
Opciones:
a) {options.get('a')}
b) {options.get('b')}
c) {options.get('c')}
d) {options.get('d')}
Respuesta correcta: {question_data.get('correct_option')}

REGLAS DE EVALUACIÓN:
1. Grounding factual (0.0 a 1.0): ¿La respuesta correcta está totalmente respaldada y demostrada de forma directa por el fragmento de clase? Si requiere suposiciones externas, dale puntaje bajo.
2. Calidad de distractores (0.0 a 1.0): ¿Los distractores son plausibles pero indiscutiblemente falsos según el fragmento?
3. Relevancia (0.0 a 1.0): ¿La pregunta evalúa conceptos clave y no detalles insignificantes?

Calcula el promedio general de estas 3 reglas como un valor entre 0.0 y 1.0.
Responde únicamente con un objeto JSON válido con este formato:
{{"score": 0.85, "reason": "explicación breve de la evaluación"}}
"""
    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False
                }
            )
            resp.raise_for_status()
            res_json = json.loads(resp.json()["response"])
            return {
                "score": float(res_json.get("score", 0.0)),
                "reason": res_json.get("reason", "No reason provided")
            }
    except Exception as e:
        logger.error(f"Failed to validate question: {e}")
        return {"score": 0.0, "reason": f"Validation error: {e}"}


def generate_quiz(course_id: str, num_questions: int = 5, bloom_level: str = "comprender") -> List[Dict[str, Any]]:
    """
    RAG quiz generator: retrieves course chunks, prompts LLM, validates quality, and prevents duplicates.
    """
    logger.info(f"Generating quiz for course_id {course_id} with {num_questions} questions at bloom level {bloom_level}")
    timer = StageTimer("quiz_pipeline")

    # 1. Retrieve course chunks
    chunks = get_course_chunks(course_id)
    timer.mark("Qdrant Chunk Retrieval")
    if not chunks:
        logger.warning(f"No chunks found for course_id {course_id}")
        return []
    
    # Filter to ACADEMICO chunks only
    academic_chunks = [c for c in chunks if c.get("category") == "ACADEMICO"]
    if not academic_chunks:
        # Fallback to all chunks if no academic chunks found
        academic_chunks = chunks
        
    import random
    # Shuffle to cover different parts of the course
    random.shuffle(academic_chunks)
    
    generated_questions = []
    question_embeddings = []
    
    # Try generating questions from chunks
    for chunk in academic_chunks:
        if len(generated_questions) >= num_questions:
            break
            
        chunk_content = chunk.get("content", "")
        chunk_id = chunk.get("id")
        
        # A. Generate question
        q_data = generate_quiz_question(chunk_content, bloom_level)
        if not q_data or "question" not in q_data:
            continue
            
        # B. Validate via LLM-as-a-judge
        val_res = validate_question(q_data, chunk_content)
        val_score = val_res.get("score", 0.0)
        logger.info(f"Question validation score: {val_score} (Reason: {val_res.get('reason')})")
        if val_score < 0.75:
            logger.info("Discarding question due to low validation score.")
            continue
            
        # C. Deduplication check
        q_text = q_data.get("question", "")
        q_emb = get_embedding(q_text)
        
        is_duplicate = False
        for old_emb in question_embeddings:
            sim = cosine_similarity(q_emb, old_emb)
            if sim > 0.92:
                logger.info(f"Discarding duplicate question. Similarity: {sim:.4f}")
                is_duplicate = True
                break
                
        if is_duplicate:
            continue
            
        # D. Add metadata and store
        q_data["chunk_id"] = chunk_id
        q_data["validation_score"] = val_score
        q_data["validation_reason"] = val_res.get("reason", "")
        
        generated_questions.append(q_data)
        question_embeddings.append(q_emb)
        
    timer.mark("Question Generation & Validation")
    logger.info(f"Generated {len(generated_questions)} valid questions.")
    logger.info(f"Total quiz generation time: {_format_elapsed(timer.total())}")

    save_quiz(course_id, bloom_level, generated_questions, timer.stages)
    return generated_questions


# ==========================================
# Orchestration Pipeline
# ==========================================

def process_pipeline(input_path: str, source_id: str, course_id: str, model_size: str = "base", hf_token: str = None) -> Dict[str, Any]:
    """
    Orchestrates the entire audio/video processing pipeline:
    Extract Audio -> Transcribe -> Diarize -> Align -> Semantic Chunking -> Relevance Filter -> Embeddings -> Qdrant Indexing.
    """
    temp_dir = tempfile.gettempdir()
    audio_output = os.path.join(temp_dir, f"extracted_{os.path.basename(input_path)}.wav")

    pipeline_start = datetime.now()
    timer = StageTimer("ingest_pipeline")
    logger.info(f"Pipeline started at: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. Extract audio
        extract_audio(input_path, audio_output)
        timer.mark("Audio Extraction")

        # 2. Transcribe
        transcripts = transcribe_audio(audio_output, model_size=model_size)
        timer.mark("Whisper Transcription")

        # 3. Diarize
        diarization = diarize_audio(audio_output, hf_token=hf_token)
        logger.info(f"Raw diarization intervals: {diarization}")
        timer.mark("Diarization")

        # 4. Align
        aligned = align_segments(transcripts, diarization)
        timer.mark("Alignment")

        # Persist transcript for quality monitoring
        save_transcript(input_path, model_size, aligned)

        # 5. Semantic Chunking (valley similarity threshold = 0.5)
        chunks = generate_semantic_chunks(aligned, similarity_threshold=0.5)
        timer.mark("Semantic Chunking")

        # 6. Cleaning, Classification, Summarization
        processed_chunks = []
        for chunk in chunks:
            # Classify relevance
            classification = classify_chunk(chunk["content"], chunk["speaker"])
            category = classification["category"]

            # Relevance Filter: Discard OFF_TOPIC or ERROR_ESTUDIANTIL
            if category in {"OFF_TOPIC", "ERROR_ESTUDIANTIL"}:
                logger.info(f"Auditing chunk {chunk['chunk_index']} as {category}: {classification['reason']}. Keeping for POC/testing.")
                # for the POC, we do not discard so that tests can run with any sample media
                # continue

            # Generate summary topic
            topic_summary = generate_topic_summary(chunk["content"])

            chunk["category"] = category
            chunk["confidence_score"] = classification["confidence"]
            chunk["topic_summary"] = topic_summary
            processed_chunks.append(chunk)
        timer.mark("Classification & Summarization")

        # 7. Qdrant Indexing
        point_ids = index_chunks_in_qdrant(source_id, course_id, processed_chunks)
        timer.mark("Qdrant Indexing")

        # Attach the point IDs back to the chunks
        for chunk, pid in zip(processed_chunks, point_ids):
            chunk["qdrant_vector_id"] = pid

        full_text = " ".join([c["content"] for c in processed_chunks])

        pipeline_end = datetime.now()
        total = timer.total()
        logger.info(f"Pipeline finished at: {pipeline_end.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total elapsed time: {_format_elapsed(total)}")

        return {
            "text": full_text,
            "chunks": processed_chunks,
            "qdrant_collection": COLLECTION_NAME,
            "timings": timer.stages,
            "total_seconds": round(total, 3),
        }
        
    finally:
        # Secure cleanup of temporary WAV file
        if os.path.exists(audio_output):
            try:
                os.remove(audio_output)
                logger.info(f"Cleaned up temporary audio: {audio_output}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {audio_output}: {e}")
