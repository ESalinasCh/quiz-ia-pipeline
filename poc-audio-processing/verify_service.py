import urllib.request
import urllib.error
import json
import os
import uuid

def send_multipart_request(url, filepath, model_size="tiny", source_id=None, course_id=None):
    """
    Sends a multipart/form-data POST request using built-in urllib.
    """
    boundary = '==BoundaryRef=='
    body = []
    
    # 1. Add file part
    filename = os.path.basename(filepath)
    body.append(f'--{boundary}'.encode())
    body.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode())
    body.append(b'Content-Type: audio/wav')
    body.append(b'')
    with open(filepath, "rb") as f:
        body.append(f.read())
        
    # 2. Add model_size
    body.append(f'--{boundary}'.encode())
    body.append(b'Content-Disposition: form-data; name="model_size"')
    body.append(b'')
    body.append(model_size.encode())

    # 3. Add source_id
    if source_id:
        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="source_id"')
        body.append(b'')
        body.append(source_id.encode())

    # 4. Add course_id
    if course_id:
        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="course_id"')
        body.append(b'')
        body.append(course_id.encode())
    
    # 5. Add closing boundary
    body.append(f'--{boundary}--'.encode())
    body.append(b'')
    
    payload = b'\r\n'.join(body)
    
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(payload))
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            return response.getcode(), json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        try:
            err_data = json.loads(e.read().decode())
        except Exception:
            err_data = e.reason
        return e.code, err_data
    except Exception as e:
        return 0, str(e)


def send_json_request(url, data, method="POST"):
    """
    Sends a JSON request using built-in urllib.
    """
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'Content-Length': str(len(payload))
        },
        method=method
    )
    try:
        with urllib.request.urlopen(req) as response:
            return response.getcode(), json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        try:
            err_data = json.loads(e.read().decode())
        except Exception:
            err_data = e.reason
        return e.code, err_data
    except Exception as e:
        return 0, str(e)


def run_tests():
    url_transcribe = "http://127.0.0.1:8000/api/pipeline/transcribe"
    url_search = "http://127.0.0.1:8000/api/kb/search"
    url_delete_base = "http://127.0.0.1:8000/api/kb/source"
    
    test_wav = "/home/ubuntu/quiz/test01_20s.wav"
    if not os.path.exists(test_wav):
        print(f"Error: {test_wav} does not exist.")
        return

    # Generate secure UUIDs for testing
    source_id = str(uuid.uuid4())
    course_id = str(uuid.uuid4())

    print("\n--- TEST 1: Full Ingestion, Semantic Chunking & Vectorization ---")
    print(f"Using source_id: {source_id}, course_id: {course_id}")
    code, resp = send_multipart_request(
        url_transcribe, 
        test_wav, 
        model_size="tiny", 
        source_id=source_id, 
        course_id=course_id
    )
    print(f"Status Code: {code}")
    print(f"Response:\n{json.dumps(resp, indent=2)}")
    
    assert code == 200, f"Expected 200, got {code}"
    assert "chunks" in resp, "Response missing 'chunks'"
    chunks = resp["chunks"]
    assert len(chunks) > 0, "No semantic chunks generated."
    
    # Validate structure of a chunk
    first_chunk = chunks[0]
    assert "qdrant_vector_id" in first_chunk, "Missing Qdrant point ID"
    assert "topic_summary" in first_chunk, "Missing topic summary"
    assert "category" in first_chunk, "Missing relevance category"
    print("Test 1 Passed!")

    print("\n--- TEST 2: Knowledge Base Vector Search ---")
    search_query = {"query": "masquerade", "course_id": course_id, "limit": 2}
    code, resp = send_json_request(url_search, search_query)
    print(f"Status Code: {code}")
    print(f"Response:\n{json.dumps(resp, indent=2)}")
    
    assert code == 200, f"Expected 200, got {code}"
    assert "hits" in resp, "Response missing 'hits'"
    assert len(resp["hits"]) > 0, "Vector search returned 0 matches"
    assert resp["hits"][0]["payload"]["source_id"] == source_id, "Search hit belongs to wrong source"
    print("Test 2 Passed!")

    print("\n--- TEST 3: Security checks (UUID format & model validation) ---")
    # Bad UUID check
    code, resp = send_multipart_request(
        url_transcribe, 
        test_wav, 
        model_size="tiny", 
        source_id="invalid-uuid-string"
    )
    print(f"Status Code: {code} (Expected 400)")
    assert code == 400, f"Expected 400 for invalid UUID, got {code}"
    
    # Bad search limit check
    code, resp = send_json_request(url_search, {"query": "test", "limit": 999})
    print(f"Status Code: {code} (Expected 422/400)")
    assert code in (400, 422), f"Expected validation error, got {code}"
    print("Test 3 Passed!")

    print("\n--- TEST 3.5: RAG Quiz Generation ---")
    url_generate = "http://127.0.0.1:8000/api/quizzes/generate"
    quiz_payload = {
        "course_id": course_id,
        "num_questions": 2,
        "bloom_level": "comprender"
    }
    code, resp = send_json_request(url_generate, quiz_payload)
    print(f"Status Code: {code}")
    print(f"Response:\n{json.dumps(resp, indent=2)}")
    
    assert code == 200, f"Expected 200, got {code}"
    assert "questions" in resp, "Response missing 'questions'"
    questions = resp["questions"]
    assert len(questions) > 0, "No questions were generated"
    
    # Check structure of generated questions
    for q in questions:
        assert "question" in q, "Question missing 'question' text"
        assert "options" in q, "Question missing 'options'"
        assert "correct_option" in q, "Question missing 'correct_option'"
        assert q["correct_option"] in ("a", "b", "c", "d"), f"Invalid correct_option: {q['correct_option']}"
        assert "bloom_level" in q, "Question missing 'bloom_level'"
        assert "validation_score" in q, "Question missing 'validation_score'"
        assert q["validation_score"] >= 0.75, f"Validation score lower than threshold: {q['validation_score']}"
        
    print("Test 3.5 Passed!")

    print("\n--- TEST 4: Vector Deletion ---")
    url_delete = f"{url_delete_base}/{source_id}"
    code, resp = send_json_request(url_delete, {}, method="DELETE")
    print(f"Status Code: {code}")
    print(f"Response: {resp}")
    assert code == 200, f"Expected 200, got {code}"
    print("Test 4 Passed!")

    print("\n--- TEST 5: Verify Deletion worked (Search returns empty) ---")
    code, resp = send_json_request(url_search, {"query": "masquerade", "course_id": course_id})
    print(f"Status Code: {code}")
    print(f"Response:\n{json.dumps(resp, indent=2)}")
    assert code == 200, f"Expected 200, got {code}"
    # Hits should be empty now for this course/source since it was deleted
    hits = [h for h in resp["hits"] if h["payload"]["source_id"] == source_id]
    assert len(hits) == 0, f"Expected 0 hits for source_id, got {len(hits)}"
    print("Test 5 Passed!")

    print("\nAll integration and vector database tests completed successfully!")

if __name__ == "__main__":
    run_tests()
