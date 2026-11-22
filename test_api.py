import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Ensure stdout handles UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_pipeline():
    print("==================================================")
    print("[RUN] RUNNING FASTAPI ENDPOINT & SCHEMA VERIFICATION")
    print("==================================================")

    # 1. Health Endpoint
    print("\n[Step 1] Testing GET /health...")
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed with {res.status_code}: {res.text}"
    data = res.json()
    print(f"[OK] Health check passed: status={data['status']}, indexed_docs={data['indexed_documents']}")
    assert data["status"] == "healthy"
    assert data["indexed_documents"] > 0

    # 2. Analyze Endpoint
    print("\n[Step 2] Testing POST /api/v1/analyze...")
    sample_resume = (BASE_DIR / "samples" / "sample_resume.txt").read_text(encoding="utf-8")
    sample_jd = (BASE_DIR / "samples" / "sample_job_description.txt").read_text(encoding="utf-8")

    payload = {
        "resume_text": sample_resume,
        "job_description": sample_jd,
        "target_role": "AI Engineer"
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200, f"Analyze endpoint failed: {res.text}"
    analyze_data = res.json()
    metrics = analyze_data["metrics"]
    print(f"[OK] Fit Analysis Succeeded:")
    print(f"     - Overall Match: {metrics['overall_match_score']}%")
    print(f"     - Matched Keywords: {metrics['matched_keywords'][:4]}...")
    print(f"     - Retrieved Context Chunks: {len(analyze_data['retrieved_context'])}")
    assert metrics["overall_match_score"] > 0
    assert len(analyze_data["retrieved_context"]) > 0

    # 3. Resume Enhancer Endpoint
    print("\n[Step 3] Testing POST /api/v1/enhance-resume...")
    enh_payload = {
        "resume_text": "Built a computer vision app with YOLOv8. Integrated Gemini API in React Native.",
        "job_description": sample_jd
    }
    res = client.post("/api/v1/enhance-resume", json=enh_payload)
    assert res.status_code == 200, f"Enhance resume failed: {res.text}"
    enh_data = res.json()
    print(f"[OK] Resume Enhancer Succeeded. Content length: {len(enh_data['enhanced_content'])} chars")
    assert len(enh_data["enhanced_content"]) > 0

    # 4. Interview Prep Endpoint
    print("\n[Step 4] Testing POST /api/v1/interview/prep...")
    prep_payload = {
        "missing_skills": ["chromadb", "hybrid search", "evaluations"],
        "target_role": "AI Engineer"
    }
    res = client.post("/api/v1/interview/prep", json=prep_payload)
    assert res.status_code == 200, f"Interview prep failed: {res.text}"
    prep_data = res.json()
    print(f"[OK] Interview Prep Succeeded. Guide length: {len(prep_data['interview_guide'])} chars")
    assert len(prep_data["interview_guide"]) > 0

    # 5. Interview Answer Evaluation Endpoint
    print("\n[Step 5] Testing POST /api/v1/interview/evaluate...")
    eval_payload = {
        "question": "Why use Hybrid Search with BM25 and Dense Vectors in a production RAG system?",
        "user_answer": "Dense vectors capture semantic intent, while BM25 captures exact technical keywords. We fuse their ranks using Reciprocal Rank Fusion."
    }
    res = client.post("/api/v1/interview/evaluate", json=eval_payload)
    assert res.status_code == 200, f"Interview eval failed: {res.text}"
    eval_data = res.json()
    print(f"[OK] Answer Evaluation Succeeded: {eval_data['evaluation'][:100]}...")
    assert "Evaluation:" in eval_data["evaluation"] or "Score" in eval_data["evaluation"]

    # 6. Roadmap Endpoint
    print("\n[Step 6] Testing POST /api/v1/roadmap...")
    road_payload = {
        "missing_skills": ["chromadb", "hybrid search", "evaluations"],
        "target_role": "AI Engineer",
        "duration_weeks": 4
    }
    res = client.post("/api/v1/roadmap", json=road_payload)
    assert res.status_code == 200, f"Roadmap failed: {res.text}"
    road_data = res.json()
    print(f"[OK] Roadmap Succeeded. Plan length: {len(road_data['roadmap_plan'])} chars")
    assert len(road_data["roadmap_plan"]) > 0

    print("\n==================================================")
    print("[SUCCESS] ALL FASTAPI ENDPOINTS & SCHEMAS VERIFIED!")
    print("==================================================")

if __name__ == "__main__":
    test_api_pipeline()
