import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Ensure stdout handles UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag.vector_store import ChromaVectorStore
from src.rag.knowledge_base import KnowledgeBaseIngestor
from src.rag.hybrid_retriever import HybridRetriever
from src.services.llm_client import LLMClient
from src.services.analyzer_service import AnalyzerService
from src.parsers.resume_parser import ResumeParser

def run_smoke_test():
    print("==================================================")
    print("[RUN] RUNNING RAG CAREER COACH PIPELINE SMOKE TEST")
    print("==================================================")

    # 1. Initialize Vector Store & Ingest
    print("\n[Step 1] Initializing Vector Store and Ingesting Knowledge Base...")
    vector_store = ChromaVectorStore(collection_name="career_knowledge_base_test")
    ingestor = KnowledgeBaseIngestor(vector_store)
    count = ingestor.ingest_all(force_refresh=True)
    print(f"[OK] Ingestion complete. Indexed documents count: {count}")
    assert count > 0, "No documents were indexed!"

    # 2. Hybrid Retriever
    print("\n[Step 2] Testing Hybrid Retriever (BM25 + Dense RRF)...")
    retriever = HybridRetriever(vector_store)
    results = retriever.retrieve(query="What is Hybrid Search with BM25?", top_k=2)
    print(f"[OK] Retrieved {len(results)} chunks.")
    for i, r in enumerate(results, 1):
        print(f"   Chunk {i} (RRF Score: {r.get('rrf_score')}): {r['content'][:80]}...")
    assert len(results) > 0, "Retriever failed to return chunks!"

    # 3. Resume Parser
    print("\n[Step 3] Testing Resume Parser...")
    sample_resume = (BASE_DIR / "samples" / "sample_resume.txt").read_text(encoding="utf-8")
    sections = ResumeParser.extract_sections(sample_resume)
    print(f"[OK] Identified sections: {list(sections.keys())}")

    # 4. End-to-End Analysis
    print("\n[Step 4] Testing End-to-End Analysis (AnalyzerService)...")
    sample_jd = (BASE_DIR / "samples" / "sample_job_description.txt").read_text(encoding="utf-8")
    llm = LLMClient() # Offline / Gemini fallback
    analyzer = AnalyzerService(retriever, llm)
    analysis = analyzer.analyze(sample_resume, sample_jd, target_role="AI Engineer")

    metrics = analysis["metrics"]
    print(f"[OK] Match Metrics Calculated:")
    print(f"   - Overall Fit Score: {metrics['overall_match_score']}%")
    print(f"   - Keyword Match: {metrics['keyword_match_pct']}%")
    print(f"   - Semantic Alignment: {metrics['semantic_match_pct']}%")
    print(f"   - Matched Skills: {metrics['matched_keywords']}")
    print(f"   - Missing Skills: {metrics['missing_keywords']}")

    print("\n==================================================")
    print("[SUCCESS] ALL TESTS PASSED! RAG PIPELINE OPERATIONAL!")
    print("==================================================")

if __name__ == "__main__":
    run_smoke_test()
