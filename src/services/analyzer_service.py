import re
from typing import Dict, Any, List
import numpy as np
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.embeddings import LocalEmbeddingEngine
from src.services.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("AnalyzerService")

class AnalyzerService:
    """
    Performs comprehensive Gap Analysis between Resume and Job Description,
    grounded in industry competency frameworks via Hybrid RAG.
    """
    # Common tech keywords for lexical matching
    TECH_KEYWORDS = [
        "python", "pytorch", "tensorflow", "keras", "scikit-learn", "numpy", "pandas",
        "sql", "docker", "kubernetes", "git", "fastapi", "flask", "django", "aws", "gcp",
        "azure", "rag", "langchain", "llamaindex", "chromadb", "faiss", "pinecone", "weaviate",
        "transformers", "huggingface", "llm", "lora", "qlora", "mlops", "ci/cd", "rest", "graphql",
        "yolo", "opencv", "cuda", "onnx", "triton", "vllm", "ollama", "ragas", "evaluations"
    ]

    def __init__(self, hybrid_retriever: HybridRetriever, llm_client: LLMClient):
        self.retriever = hybrid_retriever
        self.llm = llm_client
        self.embedding_engine = LocalEmbeddingEngine()

    def extract_keywords(self, text: str) -> set:
        """Extracts recognized technical keywords from raw text."""
        words = set(re.findall(r"\b[a-zA-Z0-9_\-\.\/]+\b", text.lower()))
        return {kw for kw in self.TECH_KEYWORDS if kw in words or kw in text.lower()}

    def compute_match_metrics(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Computes both Lexical (Keyword) and Semantic (Vector Cosine) overlap.
        """
        # 1. Lexical Keyword Overlap
        resume_kws = self.extract_keywords(resume_text)
        jd_kws = self.extract_keywords(jd_text)

        matched_kws = resume_kws.intersection(jd_kws)
        missing_kws = jd_kws.difference(resume_kws)

        keyword_match_pct = (len(matched_kws) / max(len(jd_kws), 1)) * 100.0

        # 2. Semantic Vector Similarity
        resume_vec = np.array(self.embedding_engine.embed_query(resume_text[:2000]))
        jd_vec = np.array(self.embedding_engine.embed_query(jd_text[:2000]))

        # Cosine similarity
        norm_r = np.linalg.norm(resume_vec)
        norm_j = np.linalg.norm(jd_vec)
        if norm_r > 0 and norm_j > 0:
            semantic_sim = float(np.dot(resume_vec, jd_vec) / (norm_r * norm_j))
        else:
            semantic_sim = 0.0

        semantic_score_pct = max(0.0, min(100.0, semantic_sim * 100.0))

        # Blended Score: 50% keyword exact match + 50% semantic context match
        overall_score = round(0.5 * keyword_match_pct + 0.5 * semantic_score_pct, 1)

        return {
            "overall_match_score": overall_score,
            "keyword_match_pct": round(keyword_match_pct, 1),
            "semantic_match_pct": round(semantic_score_pct, 1),
            "matched_keywords": sorted(list(matched_kws)),
            "missing_keywords": sorted(list(missing_kws)),
            "total_jd_keywords": len(jd_kws)
        }

    def analyze(self, resume_text: str, jd_text: str, target_role: str = "AI Engineer") -> Dict[str, Any]:
        """
        Executes end-to-end skill gap analysis with RAG context injection.
        """
        # 1. Quantitative metrics
        metrics = self.compute_match_metrics(resume_text, jd_text)

        # 2. RAG Retrieval: Fetch industry competency standards for the role
        retrieved_competencies = self.retriever.retrieve(
            query=f"{target_role} required core skills and architecture competencies",
            top_k=2,
            where_filter={"type": "competency"}
        )
        competency_context = "\n---\n".join([doc["content"] for doc in retrieved_competencies])

        # 3. LLM Synthesis with Grounded Context
        prompt = (
            f"You are a Senior Technical Recruiter and AI Engineering Mentor.\n"
            f"Perform an objective, actionable Skill Gap Analysis.\n\n"
            f"TARGET ROLE: {target_role}\n\n"
            f"INDUSTRY COMPETENCY BENCHMARKS (Retrieved Ground Truth):\n{competency_context}\n\n"
            f"QUANTITATIVE METRICS:\n"
            f"- Overall Match: {metrics['overall_match_score']}%\n"
            f"- Matched Keywords: {', '.join(metrics['matched_keywords']) or 'None'}\n"
            f"- Missing Keywords: {', '.join(metrics['missing_keywords']) or 'None'}\n\n"
            f"JOB DESCRIPTION:\n{jd_text[:1500]}\n\n"
            f"CANDIDATE RESUME:\n{resume_text[:2000]}\n\n"
            f"Provide your evaluation in 3 structured sections:\n"
            f"1. Executive Summary & Fit Assessment\n"
            f"2. Core Strengths & Demonstrated Competencies\n"
            f"3. High-Priority Technical Gaps & Exact Steps to Address Them"
        )

        analysis_text = self.llm.generate(prompt, task_type="skill_gap")

        return {
            "metrics": metrics,
            "retrieved_context": retrieved_competencies,
            "ai_insights": analysis_text
        }
