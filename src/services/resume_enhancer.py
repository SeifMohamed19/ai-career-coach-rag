from typing import Dict, Any, List
from src.rag.hybrid_retriever import HybridRetriever
from src.services.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("ResumeEnhancer")

class ResumeEnhancer:
    """
    RAG-powered Resume Bullet Optimizer.
    Retrieves proven STAR impact formulas and rewrites candidate experience.
    """
    def __init__(self, hybrid_retriever: HybridRetriever, llm_client: LLMClient):
        self.retriever = hybrid_retriever
        self.llm = llm_client

    def enhance_bullets(self, raw_bullets_or_experience: str, target_jd: str = "") -> Dict[str, Any]:
        """
        Retrieves resume patterns from vector store and rewrites bullets with metrics & STAR formatting.
        """
        # 1. Retrieve impact patterns from knowledge base
        retrieved_patterns = self.retriever.retrieve(
            query="Google XYZ formula STAR method action verbs quantifiable metrics",
            top_k=2,
            where_filter={"type": "resume_pattern"}
        )
        pattern_context = "\n---\n".join([doc["content"] for doc in retrieved_patterns])

        # 2. Build synthesis prompt
        prompt = (
            f"You are an Elite Resume Coach for top-tier Tech & AI companies.\n"
            f"Transform the following candidate experience bullet points into high-impact, ATS-optimized, "
            f"and metrics-driven accomplishments using the Google XYZ Formula: 'Accomplished [X] as measured by [Y], by doing [Z]'.\n\n"
            f"GROUND TRUTH RETRIEVED GUIDELINES & EXAMPLES:\n{pattern_context}\n\n"
            f"TARGET JOB REQUIREMENTS (Context):\n{target_jd[:1000] if target_jd else 'General AI/Software Engineering'}\n\n"
            f"CANDIDATE BULLETS TO REWRITE:\n{raw_bullets_or_experience}\n\n"
            f"For each bullet point, output:\n"
            f"- **Original**\n"
            f"- **Critique** (What was missing? Vagueness, missing metrics, weak verbs?)\n"
            f"- **Enhanced STAR Version** (Concrete tools, action verb, realistic metrics)\n"
            f"- **Why this ranks higher**"
        )

        response = self.llm.generate(prompt, task_type="resume_enhancer")

        return {
            "retrieved_patterns": retrieved_patterns,
            "enhanced_content": response
        }
