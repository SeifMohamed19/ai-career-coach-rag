from typing import Dict, Any, List
from src.rag.hybrid_retriever import HybridRetriever
from src.services.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("RoadmapService")

class RoadmapService:
    """
    RAG-grounded personalized learning roadmap generator.
    Translates skill gaps into a structured, project-based engineering curriculum.
    """
    def __init__(self, hybrid_retriever: HybridRetriever, llm_client: LLMClient):
        self.retriever = hybrid_retriever
        self.llm = llm_client

    def generate_roadmap(
        self,
        missing_skills: List[str],
        target_role: str = "AI Engineer",
        duration_weeks: int = 4
    ) -> Dict[str, Any]:
        """
        Retrieves competency benchmarks and produces a week-by-week implementation-focused roadmap.
        """
        # Retrieve competency standards
        retrieved_competencies = self.retriever.retrieve(
            query=f"{target_role} skills tools architecture",
            top_k=2,
            where_filter={"type": "competency"}
        )
        comp_context = "\n---\n".join([doc["content"] for doc in retrieved_competencies])

        prompt = (
            f"You are a Principal AI Architect and Career Mentor.\n"
            f"Design an aggressive, project-driven {duration_weeks}-Week Learning Roadmap for an aspiring {target_role}.\n"
            f"Identified Technical Gaps: {', '.join(missing_skills) if missing_skills else 'Advanced AI/ML Engineering'}\n\n"
            f"INDUSTRY BENCHMARKS:\n{comp_context}\n\n"
            f"Guidelines for the Roadmap:\n"
            f"- Prioritize hands-on building over passive reading.\n"
            f"- For each week, provide:\n"
            f"  * Core Concept & Why It Matters in Production\n"
            f"  * 1 Milestone Mini-Project (something they can push to GitHub)\n"
            f"  * Free, high-quality open-source learning resources/docs\n"
            f"  * 'Interview Checkpoint' question they must be able to answer by Sunday."
        )

        response = self.llm.generate(prompt, task_type="roadmap")

        return {
            "retrieved_context": retrieved_competencies,
            "roadmap_plan": response
        }
