from typing import Dict, Any, List
from src.rag.hybrid_retriever import HybridRetriever
from src.services.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("InterviewService")

class InterviewService:
    """
    RAG-driven Interview Simulator.
    Retrieves real technical questions matching the candidate's exact gaps,
    and provides rubric-grounded answer evaluation.
    """
    def __init__(self, hybrid_retriever: HybridRetriever, llm_client: LLMClient):
        self.retriever = hybrid_retriever
        self.llm = llm_client

    def generate_interview_prep(self, missing_skills: List[str], target_role: str = "AI Engineer") -> Dict[str, Any]:
        """
        Retrieves interview questions from the vector store targeted specifically at identified gaps.
        """
        query = f"{target_role} " + " ".join(missing_skills[:5])
        retrieved_questions = self.retriever.retrieve(
            query=query,
            top_k=3,
            where_filter={"type": "interview_question"}
        )

        q_context = "\n---\n".join([doc["content"] for doc in retrieved_questions])

        prompt = (
            f"You are a Staff AI Engineer conducting a technical screening interview.\n"
            f"Generate a customized technical interview session for a candidate targeting '{target_role}'.\n"
            f"The candidate has identified skill gaps in: {', '.join(missing_skills) if missing_skills else 'Advanced AI/ML Concepts'}.\n\n"
            f"CURATED INTERVIEW KNOWLEDGE BASE (Retrieved Ground Truth):\n{q_context}\n\n"
            f"Provide:\n"
            f"1. 3 Tailored Interview Questions (incorporating the retrieved concepts)\n"
            f"2. For each question: 'What the interviewer is secretly testing for'\n"
            f"3. Ideal Model Answer breakdown (including architectural trade-offs to mention)\n"
            f"4. Common Pitfalls to avoid"
        )

        response = self.llm.generate(prompt, task_type="interview_prep")

        return {
            "retrieved_questions": retrieved_questions,
            "interview_guide": response
        }

    def evaluate_answer(self, question: str, user_answer: str) -> str:
        """
        Evaluates a candidate's answer against technical accuracy and depth.
        """
        prompt = (
            f"You are an AI Engineering Interviewer evaluating a candidate's response.\n\n"
            f"QUESTION: {question}\n\n"
            f"CANDIDATE'S ANSWER:\n{user_answer}\n\n"
            f"Provide constructive feedback:\n"
            f"1. Score (out of 10) and Grade (Hire / Lean Hire / No Hire)\n"
            f"2. What was strong\n"
            f"3. What was missing or technically imprecise\n"
            f"4. How to elevate this answer to Senior level"
        )
        return self.llm.generate(prompt, task_type="interview_eval")
