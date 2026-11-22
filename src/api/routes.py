from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from src.api.schemas import (
    HealthResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    EnhanceResumeRequest,
    EnhanceResumeResponse,
    InterviewPrepRequest,
    InterviewPrepResponse,
    InterviewEvalRequest,
    InterviewEvalResponse,
    RoadmapRequest,
    RoadmapResponse,
)
from src.rag.vector_store import ChromaVectorStore
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.knowledge_base import KnowledgeBaseIngestor
from src.services.llm_client import LLMClient
from src.services.analyzer_service import AnalyzerService
from src.services.resume_enhancer import ResumeEnhancer
from src.services.interview_service import InterviewService
from src.services.roadmap_service import RoadmapService
from src.utils.logger import get_logger

logger = get_logger("API_Routes")
router = APIRouter()

# Global service container for dependency injection
class ServiceContainer:
    _instance = None

    def __init__(self):
        logger.info("Initializing API Service Container & Hybrid RAG Engine...")
        self.vector_store = ChromaVectorStore(collection_name="career_knowledge_base")
        self.ingestor = KnowledgeBaseIngestor(self.vector_store)
        self.ingestor.ingest_all(force_refresh=False)
        self.retriever = HybridRetriever(self.vector_store)
        self.llm = LLMClient()
        self.analyzer = AnalyzerService(self.retriever, self.llm)
        self.enhancer = ResumeEnhancer(self.retriever, self.llm)
        self.interview = InterviewService(self.retriever, self.llm)
        self.roadmap = RoadmapService(self.retriever, self.llm)
        logger.info("API Service Container initialized successfully.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_services() -> ServiceContainer:
    return ServiceContainer.get_instance()

# --- Health Check ---
@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(services: ServiceContainer = Depends(get_services)):
    """Health check endpoint returning service status and indexed document count."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        indexed_documents=services.vector_store.count()
    )

# --- 1. Gap Analysis Endpoint ---
@router.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["RAG Career Services"])
def analyze_fit(request: AnalyzeRequest, services: ServiceContainer = Depends(get_services)):
    """
    Computes lexical (BM25) and semantic (vector) fit, identifies skill gaps,
    and returns grounded AI recruiter insights.
    """
    try:
        result = services.analyzer.analyze(
            resume_text=request.resume_text,
            jd_text=request.job_description,
            target_role=request.target_role
        )
        return AnalyzeResponse(
            metrics=result["metrics"],
            ai_insights=result["ai_insights"],
            retrieved_context=result["retrieved_context"]
        )
    except Exception as e:
        logger.error(f"Error in analyze_fit endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. Resume Enhancer Endpoint ---
@router.post("/api/v1/enhance-resume", response_model=EnhanceResumeResponse, tags=["RAG Career Services"])
def enhance_resume(request: EnhanceResumeRequest, services: ServiceContainer = Depends(get_services)):
    """
    Transforms candidate bullet points using retrieved STAR heuristics and the Google XYZ formula.
    """
    try:
        result = services.enhancer.enhance_bullets(
            raw_bullets_or_experience=request.resume_text,
            target_jd=request.job_description or ""
        )
        return EnhanceResumeResponse(
            enhanced_content=result["enhanced_content"],
            retrieved_patterns=result["retrieved_patterns"]
        )
    except Exception as e:
        logger.error(f"Error in enhance_resume endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. Interview Prep & Evaluation Endpoints ---
@router.post("/api/v1/interview/prep", response_model=InterviewPrepResponse, tags=["Interview Prep"])
def generate_interview_prep(request: InterviewPrepRequest, services: ServiceContainer = Depends(get_services)):
    """
    Retrieves real technical interview questions targeting candidate skill gaps.
    """
    try:
        result = services.interview.generate_interview_prep(
            missing_skills=request.missing_skills,
            target_role=request.target_role
        )
        return InterviewPrepResponse(
            interview_guide=result["interview_guide"],
            retrieved_questions=result["retrieved_questions"]
        )
    except Exception as e:
        logger.error(f"Error in generate_interview_prep endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/v1/interview/evaluate", response_model=InterviewEvalResponse, tags=["Interview Prep"])
def evaluate_interview_answer(request: InterviewEvalRequest, services: ServiceContainer = Depends(get_services)):
    """
    Evaluates a candidate's technical interview response against engineering rubrics.
    """
    try:
        evaluation = services.interview.evaluate_answer(
            question=request.question,
            user_answer=request.user_answer
        )
        return InterviewEvalResponse(
            question=request.question,
            evaluation=evaluation
        )
    except Exception as e:
        logger.error(f"Error in evaluate_interview_answer endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. Career Roadmap Endpoint ---
@router.post("/api/v1/roadmap", response_model=RoadmapResponse, tags=["Career Roadmap"])
def generate_roadmap(request: RoadmapRequest, services: ServiceContainer = Depends(get_services)):
    """
    Generates a structured, project-driven career transition curriculum.
    """
    try:
        result = services.roadmap.generate_roadmap(
            missing_skills=request.missing_skills,
            target_role=request.target_role,
            duration_weeks=request.duration_weeks
        )
        return RoadmapResponse(
            roadmap_plan=result["roadmap_plan"],
            retrieved_context=result["retrieved_context"]
        )
    except Exception as e:
        logger.error(f"Error in generate_roadmap endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
