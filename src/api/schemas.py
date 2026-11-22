from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(default="1.0.0", description="API version")
    indexed_documents: int = Field(..., description="Number of knowledge base documents in ChromaDB")

# --- 1. Analyze / Gap Analysis Schemas ---
class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=10, description="Full text or parsed sections of the resume")
    job_description: str = Field(..., min_length=10, description="Target job description")
    target_role: str = Field(default="AI Engineer", description="Target engineering role")

class MatchMetrics(BaseModel):
    overall_match_score: float
    keyword_match_pct: float
    semantic_match_pct: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    total_jd_keywords: int

class RetrievedChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    rrf_score: Optional[float] = None
    similarity: Optional[float] = None

class AnalyzeResponse(BaseModel):
    metrics: MatchMetrics
    ai_insights: str
    retrieved_context: List[RetrievedChunk]

# --- 2. Resume Enhancer Schemas ---
class EnhanceResumeRequest(BaseModel):
    resume_text: str = Field(..., min_length=10, description="Candidate resume text or bullet points")
    job_description: Optional[str] = Field(default="", description="Optional target job description context")

class EnhanceResumeResponse(BaseModel):
    enhanced_content: str
    retrieved_patterns: List[RetrievedChunk]

# --- 3. Interview Prep & Evaluation Schemas ---
class InterviewPrepRequest(BaseModel):
    missing_skills: List[str] = Field(default_factory=list, description="Identified technical skill gaps")
    target_role: str = Field(default="AI Engineer", description="Target role title")

class InterviewPrepResponse(BaseModel):
    interview_guide: str
    retrieved_questions: List[RetrievedChunk]

class InterviewEvalRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Interview question being answered")
    user_answer: str = Field(..., min_length=3, description="Candidate's technical answer")

class InterviewEvalResponse(BaseModel):
    question: str
    evaluation: str

# --- 4. Roadmap Schemas ---
class RoadmapRequest(BaseModel):
    missing_skills: List[str] = Field(default_factory=list, description="Identified technical skill gaps")
    target_role: str = Field(default="AI Engineer", description="Target engineering role")
    duration_weeks: int = Field(default=4, ge=1, le=12, description="Roadmap duration in weeks")

class RoadmapResponse(BaseModel):
    roadmap_plan: str
    retrieved_context: List[RetrievedChunk]
