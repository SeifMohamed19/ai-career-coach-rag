import json
from pathlib import Path
from typing import List, Dict, Any
from src.config import COMPETENCIES_DIR, INTERVIEW_DIR, RESUME_PATTERNS_DIR
from src.rag.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("KnowledgeBaseIngestor")

class KnowledgeBaseIngestor:
    """
    Ingests curated domain files (Competencies, Interview Questions, Resume Patterns)
    into the ChromaDB vector database.
    """
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def ingest_all(self, force_refresh: bool = False) -> int:
        """
        Parses all knowledge files and loads them into ChromaDB if collection is empty
        or if force_refresh is True.
        """
        current_count = self.vector_store.count()
        if current_count > 0 and not force_refresh:
            logger.info(f"Knowledge base already contains {current_count} documents. Skipping re-indexing.")
            return current_count

        logger.info("Ingesting knowledge base documents into ChromaDB...")
        ids, documents, metadatas = [], [], []

        # 1. Competencies
        if COMPETENCIES_DIR.exists():
            for filepath in COMPETENCIES_DIR.glob("*.json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    role = data.get("role", filepath.stem)
                    for level, level_data in data.get("levels", {}).items():
                        skills = ", ".join(level_data.get("required_skills", []))
                        tools = ", ".join(level_data.get("tools", []))
                        concepts = "; ".join(level_data.get("architectural_concepts", []))

                        content = (
                            f"Role: {role} ({level})\n"
                            f"Required Core Skills: {skills}\n"
                            f"Key Technologies & Tools: {tools}\n"
                            f"System & Architectural Concepts: {concepts}"
                        )
                        doc_id = f"comp_{role.lower().replace(' ', '_')}_{level}"
                        ids.append(doc_id)
                        documents.append(content)
                        metadatas.append({
                            "type": "competency",
                            "role": role,
                            "level": level,
                            "source": filepath.name
                        })

        # 2. Interview Questions
        if INTERVIEW_DIR.exists():
            for filepath in INTERVIEW_DIR.glob("*.json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    topic = data.get("topic", "Technical")
                    for q in data.get("questions", []):
                        points = "; ".join(q.get("key_points_to_cover", []))
                        tags = ", ".join(q.get("tags", []))
                        content = (
                            f"Topic: {topic}\n"
                            f"Question: {q.get('question')}\n"
                            f"Difficulty: {q.get('difficulty', 'Medium')}\n"
                            f"Expected Answers & Key Concepts: {points}\n"
                            f"Tags: {tags}"
                        )
                        doc_id = f"interview_{q.get('id', hash(q.get('question')))}"
                        ids.append(doc_id)
                        documents.append(content)
                        metadatas.append({
                            "type": "interview_question",
                            "topic": topic,
                            "difficulty": q.get("difficulty", "Medium"),
                            "source": filepath.name
                        })

        # 3. Resume Patterns
        if RESUME_PATTERNS_DIR.exists():
            for filepath in RESUME_PATTERNS_DIR.glob("*.json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    framework = data.get("framework", "STAR")
                    for idx, ex in enumerate(data.get("transformation_examples", [])):
                        content = (
                            f"Framework: {framework}\n"
                            f"Weak Example: {ex.get('weak')}\n"
                            f"Strong Metric-Driven Transformation: {ex.get('strong')}"
                        )
                        doc_id = f"resume_pattern_{idx}"
                        ids.append(doc_id)
                        documents.append(content)
                        metadatas.append({
                            "type": "resume_pattern",
                            "framework": framework,
                            "source": filepath.name
                        })

        if documents:
            self.vector_store.add_documents(ids=ids, documents=documents, metadatas=metadatas)
            logger.info(f"Successfully ingested {len(documents)} knowledge chunks into ChromaDB.")

        return len(documents)
