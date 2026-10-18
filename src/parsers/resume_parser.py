import re
from typing import Dict, Any, Optional
from io import BytesIO
import pypdf
from src.utils.logger import get_logger

logger = get_logger("ResumeParser")

class ResumeParser:
    """
    Parses PDF and plaintext resumes, extracts text content,
    and isolates common sections (Skills, Experience, Projects, Education).
    """

    SECTION_PATTERNS = {
        "skills": re.compile(r"(technical\s+skills|skills|technologies|proficiencies|core\s+competencies)", re.IGNORECASE),
        "experience": re.compile(r"(work\s+experience|professional\s+experience|experience|employment)", re.IGNORECASE),
        "projects": re.compile(r"(projects|technical\s+projects|selected\s+projects|key\s+projects)", re.IGNORECASE),
        "education": re.compile(r"(education|academic\s+background|qualifications)", re.IGNORECASE),
        "certifications": re.compile(r"(certifications|certificates|licenses)", re.IGNORECASE),
    }

    @classmethod
    def extract_text_from_pdf(cls, file_bytes: bytes) -> str:
        """Extracts and concatenates text from a PDF byte stream."""
        try:
            reader = pypdf.PdfReader(BytesIO(file_bytes))
            text_pages = []
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                text_pages.append(extracted)
            full_text = "\n".join(text_pages)
            return cls.clean_text(full_text)
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise ValueError(f"Failed to parse PDF resume: {e}")

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Normalizes whitespace and removes unprintable artifacts."""
        if not text:
            return ""
        # Replace non-breaking spaces and tabs
        text = text.replace("\xa0", " ").replace("\t", " ")
        # Reduce multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing whitespaces per line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    @classmethod
    def extract_sections(cls, full_text: str) -> Dict[str, str]:
        """
        Segments resume text into standard sections based on detected headers.
        """
        lines = full_text.split("\n")
        sections: Dict[str, list] = {"general": []}
        current_section = "general"

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            matched_section = None
            # Check if this line looks like a header (short length, matches pattern)
            if len(line_stripped.split()) <= 4:
                for sec_name, pattern in cls.SECTION_PATTERNS.items():
                    if pattern.search(line_stripped):
                        matched_section = sec_name
                        break

            if matched_section:
                current_section = matched_section
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line_stripped)

        return {k: "\n".join(v) for k, v in sections.items() if v}
