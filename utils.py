"""
utils.py
--------
Small reusable helper functions: converting ResumeData into JSON, CSV,
and a formatted text report, plus logging setup and file-validation
helpers. Keeping these separate from app.py keeps the Streamlit file
focused on UI only.
"""

import io
import json
import logging
from typing import Tuple

import pandas as pd

from models import ResumeData

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = (".pdf",)


def setup_logging() -> logging.Logger:
    """Configure root logging once for the whole app."""
    # ---------------------------------------------------------------------
    # Configure root logger format/level once, then return a named logger
    # ---------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("resume_extractor")


def validate_uploaded_file(filename: str, size_bytes: int) -> None:
    """
    Validate that the uploaded file is a PDF and within the size limit.

    Raises:
        ValueError: if the file type or size is invalid.
    """
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        # ---------------------------------------------------------------
        # Reject non-PDF file extensions
        # ---------------------------------------------------------------
        raise ValueError(f"Unsupported file type. Please upload a PDF file (got: {filename}).")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        # ---------------------------------------------------------------
        # Reject files larger than the configured max upload size
        # ---------------------------------------------------------------
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB."
        )


def resume_to_json_bytes(data: ResumeData) -> bytes:
    """Serialize ResumeData to pretty-printed JSON bytes for download."""
    # ---------------------------------------------------------------------
    # Dump Pydantic model -> dict -> pretty JSON -> UTF-8 bytes
    # ---------------------------------------------------------------------
    return json.dumps(data.model_dump(), indent=2, ensure_ascii=False).encode("utf-8")


def resume_to_csv_bytes(data: ResumeData) -> bytes:
    """
    Flatten ResumeData into a single-row CSV (simple fields) for
    download. Nested lists (education, experience, etc.) are joined
    into semicolon-separated strings so they fit in one cell.
    """
    # ---------------------------------------------------------------------
    # Convert model to plain dict and pull out top-level sub-sections
    # ---------------------------------------------------------------------
    d = data.model_dump()
    personal = d.get("personal_information", {})
    professional = d.get("professional_information", {})
    skills = d.get("skills", {})

    def join_list(items):
        # -------------------------------------------------------------
        # Join a simple list of strings with "; " (empty -> "")
        # -------------------------------------------------------------
        return "; ".join(str(i) for i in items) if items else ""

    def join_records(records, fields):
        # -------------------------------------------------------------
        # Flatten a list of dict records into one "||"-joined string,
        # with each record's chosen fields joined by " | "
        # -------------------------------------------------------------
        parts = []
        for r in records:
            parts.append(" | ".join(f"{f}: {r.get(f, '')}" for f in fields))
        return " || ".join(parts)

    # ---------------------------------------------------------------------
    # Build the single flattened CSV row combining all sections
    # ---------------------------------------------------------------------
    row = {
        **personal,
        **professional,
        "professional_summary": d.get("professional_summary"),
        "programming_languages": join_list(skills.get("programming_languages")),
        "frameworks": join_list(skills.get("frameworks")),
        "libraries": join_list(skills.get("libraries")),
        "databases": join_list(skills.get("databases")),
        "cloud_platforms": join_list(skills.get("cloud_platforms")),
        "devops_tools": join_list(skills.get("devops_tools")),
        "ai_tools": join_list(skills.get("ai_tools")),
        "soft_skills": join_list(skills.get("soft_skills")),
        "education": join_records(d.get("education", []),
                                   ["degree", "field", "university", "start_year", "end_year", "gpa"]),
        "experience": join_records(d.get("experience", []),
                                    ["company", "position", "location", "start_date", "end_date"]),
        "projects": join_records(d.get("projects", []),
                                  ["name", "description", "github_link", "live_demo"]),
        "certifications": join_list(d.get("certifications")),
        "languages": join_list(d.get("languages")),
        "achievements": join_list(d.get("achievements")),
        "internships": join_list(d.get("internships")),
        "awards": join_list(d.get("awards")),
        "volunteer_experience": join_list(d.get("volunteer_experience")),
    }

    # ---------------------------------------------------------------------
    # Wrap the single row in a DataFrame and export to CSV bytes
    # ---------------------------------------------------------------------
    df = pd.DataFrame([row])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def resume_to_text_report(data: ResumeData) -> bytes:
    """Build a nicely formatted, human-readable plain-text report."""
    d = data.model_dump()
    lines = []

    def section(title):
        # -------------------------------------------------------------
        # Append a blank line, section title, and underline of dashes
        # -------------------------------------------------------------
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    def bullet_list(items):
        # -------------------------------------------------------------
        # Append each item as a "- item" bullet, or a placeholder if empty
        # -------------------------------------------------------------
        if not items:
            lines.append("  (none listed)")
        for item in items:
            lines.append(f"  - {item}")

    # ---------------------------------------------------------------------
    # Report header
    # ---------------------------------------------------------------------
    lines.append("=" * 60)
    lines.append("RESUME EXTRACTION REPORT")
    lines.append("=" * 60)

    # ---------------------------------------------------------------------
    # Personal information section
    # ---------------------------------------------------------------------
    section("Personal Information")
    for k, v in d.get("personal_information", {}).items():
        lines.append(f"  {k.replace('_', ' ').title()}: {v or 'N/A'}")

    # ---------------------------------------------------------------------
    # Professional information section
    # ---------------------------------------------------------------------
    section("Professional Information")
    for k, v in d.get("professional_information", {}).items():
        lines.append(f"  {k.replace('_', ' ').title()}: {v or 'N/A'}")

    # ---------------------------------------------------------------------
    # Professional summary section
    # ---------------------------------------------------------------------
    section("Professional Summary")
    lines.append(f"  {d.get('professional_summary') or 'N/A'}")

    # ---------------------------------------------------------------------
    # Skills section (each category listed as comma-separated values)
    # ---------------------------------------------------------------------
    section("Skills")
    skills = d.get("skills", {})
    for k, v in skills.items():
        lines.append(f"  {k.replace('_', ' ').title()}: {', '.join(v) if v else 'None'}")

    # ---------------------------------------------------------------------
    # Education section
    # ---------------------------------------------------------------------
    section("Education")
    if not d.get("education"):
        lines.append("  (none listed)")
    for edu in d.get("education", []):
        lines.append(
            f"  - {edu.get('degree', 'N/A')} in {edu.get('field', 'N/A')}, "
            f"{edu.get('university', 'N/A')} "
            f"({edu.get('start_year', '?')} - {edu.get('end_year', '?')}) "
            f"GPA: {edu.get('gpa', 'N/A')}"
        )

    # ---------------------------------------------------------------------
    # Experience section (including nested responsibilities bullets)
    # ---------------------------------------------------------------------
    section("Experience")
    if not d.get("experience"):
        lines.append("  (none listed)")
    for exp in d.get("experience", []):
        lines.append(
            f"  - {exp.get('position', 'N/A')} at {exp.get('company', 'N/A')} "
            f"({exp.get('start_date', '?')} - {exp.get('end_date', '?')}), "
            f"{exp.get('location', 'N/A')}"
        )
        for r in exp.get("responsibilities", []):
            lines.append(f"      * {r}")

    # ---------------------------------------------------------------------
    # Projects section
    # ---------------------------------------------------------------------
    section("Projects")
    if not d.get("projects"):
        lines.append("  (none listed)")
    for proj in d.get("projects", []):
        lines.append(f"  - {proj.get('name', 'N/A')}: {proj.get('description', 'N/A')}")

    # ---------------------------------------------------------------------
    # Remaining simple list sections (certifications, languages, etc.)
    # ---------------------------------------------------------------------
    for key in ("certifications", "languages", "achievements", "internships", "awards", "volunteer_experience"):
        section(key.replace("_", " ").title())
        bullet_list(d.get(key, []))

    # ---------------------------------------------------------------------
    # Join all lines into the final text report and encode to bytes
    # ---------------------------------------------------------------------
    return "\n".join(lines).encode("utf-8")
