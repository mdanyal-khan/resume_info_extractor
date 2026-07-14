"""
extractor.py
------------
High-level orchestration layer: given an uploaded file object, this
module runs the full pipeline (PDF -> text -> LLM -> validated data)
and returns everything the UI needs. This keeps app.py thin and free
of business logic.
"""

import logging
from dataclasses import dataclass
from typing import BinaryIO

from pdf_reader import extract_text_from_pdf, PDFReadError
from parser import extract_resume_data
from models import ResumeData

logger = logging.getLogger(__name__)


@dataclass
class ExtractionOutcome:
    """Everything the UI needs after running the pipeline."""
    resume_text: str
    num_pages: int
    filename: str
    data: ResumeData


def run_extraction_pipeline(llm, file_obj: BinaryIO, filename: str) -> ExtractionOutcome:
    """
    Run the full resume-extraction pipeline.

    Args:
        llm: A loaded LangChain-compatible LLM (see llm.py).
        file_obj: The uploaded PDF file-like object.
        filename: Original filename (for display/logging).

    Returns:
        ExtractionOutcome with raw text, page count, and validated data.

    Raises:
        PDFReadError: If the PDF can't be read.
        ValueError: If the LLM output can't be parsed/validated.
    """
    logger.info("Pipeline started for '%s'", filename)

    # ---------------------------------------------------------------------
    # Step 1: PDF -> plain text
    # ---------------------------------------------------------------------
    pdf_result = extract_text_from_pdf(file_obj, filename)
    logger.info("PDF text extracted: %d pages, %d chars", pdf_result.num_pages, len(pdf_result.text))

    # ---------------------------------------------------------------------
    # Step 2: plain text -> LLM -> validated ResumeData
    # ---------------------------------------------------------------------
    data = extract_resume_data(llm, pdf_result.text)
    logger.info("Structured data validated successfully for '%s'", filename)

    # ---------------------------------------------------------------------
    # Step 3: bundle everything the UI needs into one return object
    # ---------------------------------------------------------------------
    return ExtractionOutcome(
        resume_text=pdf_result.text,
        num_pages=pdf_result.num_pages,
        filename=filename,
        data=data,
    )
