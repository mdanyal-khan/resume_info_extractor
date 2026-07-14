"""
pdf_reader.py
-------------
Handles everything related to reading a PDF resume and turning it into
plain text. Uses pdfplumber, which is good at pulling text out of
PDFs page by page.

Beginner note: PDFs can be tricky -- some are scanned images (no real
text layer), some are encrypted/password protected, and some are just
corrupted files. This module tries to handle those cases gracefully
and always raises a clear, human-readable error instead of crashing.
"""

import logging
from dataclasses import dataclass
from typing import BinaryIO

import pdfplumber

logger = logging.getLogger(__name__)


class PDFReadError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


@dataclass
class PDFExtractionResult:
    """Container for the result of reading a PDF."""
    text: str
    num_pages: int
    filename: str


def extract_text_from_pdf(file_obj: BinaryIO, filename: str) -> PDFExtractionResult:
    """
    Extract all text from a PDF file, page by page, and combine it into
    a single string.

    Args:
        file_obj: A file-like object (e.g. from Streamlit's file_uploader).
        filename: Original filename, used for logging/error messages.

    Returns:
        PDFExtractionResult with the combined text, page count, and filename.

    Raises:
        PDFReadError: If the PDF is encrypted, corrupted, or has no text.
    """
    logger.info("Starting PDF extraction for file: %s", filename)

    try:
        # ---------------------------------------------------------------
        # Open the PDF and iterate over every page
        # ---------------------------------------------------------------
        with pdfplumber.open(file_obj) as pdf:
            num_pages = len(pdf.pages)

            if num_pages == 0:
                # -------------------------------------------------------
                # Guard: reject PDFs with zero pages
                # -------------------------------------------------------
                raise PDFReadError(f"'{filename}' has no pages.")

            page_texts = []
            # -----------------------------------------------------------
            # Loop over each page and extract its text individually
            # -----------------------------------------------------------
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    # ---------------------------------------------------
                    # Attempt text extraction for this page
                    # ---------------------------------------------------
                    page_text = page.extract_text() or ""
                except Exception as page_err:  # noqa: BLE001
                    # A single bad page shouldn't kill the whole extraction.
                    # -----------------------------------------------------
                    # Log and treat this page as empty rather than aborting
                    # -----------------------------------------------------
                    logger.warning(
                        "Failed to extract text from page %s of '%s': %s",
                        page_number, filename, page_err,
                    )
                    page_text = ""
                page_texts.append(page_text)

            # -----------------------------------------------------------
            # Join all page texts into one combined string
            # -----------------------------------------------------------
            combined_text = "\n\n".join(page_texts).strip()

    except pdfplumber.pdfminer.pdfdocument.PDFEncryptionError as enc_err:
        # -------------------------------------------------------------------
        # Handle password-protected/encrypted PDFs with a clear message
        # -------------------------------------------------------------------
        logger.error("Encrypted PDF error for '%s': %s", filename, enc_err)
        raise PDFReadError(
            f"'{filename}' is password-protected/encrypted. "
            "Please upload an unlocked PDF."
        ) from enc_err
    except PDFReadError:
        # -------------------------------------------------------------------
        # Re-raise our own error type unchanged (already has a clear message)
        # -------------------------------------------------------------------
        raise
    except Exception as err:  # noqa: BLE001
        # -------------------------------------------------------------------
        # Catch-all for corrupted/unreadable files -> wrap in PDFReadError
        # -------------------------------------------------------------------
        logger.error("Failed to open/read PDF '%s': %s", filename, err)
        raise PDFReadError(
            f"Could not read '{filename}'. The file may be corrupted, "
            "not a valid PDF, or password protected."
        ) from err

    if not combined_text:
        # ---------------------------------------------------------------
        # Guard: reject PDFs that yielded no text at all (likely scanned)
        # ---------------------------------------------------------------
        raise PDFReadError(
            f"No extractable text found in '{filename}'. "
            "It might be a scanned image without a text layer -- "
            "try a text-based PDF instead."
        )

    logger.info(
        "Successfully extracted text from '%s' (%d pages, %d characters)",
        filename, num_pages, len(combined_text),
    )

    # ---------------------------------------------------------------------
    # Success: package results into the return dataclass
    # ---------------------------------------------------------------------
    return PDFExtractionResult(text=combined_text, num_pages=num_pages, filename=filename)
