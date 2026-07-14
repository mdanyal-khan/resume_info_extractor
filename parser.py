"""
parser.py
---------
Builds the prompt sent to Qwen2.5-7B-Instruct and turns its raw text
response into a validated `ResumeData` Pydantic object.

Since Qwen2.5-7B-Instruct doesn't have guaranteed "native" structured
output (unlike some OpenAI models), we use LangChain's
`PydanticOutputParser`, which:
  1. Generates format instructions describing the exact JSON shape.
  2. Parses the model's raw text output into JSON.
  3. Validates it against our Pydantic schema.

We also add a small amount of manual cleanup + a retry step, because
open-source models sometimes wrap JSON in markdown fences or add stray
text even when told not to.
"""

import json
import logging
import re
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from models import ResumeData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Parser setup
# ---------------------------------------------------------------------
# The parser knows how to turn model output into a ResumeData object,
# and can generate "format_instructions" describing the schema to the LLM.
output_parser = PydanticOutputParser(pydantic_object=ResumeData)

# ---------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------
# Strict instructions to keep the LLM's output as close to pure JSON as
# possible, since it must later be parsed and validated automatically.
EXTRACTION_PROMPT_TEMPLATE = """You are a precise resume-parsing engine. Your ONLY job is to
read the resume text below and extract information into a single valid JSON object.

STRICT RULES (follow all of them exactly):
1. Extract ONLY information that is explicitly present in the resume text.
2. NEVER guess, infer, or invent missing values.
3. If a single value (like email or phone) is missing, set it to null.
4. If a list (like skills or education) has no items, return an empty list [].
5. Return ONLY valid JSON. Do not include Markdown code fences (no ```).
6. Do NOT include any explanations, notes, comments, or extra text before or after the JSON.
7. Do NOT add any keys that are not in the schema below.
8. Every key in your JSON output MUST exactly match the schema's key names.
9. The JSON you return must be directly parseable by a JSON parser with no post-processing.

SCHEMA (the exact structure your JSON output must follow):
{format_instructions}

RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"

Return only the JSON object now:"""


def build_extraction_prompt(resume_text: str) -> str:
    """Fill in the prompt template with format instructions + resume text."""
    # ---------------------------------------------------------------------
    # Build prompt object
    # ---------------------------------------------------------------------
    # `partial_variables` pre-fills the schema description so callers only
    # need to supply the resume text at format-time.
    prompt = PromptTemplate(
        template=EXTRACTION_PROMPT_TEMPLATE,
        input_variables=["resume_text"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    # ---------------------------------------------------------------------
    # Render final prompt string
    # ---------------------------------------------------------------------
    return prompt.format(resume_text=resume_text)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if the model added them anyway."""
    # ---------------------------------------------------------------------
    # Trim whitespace, then strip leading ```/```json and trailing ```
    # ---------------------------------------------------------------------
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_block(text: str) -> str:
    """
    Extract the first {...} JSON object found in the text, in case the
    model added stray text before/after the JSON despite instructions.
    """
    # ---------------------------------------------------------------------
    # Greedy regex match of the outermost { ... } block
    # ---------------------------------------------------------------------
    # Falls back to returning the original text unchanged if no JSON
    # object-like structure is found.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def parse_llm_output(raw_output: str) -> ResumeData:
    """
    Convert the LLM's raw text output into a validated ResumeData object.

    Raises:
        ValueError: If the output cannot be parsed/validated even after cleanup.
    """
    # ---------------------------------------------------------------------
    # Cleanup pass: remove markdown fences, isolate the JSON object
    # ---------------------------------------------------------------------
    cleaned = _strip_markdown_fences(raw_output)
    cleaned = _extract_json_block(cleaned)

    try:
        # -------------------------------------------------------------
        # Primary parse path: LangChain's PydanticOutputParser
        # -------------------------------------------------------------
        return output_parser.parse(cleaned)
    except (ValidationError, json.JSONDecodeError, Exception) as first_err:  # noqa: BLE001
        # -------------------------------------------------------------
        # Primary parse failed — log and fall through to manual repair
        # -------------------------------------------------------------
        logger.warning("Initial parse failed (%s). Attempting manual JSON repair.", first_err)

        # Fallback: try plain json.loads + manual Pydantic validation,
        # which sometimes succeeds even when LangChain's parser is strict.
        try:
            # ---------------------------------------------------------
            # Fallback parse path: raw json.loads + manual validation
            # ---------------------------------------------------------
            data = json.loads(cleaned)
            return ResumeData.model_validate(data)
        except Exception as second_err:  # noqa: BLE001
            # -----------------------------------------------------------
            # Both parse attempts failed — raise a clear, wrapped error
            # -----------------------------------------------------------
            logger.error("Manual JSON repair also failed: %s", second_err)
            raise ValueError(
                "Could not parse the model's output into valid structured data. "
                f"Original error: {first_err}"
            ) from second_err


def extract_resume_data(llm, resume_text: str, max_retries: int = 1) -> ResumeData:
    """
    Full pipeline: build prompt -> call LLM -> parse+validate output.
    Retries once (by default) if parsing fails, since LLM sampling can
    occasionally produce malformed JSON.

    Args:
        llm: A LangChain-compatible LLM object with an `.invoke()` method.
        resume_text: The extracted plain text of the resume.
        max_retries: How many times to retry on parse failure.

    Returns:
        A validated ResumeData object.
    """
    # ---------------------------------------------------------------------
    # Build the prompt once; reused across retries
    # ---------------------------------------------------------------------
    prompt = build_extraction_prompt(resume_text)

    last_error: Optional[Exception] = None
    # ---------------------------------------------------------------------
    # Retry loop: attempt LLM call + parse up to (max_retries + 1) times
    # ---------------------------------------------------------------------
    for attempt in range(max_retries + 1):
        logger.info("LLM extraction attempt %d/%d", attempt + 1, max_retries + 1)
        try:
            # -------------------------------------------------------------
            # Call the LLM with the built prompt
            # -------------------------------------------------------------
            raw_output = llm.invoke(prompt)
            # HuggingFacePipeline may return a string directly, or a dict/list
            # depending on backend; normalize to string.
            if not isinstance(raw_output, str):
                # ---------------------------------------------------------
                # Normalize non-string outputs to string before parsing
                # ---------------------------------------------------------
                raw_output = str(raw_output)
            # -------------------------------------------------------------
            # Parse + validate; success returns immediately from the loop
            # -------------------------------------------------------------
            return parse_llm_output(raw_output)
        except Exception as err:  # noqa: BLE001
            # -----------------------------------------------------------
            # Record failure and continue to next retry attempt (if any)
            # -----------------------------------------------------------
            last_error = err
            logger.warning("Attempt %d failed: %s", attempt + 1, err)

    # ---------------------------------------------------------------------
    # All attempts exhausted — raise with the last recorded error
    # ---------------------------------------------------------------------
    raise ValueError(f"Failed to extract structured data after retries: {last_error}")