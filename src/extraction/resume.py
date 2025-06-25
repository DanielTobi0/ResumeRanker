import json
import logging

from openai import OpenAI

from src.models.schema import StructuredResume


def extract_resume_details(client: OpenAI, resume: str) -> dict:
    """
    Extract structured information from a resume using the OpenAI API.

    This function parses raw resume text and extracts key information into a structured format
    defined by the StructuredResume schema. It uses OpenAI's parsing capabilities to
    categorize resume content including contact information, skills, work experience,
    education, and certifications.

    Args:
        client (OpenAI): An initialized OpenAI client instance
        resume (str): The raw text of the resume to be analyzed

    Returns:
        dict: A dictionary containing the structured resume information matching the
              StructuredResume schema. Returns an empty dict if extraction fails.

    Note:
        If an error occurs during extraction, the error is printed to stdout and
        an empty dictionary is returned.
    """
    resume_details_extraction_prompt = """
    You are a resume analyzer. 
    Extract the key infomation into these categories:
    Make sure to expand all abbreviations.
    Set unavailable information as None
    """
    try:
        response = client.responses.parse(
            model="gpt-4.1-nano",
            input=[
                {"role": "system", "content": resume_details_extraction_prompt},
                {"role": "user", "content": resume},
            ],
            text_format=StructuredResume,
        )
        return json.loads(response.output_text)
    except Exception as e:
        logging.error(f"Failed to extract resume details: {e}")
        return {}
