import json
import logging

from openai import OpenAI

from src.models.schema import ExtractedJobRequirements


def extract_job_requirements(client: OpenAI, job_description: str) -> dict:
    """
    Extract structured requirements from a job description using the OpenAI API.

    This function analyzes raw job description text and identifies key requirements,
    organizing them according to the ExtractedJobRequirements schema. The extraction
    categorizes information into job context, role description, hard requirements,
    and preferred qualifications.

    Args:
        client (OpenAI): An initialized OpenAI client instance
        job_description (str): The raw text of the job description to be analyzed

    Returns:
        dict: A dictionary containing the structured job requirements matching the
              ExtractedJobRequirements schema. Returns an empty dict if extraction fails.

    Note:
        If an error occurs during extraction, the error is printed to stdout and
        an empty dictionary is returned.
    """
    job_extraction_system_prompt = """
    You are a job description analyzer. 
    Extract the key requirements from this job description into these categories:
    Make sure to expand all abbreviations.
    """
    try:
        response = client.responses.parse(
            model="gpt-4.1-nano",
            input=[
                {"role": "system", "content": job_extraction_system_prompt},
                {"role": "user", "content": job_description},
            ],
            text_format=ExtractedJobRequirements,
        )
        result = json.loads(response.output_text)
        return result
    except Exception as e:
        logging.error(f"Failed to extract job requirements: {e}")
        return {}
