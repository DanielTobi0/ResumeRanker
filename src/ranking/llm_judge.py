import json
import logging

from openai import OpenAI

from src.models.schema import LLMJudgment
from src.ranking.prompts import judge_prompt_template


def llm_as_a_judge(
    client: OpenAI,
    job_requirements: dict,
    resume_info: dict,
) -> dict:
    """
    Use an LLM to evaluate how well a resume matches job requirements.

    This function formats job requirements and resume data for the LLM,
    then prompts it to perform a detailed analysis and provide scores
    and insights according to the LLMJudgment schema.

    Args:
        client (OpenAI): An initialized OpenAI client
        job_requirements (dict): Structured job requirements data
        resume_info (dict): Structured resume data to evaluate

    Returns:
        dict: A dictionary containing detailed evaluation results conforming to
              the LLMJudgment schema. Returns an empty dict if evaluation fails.

    Note:
        The evaluation includes detailed analysis, pros/cons lists, a numerical score,
        and specific criteria matching assessments.
    """
    formatted_prompt = judge_prompt_template.format(
        job_requirements=json.dumps(job_requirements, indent=2),
        resume_info=json.dumps(resume_info, indent=2),
    )
    try:
        response = client.responses.parse(
            model="gpt-4.1-nano",
            input=[
                {"role": "system", "content": "You are a strict resume ranker."},
                {"role": "user", "content": formatted_prompt},
            ],
            text_format=LLMJudgment,
        )
        return json.loads(response.output_text)
    except Exception as e:
        logging.error(f"LLM judgement failed: {e}")
        return {}
