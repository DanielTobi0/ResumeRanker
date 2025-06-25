import math

from openai import OpenAI

from src.ranking.llm_judge import llm_as_a_judge
from src.ranking.text_formatting import format_job_description, format_resume
from src.utils.helpers import safe_get_nested


def score_and_rank(
    client: OpenAI,
    bi_encoder_ranking: list[dict],
    structured_resumes: list[dict],
    structured_job_description: dict,
    cross_encoder_model,
    top_n: int = 3,
    llm_weight: float = 0.7,
    cross_encoder_weight: float = 0.3,
) -> list[dict]:
    """
    Score and rank candidates using a combination of cross-encoder and LLM evaluation.

    This function takes the top N candidates from a bi-encoder ranking and performs
    a more detailed analysis using:
    1. An LLM judge to evaluate detailed matching against job requirements
    2. A cross-encoder model for direct pairwise similarity scoring

    The final ranking combines both scores with specified weights.

    Args:
        client (OpenAI): An initialized OpenAI client
        bi_encoder_ranking (list[dict]): Initial ranking from bi-encoder
        structured_resumes (list[dict]): List of structured resume data
        structured_job_description (dict): Structured job requirements
        cross_encoder_model: The cross-encoder model for pairwise scoring
        top_n (int): Number of top candidates to analyze in detail (default: 3)
        llm_weight (float): Weight for LLM score in final ranking (default: 0.7)
        cross_encoder_weight (float): Weight for cross-encoder score (default: 0.3)

    Returns:
        list[dict]: A list of dictionaries with candidate scores and analysis,
                   sorted in descending order by combined score
    """
    top_candidates = bi_encoder_ranking[:top_n]
    top_candidate_names = [c["candidate_name"] for c in top_candidates]

    candidate_data_by_name = {
        safe_get_nested(c, "contact_info", "name"): c for c in structured_resumes
    }

    job_text = format_job_description(structured_job_description)

    llm_results = []
    cross_encoder_scores = []

    for name in top_candidate_names:
        if name in candidate_data_by_name:
            candidate_data = candidate_data_by_name[name]

            # LLM Judge
            llm_judgement = llm_as_a_judge(
                client, structured_job_description, candidate_data
            )
            llm_results.append({"candidate_name": name, **llm_judgement})

            # Cross-encoder
            resume_text = format_resume(candidate_data)
            raw_score = float(cross_encoder_model.predict([(job_text, resume_text)])[0])

            # apply sigmoid to get a score between 0 and 1, then scale to 10
            score = (1 / (1 + math.exp(-raw_score))) * 10.0
            cross_encoder_scores.append(
                {"candidate_name": name, "cross_encoder_score": score}
            )

    combined_results = []
    for name in top_candidate_names:
        llm_result = next((r for r in llm_results if r["candidate_name"] == name), {})
        cross_result = next(
            (s for s in cross_encoder_scores if s["candidate_name"] == name), {}
        )

        llm_score = llm_result.get("final_score", 0)
        cross_score = cross_result.get("cross_encoder_score", 0)

        combined_score = (llm_score * llm_weight) + (cross_score * cross_encoder_weight)

        combined_results.append(
            {
                "candidate_name": name,
                "llm_score": llm_score,
                "cross_encoder_score": cross_score,
                "combined_score": combined_score,
                "llm_analysis": llm_result,
            }
        )

    return sorted(combined_results, key=lambda x: x["combined_score"], reverse=True)
