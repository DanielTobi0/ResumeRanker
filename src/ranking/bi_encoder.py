from sentence_transformers import util

from src.ranking.text_formatting import format_job_description, format_resume
from src.utils.helpers import safe_get_nested


def bi_encoder_resume_filtering(
    structured_job_description: dict,
    structured_candidate_resumes: list[dict],
    bi_encoder_model,
) -> list[dict]:
    """
    Filter and rank resumes based on similarity to job description using a bi-encoder model.

    This function compares each resume against the job description by:
    1. Converting both job description and resume to text format
    2. Generating embeddings for both texts using the bi-encoder model
    3. Calculating cosine similarity between the embeddings
    4. Ranking candidates based on similarity scores

    Args:
        structured_job_description (dict): The structured job description data
        structured_candidate_resumes (list[dict]): A list of structured candidate resume data
        bi_encoder_model: The SentenceTransformer model to use for generating embeddings

    Returns:
        list[dict]: A list of dictionaries containing candidate names and their similarity
                   scores, sorted in descending order by score
    """
    job_des_text = format_job_description(structured_job_description)

    results = []
    for candidate_data in structured_candidate_resumes:
        resume_text = format_resume(candidate_data)

        embeddings = bi_encoder_model.encode(
            [job_des_text, resume_text], convert_to_tensor=True
        )
        score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

        candidate_name = safe_get_nested(
            candidate_data, "contact_info", "name", default="Unknown Candidate"
        )
        results.append({"candidate_name": candidate_name, "rank": score})

    return sorted(results, key=lambda x: x["rank"], reverse=True)
