from src.ranking.prompts import job_signal_text, resume_signal_text
from src.utils.helpers import safe_get_nested, safe_join


def format_job_description(structured_job_description: dict) -> str:
    """
    Format a structured job description into a standardized text representation.

    This function extracts key information from the structured job description
    and arranges it into a consistent text format using the job_signal_text template.
    This standardization helps with consistent embedding generation and comparison.

    Args:
        structured_job_description (dict): The structured job description data

    Returns:
        str: A formatted text representation of the job description
    """
    job_context = safe_get_nested(structured_job_description, "job_context", default={})
    role_description = safe_get_nested(
        structured_job_description, "role_description", default={}
    )
    hard_requirements = safe_get_nested(
        structured_job_description, "hard_requirements", default={}
    )
    preferred_quals = safe_get_nested(
        structured_job_description, "preferred_qualifications", default={}
    )

    return job_signal_text.format(
        job_title=safe_get_nested(job_context, "job_title", default=""),
        seniority=safe_get_nested(job_context, "seniority_level", default=""),
        key_responsibilities=safe_join(
            safe_get_nested(role_description, "key_responsibilities", default=[])
        ),
        required_skills=safe_join(
            safe_get_nested(hard_requirements, "must_have_skills", default=[])
        ),
        required_tools=safe_join(
            safe_get_nested(hard_requirements, "must_have_tools", default=[])
        ),
        preferred_skills=safe_join(
            safe_get_nested(preferred_quals, "preferred_skills", default=[])
        ),
        preferred_tools=safe_join(
            safe_get_nested(preferred_quals, "preferred_tools", default=[])
        ),
        required_experience=safe_join(
            safe_get_nested(hard_requirements, "required_experience_type", default=[])
        ),
        domain_knowledge=safe_join(
            safe_get_nested(role_description, "domain_knowledge", default=[])
        ),
    )


def format_resume(candidate_data: dict) -> str:
    """
    Format a structured resume into a standardized text representation.

    This function extracts key information from the structured resume data
    and arranges it into a consistent text format using the resume_signal_text template.
    The resulting text focuses on the most relevant aspects for job matching.

    Args:
        candidate_data (dict): The structured resume data

    Returns:
        str: A formatted text representation of the resume
    """
    skills_section = safe_get_nested(candidate_data, "skills_section", default={})
    all_skills = (
        safe_get_nested(skills_section, "programming_languages", default=[])
        + safe_get_nested(skills_section, "frameworks_and_libraries", default=[])
        + safe_get_nested(skills_section, "platforms_and_tools", default=[])
    )

    work_experience = safe_get_nested(candidate_data, "work_experience", default=[])
    job_roles = []
    for job in work_experience:
        company = safe_get_nested(job, "company", default="")
        role = safe_get_nested(job, "job_title", default="")
        achievements = safe_get_nested(job, "achievements", default=[])
        job_roles.append(f"{role} at {company}: {' '.join(achievements)}")

    projects = safe_get_nested(candidate_data, "projects", default=[])
    project_texts = []
    for proj in projects:
        name = safe_get_nested(proj, "project_name", default="")
        desc = safe_get_nested(proj, "description", default="")
        project_texts.append(f"{name}: {desc}")

    return resume_signal_text.format(
        professional_summary=safe_get_nested(
            candidate_data, "professional_summary", default=""
        ),
        skills=safe_join(all_skills),
        job_roles_and_achivements=safe_join(job_roles),
        projects=safe_join(project_texts),
        certificates=safe_join(
            safe_get_nested(candidate_data, "certifications", default=[])
        ),
    )
