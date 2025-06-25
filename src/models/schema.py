"""
Schema definitions for the resume matching system.

This module defines the data models used throughout the application for:
- Structured job descriptions and requirements
- Parsed resume information
- Matching and evaluation results

These Pydantic models ensure type safety and provide validation for data
flowing through the application pipeline.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

# Job Description Models


class JobContext(BaseModel):
    job_title: str = Field(..., description="The title of the job.")
    seniority_level: str = Field(
        ..., description="The seniority level of the job (e.g., 'Senior', 'Junior')."
    )


class RoleDescription(BaseModel):
    key_responsibilities: List[str] = Field(
        ..., description="The key responsibilities for the role."
    )
    domain_knowledge: List[str] = Field(
        ..., description="The domain knowledge required for the role."
    )


class HardRequirements(BaseModel):
    must_have_skills: List[str] = Field(
        ..., description="The must-have skills for the job."
    )
    must_have_tools: List[str] = Field(
        ..., description="The must-have tools for the job."
    )
    required_experience_type: List[str] = Field(
        ...,
        description="The required type of experience (e.g., 'SaaS environment', 'startup experience').",
    )
    minimum_experience_years: int = Field(
        ..., description="The minimum years of experience required."
    )
    required_education: List[str] = Field(
        ..., description="The required educational qualifications."
    )


class PreferredQualifications(BaseModel):
    preferred_skills: List[str] = Field(
        ..., description="The preferred skills for the job."
    )
    preferred_tools: List[str] = Field(
        ..., description="The preferred tools for the job."
    )
    preferred_certifications: List[str] = Field(
        ..., description="The preferred certifications for the job."
    )


class ExtractedJobRequirements(BaseModel):
    """
    A structured representation of the job requirements extracted from a job description.

    This model organizes job requirements into logical categories to facilitate
    matching and comparison with candidate resumes.
    """

    job_context: JobContext = Field(
        ..., description="The context of the job, including title and seniority."
    )
    role_description: RoleDescription = Field(
        ..., description="The description of the role and its responsibilities."
    )
    hard_requirements: HardRequirements = Field(
        ..., description="The mandatory requirements for the job."
    )
    preferred_qualifications: PreferredQualifications = Field(
        ..., description="The preferred qualifications for the job."
    )


# Resume Models


class ContactInfo(BaseModel):
    name: str = Field(..., description="The candidate's full name.")
    email: Optional[str] = Field(None, description="The candidate's email address.")
    phone: Optional[str] = Field(None, description="The candidate's phone number.")
    linkedin: Optional[str] = Field(
        None, description="The candidate's LinkedIn profile URL."
    )


class Education(BaseModel):
    institution: str = Field(
        ..., description="The name of the educational institution."
    )
    degree: str = Field(..., description="The degree obtained.")
    field_of_study: str = Field(..., description="The field of study.")
    graduation_year: Optional[int] = Field(None, description="The year of graduation.")


class SkillSection(BaseModel):
    programming_languages: List[str] = Field(
        ..., description="List of programming languages."
    )
    frameworks_and_libraries: List[str] = Field(
        ..., description="List of frameworks and libraries."
    )
    platforms_and_tools: List[str] = Field(
        ..., description="List of platforms and tools."
    )


class WorkExperience(BaseModel):
    company: str = Field(..., description="The name of the company.")
    job_title: str = Field(..., description="The job title.")
    start_date: str = Field(..., description="The start date of the employment.")
    end_date: str = Field(..., description="The end date of the employment.")
    achievements: List[str] = Field(
        ..., description="List of key achievements and responsibilities."
    )
    used_skills_and_tools: List[str] = Field(
        ..., description="Skills and tools used in this role."
    )


class Project(BaseModel):
    project_name: str = Field(..., description="The name of the project.")
    description: str = Field(..., description="A brief description of the project.")
    technologies_used: List[str] = Field(
        ..., description="Technologies used in the project."
    )


class StructuredResume(BaseModel):
    """
    A structured representation of a candidate's resume.

    This model organizes resume information into standardized sections for
    consistent processing and comparison with job requirements.
    """

    contact_info: ContactInfo = Field(
        ..., description="The candidate's contact information."
    )
    professional_summary: str = Field(
        ..., description="The professional summary or objective."
    )
    skills_section: SkillSection = Field(..., description="The skills section.")
    work_experience: List[WorkExperience] = Field(
        ..., description="A list of work experiences."
    )
    projects: List[Project] = Field(..., description="A list of personal projects.")
    education: List[Education] = Field(
        ..., description="A list of educational qualifications."
    )
    certifications: List[str] = Field(..., description="A list of certifications.")


# Matching and Evaluation Models


class MatchCriterion(BaseModel):
    criterion: str = Field(
        ...,
        description="The criterion being evaluated (e.g., 'Minimum Experience', 'Python Skill').",
    )
    is_match: bool = Field(
        ..., description="Whether the candidate meets this criterion."
    )
    comment: str = Field(
        ..., description="A brief explanation of why it is or isn't a match."
    )


class LLMJudgment(BaseModel):
    """
    Evaluation results from the LLM comparing a resume against job requirements.
    """

    detailed_analysis: str = Field(
        ...,
        description="A detailed, step-by-step analysis of how the resume aligns with the job requirements.",
    )
    pros: List[str] = Field(
        ..., description="A list of the candidate's key strengths for this role."
    )
    cons: List[str] = Field(
        ...,
        description="A list of the candidate's key weaknesses or gaps for this role.",
    )
    final_score: float = Field(
        ..., ge=0, le=10, description="The final numerical score from 0 to 10."
    )
    match_criteria: List[MatchCriterion] = Field(
        ..., description="A list of specific criteria matches."
    )
