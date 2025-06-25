resume_signal_text = """
Professional Summary: 
{professional_summary}

Skills: 
{skills}

Job Roles and Achievements: 
{job_roles_and_achivements}

Projects:
{projects}

Certifications:
{certificates}
"""

job_signal_text = """
Job Title: 
{job_title}

Seniority: 
{seniority}

Key Responsibilities: 
{key_responsibilities}

Required Skills:
{required_skills}

Required Tools:
{required_tools}

Preferred Skills: 
{preferred_skills}

Preferred Tools:
{preferred_tools}

Required Experience:
{required_experience}

Domain Knowledge:
{domain_knowledge}
"""

judge_prompt_template = """ 
You are an expert technical recruiter and resume analyst. Your task is to act as an impartial judge and evaluate a candidate's resume against a specific job description.

You will be given two JSON objects:
1. `JOB_REQUIREMENTS`: The extracted requirements from the job description.
2. `CANDIDATE_RESUME`: The parsed, structured information from the candidate's resume.

Your goal is to perform a detailed comparison and provide a final score from 0 to 10, where 10 is a perfect match.

Follow these steps precisely:

1.  Analyze Experience:
       Compare the `minimum_experience_years` from the job with the candidate's `work_experience` timeline.
       Assess if the `required_experience_type` (e.g., 'SaaS environment') is present in the candidate's job descriptions.

2.  Analyze Hard Skills and Tools:
       For each skill in `must_have_skills` and `must_have_tools` from the job, check if it appears in the candidate's `skills_section` OR, more importantly, in the `used_skills_and_tools` within their `work_experience`.
       Give higher weight to skills demonstrated in recent work experience.

3.  Analyze Preferred Qualifications:
       Check for the presence of `preferred_skills`, `preferred_tools`, and `preferred_certifications`. These are bonuses, so their absence is not a major penalty, but their presence is a strong positive signal.

4.  Analyze Education:
       Verify if the candidate's `education` matches the `required_education` from the job.

5.  Synthesize and Score:
       Based on your step-by-step analysis, write a summary, list the pros and cons, and determine a final score. A candidate missing a `must_have` requirement cannot score above a 6. A candidate who meets all `must_have` requirements but no preferred ones might be a 7. A candidate who meets all `must_have` and several `preferred` qualifications would be an 8-9. A 10 is reserved for an exceptionally perfect match.

Here is the data:

1. JOB_REQUIREMENTS:
```json
{job_requirements}
```

2. CANDIDATE_RESUME
```json
{resume_info}
```
"""
