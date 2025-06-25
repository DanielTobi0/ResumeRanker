from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RESUMES_DIR = ROOT_DIR / "resumes"

JOB_DESCRIPTION_PATH = ROOT_DIR / "job_description.txt"
STRUCTURED_JOB_DESCRIPTION_PATH = DATA_DIR / "structured_job_description.json"
STRUCTURED_RESUMES_PATH = DATA_DIR / "structured_resumes.json"
BI_ENCODER_RANKING_PATH = DATA_DIR / "bi_encoder_ranking.json"
FINAL_RANKING_PATH = DATA_DIR / "final_resume_ranking.json"
