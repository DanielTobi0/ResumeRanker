import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from src.config import (
    BI_ENCODER_RANKING_PATH,
    DATA_DIR,
    FINAL_RANKING_PATH,
    JOB_DESCRIPTION_PATH,
    RESUMES_DIR,
    STRUCTURED_JOB_DESCRIPTION_PATH,
    STRUCTURED_RESUMES_PATH,
)
from src.extraction.job_description import extract_job_requirements
from src.extraction.resume import extract_resume_details
from src.models.encoders import get_bi_encoder, get_cross_encoder
from src.ranking.bi_encoder import bi_encoder_resume_filtering
from src.ranking.cross_encoder_llm import score_and_rank
from src.utils.text_extractor import load_raw_resume_texts

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ResumeRankingPipeline:
    """
    Orchestrates the entire resume ranking process.

    This class coordinates the multi-stage pipeline for evaluating resumes against
    a job description, including extraction, embedding, and ranking phases.
    It handles both the processing logic and data persistence between steps.

    Attributes:
        client: The OpenAI client used for text extraction and evaluation
        job_description_path: Path to the job description file
        resumes_dir: Directory containing resume files
        data_dir: Directory for storing intermediate and final results
        bi_encoder_model: The SentenceTransformer model for initial filtering
        cross_encoder_model: The CrossEncoder model for pairwise comparison
    """

    def __init__(
        self,
        top_n,
        llm_weights,
        cross_encoder_weight,
        job_description_path: Path = JOB_DESCRIPTION_PATH,
        resumes_dir: Path = RESUMES_DIR,
        data_dir: Path = DATA_DIR,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.job_description_path = job_description_path
        self.resumes_dir = resumes_dir
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

        # Default scoring metrics
        self.top_n = top_n
        self.llm_weights = llm_weights
        self.cross_encoder_weight = cross_encoder_weight

        # Paths for storing processed data
        self.structured_job_description_path = STRUCTURED_JOB_DESCRIPTION_PATH
        self.structured_resumes_path = STRUCTURED_RESUMES_PATH
        self.bi_encoder_ranking_path = BI_ENCODER_RANKING_PATH
        self.final_ranking_path = FINAL_RANKING_PATH

        # In-memory data
        self.structured_job_description: Dict[str, Any] = {}
        self.structured_resumes: List[Dict[str, Any]] = []

        # Load models
        self.bi_encoder_model = get_bi_encoder()
        self.cross_encoder_model = get_cross_encoder()

    def _extract_job_description(self):
        """
        Extracts structured data from the job description.

        Reads the job description text file, processes it using OpenAI,
        and saves the structured output to a JSON file.
        """
        logging.info("Extracting structured data from job description...")
        with open(self.job_description_path) as f:
            job_description = f.read()
        self.structured_job_description = extract_job_requirements(
            self.client, job_description
        )
        with open(self.structured_job_description_path, "w") as f:
            json.dump(self.structured_job_description, f, indent=2)
        logging.info(
            f"Saved structured job description to {self.structured_job_description_path}"
        )

    def _extract_resume_details(self):
        """
        Extracts structured data from resumes.

        Loads all resumes from the specified directory, processes each one
        using OpenAI to extract structured information, and saves the
        results to a JSON file.
        """
        logging.info("Extracting structured data from resumes...")
        raw_resumes = load_raw_resume_texts(self.resumes_dir)
        self.structured_resumes = [
            extract_resume_details(self.client, resume)
            for _, resume in raw_resumes.items()
        ]
        with open(self.structured_resumes_path, "w") as f:
            json.dump(self.structured_resumes, f, indent=2)
        logging.info(f"Saved structured resumes to {self.structured_resumes_path}")

    def _filter_with_bi_encoder(self):
        """
        Filters resumes using a bi-encoder model.

        This initial ranking step uses semantic similarity to efficiently
        compare resumes against the job description, producing a ranked
        list of candidates based on overall relevance.
        """
        logging.info("Filtering resumes with bi-encoder...")
        bi_encoder_ranking = bi_encoder_resume_filtering(
            self.structured_job_description,
            self.structured_resumes,
            self.bi_encoder_model,
        )
        with open(self.bi_encoder_ranking_path, "w") as f:
            json.dump(bi_encoder_ranking, f, indent=2)
        logging.info(f"Saved bi-encoder ranking to {self.bi_encoder_ranking_path}")

    def _load_bi_encoder_ranking(self):
        """
        Loads the bi-encoder ranking from a file.

        Returns:
            list: The previously computed bi-encoder ranking results
        """
        with open(self.bi_encoder_ranking_path, "r") as f:
            return json.load(f)

    def _score_and_rank(self):
        """
        Scores and ranks the filtered resumes.

        This final ranking step applies more intensive evaluation methods
        to the top candidates from the bi-encoder ranking. It combines scores
        from a cross-encoder model and an LLM judge to produce a comprehensive
        evaluation of each candidate.
        """
        logging.info("Scoring and ranking with cross-encoder and LLM...")
        bi_encoder_ranking = self._load_bi_encoder_ranking()
        final_ranking = score_and_rank(
            client=self.client,
            bi_encoder_ranking=bi_encoder_ranking,
            structured_resumes=self.structured_resumes,
            structured_job_description=self.structured_job_description,
            cross_encoder_model=self.cross_encoder_model,
            top_n=self.top_n,
            llm_weight=self.llm_weights,
            cross_encoder_weight=self.cross_encoder_weight,
        )
        with open(self.final_ranking_path, "w") as f:
            json.dump(final_ranking, f, indent=2)
        logging.info(f"Saved final ranking to {self.final_ranking_path}")

    def run(self):
        """
        Executes the full resume ranking pipeline.

        This method runs all steps of the pipeline in sequence:
        1. Extract structured data from the job description
        2. Extract structured data from all resumes
        3. Filters resumes using the bi-encoder.
        4. Scores and ranks the top candidates.
        """
        self._extract_job_description()
        self._extract_resume_details()
        self._filter_with_bi_encoder()
        self._score_and_rank()
        logging.info("Resume ranking pipeline completed successfully.")


def main():
    """
    Parses command-line arguments and runs the resume ranking pipeline.

    This function handles CLI argument parsing and initializes the
    ResumeRankingPipeline with the specified paths before executing it.
    """
    parser = argparse.ArgumentParser(
        description="Rank resumes against a job description."
    )
    parser.add_argument(
        "--top-resumes",
        type=int,
        default=5,
        help="Number of top ranking resumes the recuriter wants to see.",
    )
    parser.add_argument(
        "--llm-scoring-weight",
        type=float,
        default=0.7,
        help="This value is multiple against the LLM-as-a-judge resume ranking (8 * 0.7)",
    )
    parser.add_argument(
        "--cross-encoder-weight",
        type=float,
        default=0.3,
        help="This value is multiple against the LLM-as-a-judge resume ranking (8 * 0.3)",
    )
    parser.add_argument(
        "--job-description",
        type=Path,
        default=JOB_DESCRIPTION_PATH,
        help="Path to the job description file.",
    )
    parser.add_argument(
        "--resumes-dir",
        type=Path,
        default=RESUMES_DIR,
        help="Directory containing resume files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory to store intermediate and final results.",
    )
    args = parser.parse_args()

    pipeline = ResumeRankingPipeline(
        top_n=args.top_resumes,
        llm_weights=args.llm_scoring_weight,
        cross_encoder_weight=args.cross_encoder_weight,
        job_description_path=args.job_description,
        resumes_dir=args.resumes_dir,
        data_dir=args.data_dir,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
