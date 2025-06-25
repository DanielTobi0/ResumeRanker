# Resume Ranking with Encoder and Decoder Models

This uses a combination of bi-encoders, cross-encoders, and LLM, to rank candidate resumes against a given job description with strong accuracy.

## Features

- **Ranking Architecture:** starting with a fast bi-encoder for initial filtering, followed by cross-encoder and LLM-as-a-judge for detailed re-ranking.
- **Structured Data Extraction:** uses LLMs to parse unstructured text from resumes and job descriptions into a structured JSON format, capturing key details like skills, experience, and qualifications.
- **Resume Parsing:** supports resumes in various formats, including `.pdf`, `.docx`, and `.txt`.
- **Ensemble Scoring:** combines the strengths of different models by generating a weighted final score from both a cross-encoder and an LLM-based evaluation.

## How It Works

The pipeline processes resumes in several sequential stages:

1.  **Text Extraction:** Raw text is extracted from all resume files in the `resumes/` directory.
2.  **Structured Data Generation:**
    *   The raw text of the job description is converted into a structured JSON object outlining key requirements.
    *   Each resume's text is converted into a structured JSON object, detailing the candidate's profile.
3.  **Initial Filtering (Bi-Encoder):** A computationally efficient **bi-encoder** model creates embeddings for the job description and all resumes. It then calculates the cosine similarity between them to perform an initial ranking, quickly identifying a list of promising candidates.
4.  **Detailed Re-ranking (Cross-Encoder & LLM):** The top candidates from the initial filtering are passed to two more powerful models for a finer-grained analysis:
    *   **LLM-as-a-Judge:** LLM to evaluates each top candidate against the structured job requirements, providing a detailed analysis and a final score.
    *   **Cross-Encoder:** A cross-encoder model performs a direct, pairwise comparison between the job description and each resume, yielding a highly accurate similarity score.
5.  **Final Ranking:** The scores from the LLM-as-a-judge and the cross-encoder are combined using configurable weights to produce the final, ranked list of candidates. The results are saved in the `data/` directory.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DanielTobi0/ResumeRanker.git
    cd ResumeProject-main
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your-api-key-here"
    ```

### Usage

1.  Place the job description in the `job_description.txt` file.
2.  Add all candidate resumes to the `resumes/` directory.
3.  Run the main pipeline from the command line:
    ```bash
    python main.py
    ```

#### Command-Line Arguments

You can customize the pipeline's execution with the following arguments:

-   `--top-resumes`: The number of top-ranking resumes to display. (Default: `5`)
-   `--llm-scoring-weight`: The weight to apply to the LLM-as-a-judge score. (Default: `0.7`)
-   `--cross-encoder-weight`: The weight to apply to the cross-encoder score. (Default: `0.3`)
-   `--job-description`: The path to the job description file. (Default: `job_description.txt`)
-   `--resumes-dir`: The path to the directory containing resumes. (Default: `resumes/`)
-   `--data-dir`: The directory to store intermediate and final results. (Default: `data/`)

**Example:**
```bash
python main.py --top-resumes 3 --llm-scoring-weight 0.6 --cross-encoder-weight 0.4
```

## 📂 Project Structure

```
├── job_description.txt     # Input: The job description
├── main.py                 # Main script to run the pipeline
├── requirements.txt        # Project dependencies
├── resumes/                # Input: Directory for all resume files
├── data/                   # Output: Stores structured data and final rankings
├── src/                    # Source code
│   ├── extraction/         # Modules for extracting structured data
│   ├── models/             # Model loading and schema definitions
│   ├── ranking/            # Modules for ranking logic
│   └── utils/              # Helper functions and text extractors
└── cached_model/           # Stores downloaded sentence-transformer models
```
