import logging
from pathlib import Path
from typing import Callable, Dict
import docx2txt
from PyPDF2 import PdfReader


def extract_text_from_pdf(resume_path: Path) -> str:
    """
    Extract plain text content from a PDF file.

    This function reads each page of the PDF and concatenates the extracted text.

    Args:
        resume_path (Path): Path to the PDF file

    Returns:
        str: The extracted text content

    Raises:
        Any exceptions from PyPDF2 during PDF processing
    """
    with resume_path.open("rb") as f:
        pdf_reader = PdfReader(f)
        return "".join(page.extract_text() for page in pdf_reader.pages)


def extract_text_from_docx(resume_path) -> str:
    """
    Extract plain text content from a DOCX file.

    Args:
        resume_path (Path): Path to the DOCX file

    Returns:
        str: The extracted text content

    Raises:
        Any exceptions from docx2txt during document processing
    """
    return docx2txt.process(resume_path)


def extract_text_from_txt(resume_path) -> str:
    """
    Extract plain text content from a text file.

    Args:
        resume_path (Path): Path to the text file

    Returns:
        str: The extracted text content

    Raises:
        Any exceptions during file opening/reading
    """
    with open(resume_path, "r") as file:
        return file.read()


SUPPORTED_EXTENSIONS: Dict[str, Callable] = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
    ".text": extract_text_from_txt,
}


def load_raw_resume_texts(resume_folder: Path) -> dict[str, str]:
    """
    Load text content from all supported resume files in a directory.

    This function scans the specified folder for resume files with supported
    extensions (.pdf, .docx, .txt, .text) and extracts their text content.

    Args:
        resume_folder (Path): Path to the folder containing resume files

    Returns:
        dict[str, str]: A dictionary mapping file names (without extensions)
                         to their extracted text content

    Note:
        If an error occurs during text extraction for a specific file,
        the error is printed to stdout and the file is skipped.
    """
    texts = {}
    resume_files = list(resume_folder.iterdir())
    logging.info(f"Loading resumes from {resume_folder}")
    for resume_file in resume_files:
        if resume_file.is_file():
            file_suffix = resume_file.suffix.lower()
            if file_suffix in SUPPORTED_EXTENSIONS:
                try:
                    extractor = SUPPORTED_EXTENSIONS[file_suffix]
                    file_content = extractor(resume_file)
                    texts[resume_file.stem] = file_content
                except Exception as e:
                    logging.error(f"Error processing {resume_file.name}: {str(e)}")
    return texts
