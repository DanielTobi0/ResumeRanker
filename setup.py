from setuptools import setup, find_packages

setup(
    name='ResumeRanker',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'openai',
        'python-dotenv',
        'sentence-transformers',
        'torch',
        'pandas',
        'numpy',
        'scikit-learn',
        'tqdm',
    ],
)