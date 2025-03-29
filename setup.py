from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deutschnlp",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive toolkit for analyzing German text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deutschnlp",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: German",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.7",
    install_requires=[
        "spacy>=3.0.0",
        "scikit-learn>=0.24.0",
        "flask>=2.0.0",
        "numpy>=1.19.0",
    ],
    entry_points={
        "console_scripts": [
            "deutschnlp=deutschnlp.cli:main",
        ],
    },
    include_package_data=True,
)
