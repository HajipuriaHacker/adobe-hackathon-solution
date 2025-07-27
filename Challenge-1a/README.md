PDF Structure Extraction - Adobe India Hackathon 2025 (Challenge 1a)
This repository contains a high-performance Python solution for Challenge 1a of the Adobe India Hackathon 2025. The application processes PDF documents to extract a structured outline, including the document title and a hierarchical list of headings (H1-H4), and outputs the results in a clean JSON format.

The solution is fully containerized using Docker and is designed to meet the strict performance and resource constraints of the challenge.

Key Features
No Heavy ML Models: The solution uses a sophisticated rule-based and heuristic engine, avoiding large model dependencies and adhering to the < 200MB size constraint.

Dynamic Document Profiling: Instead of fixed thresholds, the script first analyzes each PDF to dynamically learn the properties of its paragraph text (e.g., font size, line spacing), making it robust across various document styles.

Intelligent Content Exclusion: Automatically detects and excludes text from tables (using camelot-py), as well as repeating headers and footers, to reduce noise and improve accuracy.

Multi-Stage Processing Pipeline: Employs a robust pipeline that includes line-by-line classification, multi-line heading grouping, and a final structuring pass to ensure a logical and accurate document hierarchy.

Dockerized & Compliant: The entire solution is packaged in a lightweight Docker image, runs completely offline, and adheres to all specified build and run requirements.

How It Works: The Methodology
The core of the solution is the process_headings.py script, which follows a multi-stage process for each PDF:

Table & Page Analysis:

camelot-py is used to identify the bounding boxes of all tables in the document.

pdfplumber extracts all text lines along with their detailed properties (font, size, position).

Paragraph Profiling:

The script analyzes all extracted lines to create a "profile" of standard paragraph text, calculating the median font size, line spacing, and word count. This profile becomes the baseline for identifying anomalies.

Header & Footer Detection:

It identifies recurring text near the top and bottom of pages to dynamically define header and footer zones, which are then excluded from processing.

Title Extraction:

A specialized set of relaxed rules is applied to the first page to identify and group the main document title and potential subtitles.

Heading Classification:

Each remaining line is classified. A line is identified as a heading if it deviates significantly from the paragraph profile (e.g., larger font size, bold weight, significant surrounding whitespace) and passes a series of formatting checks.

Grouping & Structuring:

Consecutive single-line headings with similar styles are grouped into logical multi-line headings.

A final pass assigns hierarchy levels (H1-H4) based on font size ranking and contextual rules (e.g., promoting numbered headings, ensuring logical hierarchy).

Project Structure
text
.
├── sample_dataset/
│   ├── pdfs/            # Place input PDFs here for testing
│   └── outputs/         # Processed JSON files will be saved here
├── Dockerfile           # Defines the container environment
├── requirements.txt     # Python package dependencies
└── process_headings.py  # The core processing script
Setup and Usage
Prerequisites
Docker must be installed and running on your system.

Step 1: Build the Docker Image
Navigate to the project's root directory in your terminal and run the following command. This will build the image, installing all necessary system and Python dependencies.

bash
docker build --platform linux/amd64 -t pdf-processor:latest .
Step 2: Run the Processing Container
Before running, place your test PDF files inside the sample_dataset/pdfs/ directory. Then, execute the command below.

This command mounts the local input/output folders, runs the container in a completely offline environment, and automatically cleans up after completion.

For Windows (PowerShell):

powershell
docker run --rm -v "${pwd}/sample_dataset/pdfs:/app/input:ro" -v "${pwd}/sample_dataset/outputs:/app/output" --network none pdf-processor:latest
For Linux / macOS:

bash
docker run --rm -v "$(pwd)/sample_dataset/pdfs:/app/input:ro" -v "$(pwd)/sample_dataset/outputs:/app/output" --network none pdf-processor:latest
After the command finishes, the extracted JSON files will appear in your local sample_dataset/outputs/ directory.

Compliance with Hackathon Constraints
Constraint	Status	How It's Met
Execution Time	✓	The heuristic-based approach is significantly faster than loading and running large ML models. Performance on a 50-page PDF should be well within the ≤ 10 seconds limit.
Model Size	✓	The solution uses libraries, not pre-trained models. The final Docker image size is minimal and well under the ≤ 200MB constraint.
Network Access	✓	The docker run command uses the --network none flag, which completely disables all network connectivity during runtime.
Runtime / Architecture	✓	The Dockerfile is built on a python:3.10 base for the linux/amd64 platform, ensuring it runs correctly on the specified CPU architecture.
Input/Output	✓	The script correctly reads all PDFs from /app/input and writes corresponding filename.json files to /app/output.
Dependencies
System Dependencies
ghostscript

python3-tk

Python Packages
pdfplumber

camelot-py[cv]