# Persona-Driven Document Intelligence System

## Round 1B: "Connect What Matters — For the User Who Matters"

This project is a comprehensive solution for the Round 1B challenge. It is an intelligent system that analyzes collections of PDF documents, extracts the most relevant content based on a specific user persona and their goal, and presents the findings in a structured, ranked format.

---

## Key Features

* **Persona-Driven Analysis**: The core analysis is guided by a rich query formulated from the user's role, expertise, and specific job-to-be-done.
* **Advanced PDF Structure Parsing**: Goes beyond simple text extraction to understand document structure, identifying titles, hierarchical headings, and excluding noise from tables, headers, and footers.
* **Two-Stage Ranking Pipeline**: Utilizes a combination of a fast bi-encoder for initial candidate retrieval and a highly accurate cross-encoder for precise re-ranking, ensuring both speed and relevance.
* **Granular Summarization**: Extracts the most salient sentences from top-ranked sections to provide concise, actionable insights.
* **Offline & Self-Contained**: The entire solution runs within a Docker container with no internet access, bundling all models and dependencies for seamless execution.

---

## Methodology

Our approach is a multi-stage pipeline designed for accuracy, relevance, and robustness.

### Phase 1: Deep Document Parsing (`section_splitter.py`)

Before any semantic analysis, we first understand the document's structure.
1.  **Layout Reconstruction**: We use `pdfplumber` to extract detailed character-level data (font, size, coordinates) and `camelot-py` to detect and isolate tabular data.
2.  **Dynamic Baseline**: The script analyzes the entire document to dynamically determine the properties of standard body text (e.g., median font size). This adaptive baseline is crucial for accurately identifying headings, which are classified based on their deviation from this norm.
3.  **Noise Reduction**: Repeating headers and footers are identified and excluded. Text within detected table boundaries is also ignored, preventing it from polluting the semantic analysis.
4.  **Hierarchical Structuring**: A powerful rule-based engine classifies lines as headings based on font weight, size, indentation, and vertical spacing. These headings are then grouped and assigned a level (H1, H2, etc.) to create a complete document outline.

### Phase 2: Persona-Centric Query Formulation (`main.py`)

The system translates the user's context into a "rich query" that guides the AI models. This query combines the persona's role, focus areas, and the job-to-be-done, creating a detailed prompt for the retrieval models (e.g., "As an Investment Analyst, my goal is to analyze revenue trends...").

### Phase 3: Two-Stage Information Retrieval (`main.py`)

To balance speed and accuracy, we use a two-stage retrieval process on the parsed sections.
1.  **Retrieval (Bi-Encoder)**: A fast `all-MiniLM-L6-v2` model performs a semantic search across all sections from all documents, quickly identifying a broad set of the top 30 most promising candidates.
2.  **Re-Ranking (Cross-Encoder)**: These candidates are then passed to a more powerful `cross-encoder/nli-deberta-v3-small` model. This model performs a deeper, more contextually-aware comparison between the rich query and each candidate section, re-ranking them with high precision to produce the final `importance_rank`.

### Phase 4: Granular Analysis & Output Generation (`main.py`)

For the final, top-ranked sections, the system performs a sub-section analysis to extract the most relevant sentences. The results are then compiled into the required `challenge1b_output.json` format, including metadata and the ranked sections.

---

## Technology Stack

* **Python 3.11**
* **PyTorch**: Core deep learning framework.
* **Sentence-Transformers (Hugging Face)**: For bi-encoder and cross-encoder models.
* **pdfplumber & camelot-py**: For PDF parsing and table extraction.
* **Docker**: For creating a reproducible, self-contained, and offline execution environment.

---

## Project Structure

```
.
├── models/
│   ├── all-MiniLM-L6-v2/
│   └── models--cross-encoder--nli-deberta-v3-small/
├── Challenge_1b/
│   ├── Collection_1/
│   │   ├── PDFs/
│   │   ├── challenge1b_input.json
│   │   └── challenge1b_output.json
│   └── ... (other collections)
├── Dockerfile
├── main.py
├── section_splitter.py
├── requirements.txt
└── README.md
```

---

## Setup and Execution

The entire solution is designed to run within a Docker container.

### Prerequisites

* Docker Desktop installed and running.
* The required models (`all-MiniLM-L6-v2` and `models--cross-encoder--nli-deberta-v3-small`) must be present in the `./models` directory.

### Step 1: Build the Docker Image

Navigate to the project's root directory in your terminal and run the build command.

```bash
docker build -t challenge1b-solution .
```

### Step 2: Run the Analysis

Execute the container, passing the path to the specific collection you want to analyze. The path must point to the directory *inside the container* (`/app`). The output JSON will be generated within that same directory.

```bash
# Example for Collection 1 (Travel Planning)
docker run --rm challenge1b-solution /app/Challenge_1b/Collection_1

# Example for Collection 2 (Adobe Acrobat)
docker run --rm challenge1b-solution /app/Challenge_1b/Collection_2

# Example for Collection 3 (Recipes)
docker run --rm challenge1b-solution /app/Challenge_1b/Collection_3
```
After execution, the `challenge1b_output.json` file inside the respective collection folder will be updated with the analysis results.