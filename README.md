# PDF Title and Structured Outline Extractor

This project is a sophisticated Python script designed to analyze PDF documents, automatically extract the main title, and generate a structured hierarchical outline (H1-H4) of its headings, conforming to the Adobe Hackathon Challenge 1a guidelines.

## Approach

The solution uses a multi-stage, rule-based approach to ensure high accuracy across various PDF layouts:

1.  **Core Text Extraction**: `pdfplumber` is used to extract detailed character, line, and font information from each page.
2.  **Table and Noise Detection**: `camelot-py` detects the bounding boxes of tables to exclude their content. The script also dynamically identifies and excludes repeating header and footer text.
3.  **Paragraph Analysis**: The script establishes a "baseline" for normal paragraph text by calculating the median font size and primary font name of the body content.
4.  **Nuanced Title Extraction**: A specialized logic block processes the first page to find the main title, intelligently grouping multi-line titles and cleaning up repeated phrases.
5.  **Rule-Based Heading Classification**: Each line is passed through a classification engine that uses font size, weight, and contextual clues to identify headings while rejecting noise like dates or bracketed text.
6.  **Hierarchical Outline Structuring**: A stack-based algorithm processes headings in document order, ensuring a structurally correct outline (e.g., an H3 only appears under an H2) and capping the hierarchy at four levels (H1-H4).

## Models or Libraries Used

*   **`pdfplumber`**: Core library for extracting text and its properties.
*   **`camelot-py`**: Used for its table detection capabilities.
*   **`pandas` / `numpy`**: Dependencies for `camelot-py`.
*   **Standard Libraries**: `re`, `json`, `pathlib`, `collections`, `statistics`.
*   All libraries used are open source and installed during the Docker build process, requiring no network access at runtime.

## How to Build and Run Your Solution

### Prerequisites

*   [Docker](https://www.docker.com/products/docker-desktop/) must be installed and running.

### Build the Docker Image

Open a terminal in the root directory of this project and run the official build command. Replace `<reponame.someidentifier>` with your unique identifier.

```
docker build --platform linux/amd64 -t <reponame.someidentifier> .
```
Example: `docker build --platform linux/amd64 -t jhaaj08.solution1 .`

### Run the Docker Container

1.  Create a local folder named `input` and place the PDFs you want to process inside it.
2.  Create a local folder named `output`. Inside it, create a sub-folder with your unique identifier (e.g., `output/jhaaj08.solution1`).
3.  From your terminal, run the official run command:

    **On macOS / Linux:**
    ```
    docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/<reponame.someidentifier>/:/app/output --network none <reponame.someidentifier>
    ```

    **On Windows (PowerShell):**
    ```
    docker run --rm -v ${PWD}/input:/app/input:ro -v ${PWD}/output/<reponame.someidentifier>/:/app/output --network none <reponame.someidentifier>
    ```
    
    **On Windows (Command Prompt):**
    ```
    docker run --rm -v "%cd%/input":/app/input:ro -v "%cd%/output/<reponame.someidentifier>":/app/output --network none <reponame.someidentifier>
    ```

After the command finishes, the `output/<reponame.someidentifier>` directory will contain the extracted `filename.json` files.
```

You are now fully compliant with the official rules. Your project is ready to be zipped and submitted. Good luck
