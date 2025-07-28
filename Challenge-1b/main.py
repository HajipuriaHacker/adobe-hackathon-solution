# main.py (Balanced Code with Minimal Progress Prints)

import json
from pathlib import Path
import sys
import argparse
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import datetime
import re
import nltk

# Ensure NLTK 'punkt' tokenizer is available for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("[INFO] NLTK data not found. Downloading 'punkt' and 'punkt_tab'...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("[INFO] Download complete.")

# Import functions from section_splitter.py
from section_splitter import collect_lines_and_paras, process_pdf

# === Global variable for generic sub-heading patterns ===
GENERIC_SECTION_TITLE_PATTERNS = [
    "ingredients:", "instructions:", "directions:",
    "preparation:", "method:", "cook time:", "prep time:"
]

# --- get_refined_text function ---
def get_refined_text(full_text, query, model, top_k=5):
    """
    Extracts the top_k most relevant sentences from a text based on a query
    using a bi-encoder (SentenceTransformer).
    Includes cleanup for common PDF extraction artifacts.
    """
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    if not sentences:
        return ""
    if len(sentences) <= top_k:
        raw_summary = " ".join(sentences)
    else:
        query_embedding = model.encode(query, convert_to_tensor=True)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, sentence_embeddings, top_k=top_k)[0]
        hits = sorted(hits, key=lambda x: x['corpus_id'])
        top_sentences = [sentences[hit['corpus_id']] for hit in hits]
        raw_summary = " ".join(top_sentences)

    final_summary = re.sub(r'^[•o*\-\s]+', '', raw_summary, flags=re.MULTILINE)
    final_summary = re.sub(r'\s+', ' ', final_summary).strip()
    final_summary = re.sub(r'\s+o\s+', ' ', final_summary)

    return final_summary

# --- build_full_text_corpus function ---
def build_full_text_corpus(collection_path: Path):
    """
    Builds a corpus where each entry contains a heading and the full text content
    that follows it, stopping at the next heading of the same or higher level.
    """
    input_config_path = collection_path / "challenge1b_input.json"
    with open(input_config_path, "r") as f:
        input_config = json.load(f)

    pdfs_dir = collection_path / "PDFS"
    documents_to_process = input_config.get("documents", [])
    corpus = []

    for doc_info in documents_to_process:
        pdf_filename = doc_info.get("filename")
        pdf_path = pdfs_dir / pdf_filename

        if not pdf_path.exists():
            continue

        print(f"[INFO] Processing document: {pdf_filename}")
        all_lines, _ = collect_lines_and_paras(pdf_path)
        if not all_lines:
            continue

        structured_output_dir = collection_path / "structured_outputs"
        structured_output_dir.mkdir(exist_ok=True)
        process_pdf(pdf_path, structured_output_dir, debug_print=False)
        output_json_path = structured_output_dir / f"{pdf_path.stem}.json"

        with open(output_json_path, "r", encoding="utf-8") as f:
            structure = json.load(f)

        headings = structure.get('outline', [])
        print(f"[INFO]   └─ Detected {len(headings)} headings.")
        if not headings:
            continue

        heading_positions = {}
        for l in all_lines:
            heading_positions.setdefault((l['page'], l['text']), l)

        last_h1_recipe_title = ""

        for i, heading in enumerate(headings):
            heading_key = (heading['page'], heading['text'])
            if heading_key not in heading_positions:
                continue

            start_line = heading_positions[heading_key]
            start_page = start_line['page']
            start_top = start_line['top']
            current_level = int(heading['level'].replace('H', ''))

            if current_level == 1:
                last_h1_recipe_title = heading['text']
            elif current_level > 1:
                is_generic = any(heading['text'].lower().startswith(p) for p in GENERIC_SECTION_TITLE_PATTERNS)
                if is_generic and not last_h1_recipe_title:
                    for prev_h_idx in range(i - 1, -1, -1):
                        if int(headings[prev_h_idx]['level'].replace('H', '')) == 1:
                            last_h1_recipe_title = headings[prev_h_idx]['text']
                            break

            end_page, end_top = float('inf'), float('inf')
            for next_heading_idx in range(i + 1, len(headings)):
                next_heading = headings[next_heading_idx]
                next_level = int(next_heading['level'].replace('H', ''))
                if next_level <= current_level:
                    next_heading_key = (next_heading['page'], next_heading['text'])
                    if next_heading_key in heading_positions:
                        next_heading_line = heading_positions[next_heading_key]
                        end_page = next_heading_line['page']
                        end_top = next_heading_line['top']
                    break

            section_lines = []
            subsequent_heading_coords = set()
            for future_heading_idx in range(i + 1, len(headings)):
                future_heading = headings[future_heading_idx]
                future_heading_key = (future_heading['page'], future_heading['text'])
                if future_heading_key in heading_positions:
                    h_line = heading_positions[future_heading_key]
                    subsequent_heading_coords.add((h_line['page'], h_line['top']))

            for line in all_lines:
                line_page, line_top = line['page'], line['top']
                
                is_after_start = (line_page > start_page) or (line_page == start_page and line_top >= start_top)
                is_before_end = (line_page < end_page) or (line_page == end_page and line_top < end_top)
                is_not_subsequent = (line_page, line_top) not in subsequent_heading_coords

                if is_after_start and is_before_end and is_not_subsequent:
                    section_lines.append(line['text'])
                elif (line_page > end_page) or (line_page == end_page and line_top >= end_top):
                    break

            full_text = "\n".join(section_lines).strip()
            if full_text:
                final_title = heading['text']
                is_generic = any(final_title.lower().startswith(p) for p in GENERIC_SECTION_TITLE_PATTERNS)
                
                if is_generic and last_h1_recipe_title:
                    final_title = f"{last_h1_recipe_title} - {final_title}"

                corpus.append({
                    "document": pdf_filename,
                    "section_title": final_title,
                    "page_number": heading['page'] + 1,
                    "full_text": full_text
                })
    return corpus

# --- run_challenge function ---
def run_challenge(collection_path: Path):
    """Main pipeline for the challenge."""
    print(f"\n--- Starting Challenge for Collection: {collection_path.name} ---")

    with open(collection_path / "challenge1b_input.json", "r") as f:
        input_config = json.load(f)

    job_to_be_done = input_config.get("job_to_be_done", {}).get("task", "N/A")
    persona = input_config.get("persona", {}).get("role", "N/A")

    # Step 1: Formulate Query and Build Corpus
    print("\n--- Step 1: Building Corpus & Formulating Query ---")
    persona_role = input_config.get("persona", {}).get("role", "User")
    persona_expertise = input_config.get("persona", {}).get("focus_areas", [])
    if persona_expertise:
        expertise_str = ", ".join(persona_expertise)
        rich_query = f"As a {persona_role} with expertise in {expertise_str}, my goal is to: {job_to_be_done}"
    else:
        rich_query = f"As a {persona_role}, my goal is to: {job_to_be_done}"

    full_text_corpus = build_full_text_corpus(collection_path)
    if not full_text_corpus:
        print("[FATAL] Could not build a text corpus from the documents. Exiting.")
        return
    print(f"[INFO] Corpus built with {len(full_text_corpus)} total sections.")


    # Step 2: Filter Corpus (if applicable)
    print("\n--- Step 2: Filtering Corpus ---")
    non_vegetarian_keywords = [
        "chicken", "beef", "pork", "lamb", "fish", "shrimp", "bacon",
        "salami", "prosciutto", "meatballs", "turkey", "crab", "lobster"
    ]
    
    original_count = len(full_text_corpus)
    filtered_corpus = []
    for section in full_text_corpus:
        section_content_lower = section['full_text'].lower() + " " + section['section_title'].lower()
        if not any(keyword in section_content_lower for keyword in non_vegetarian_keywords):
            filtered_corpus.append(section)

    full_text_corpus = filtered_corpus
    print(f"[INFO] Filtering complete. {len(full_text_corpus)} sections remaining (removed {original_count - len(full_text_corpus)}).")

    if not full_text_corpus:
        print("[ERROR] No relevant sections found in the corpus after filtering. Exiting.")
        return

    # Step 3: Retrieve and Re-Rank
    print("\n--- Step 3: Retrieving & Re-Ranking Top Sections ---")
    retriever_model = SentenceTransformer(str(Path("./models/all-MiniLM-L6-v2")))
    corpus_embeddings = retriever_model.encode([item['full_text'] for item in full_text_corpus], convert_to_tensor=True, show_progress_bar=True)
    query_embedding = retriever_model.encode(rich_query, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=30)[0]
    candidate_indices = [hit['corpus_id'] for hit in hits]

    cross_encoder_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    cross_inp = [[rich_query, full_text_corpus[idx]['full_text']] for idx in candidate_indices]
    cross_scores = cross_encoder_model.predict(cross_inp, show_progress_bar=True)

    for i in range(len(cross_scores)):
        hits[i]['cross_score'] = cross_scores[i]
    hits = sorted(hits, key=lambda x: x['cross_score'][1], reverse=True)


    # Step 4: Generate Final Output
    print("\n--- Step 4: Generating Final Output ---")
    output_data = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_config.get("documents", [])],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    final_hits = hits[:5]
    for i, hit in enumerate(final_hits):
        corpus_index = hit['corpus_id']
        section = full_text_corpus[corpus_index]
        
        print(f"[INFO] Refining text for ranked section #{i+1}: '{section['section_title'][:60]}...'")

        output_data["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": i + 1,
            "page_number": section["page_number"]
        })

        refined_summary = get_refined_text(
            section["full_text"],
            rich_query,
            retriever_model,
            top_k=5
        )

        output_data["subsection_analysis"].append({
            "document": section["document"],
            "refined_text": refined_summary,
            "page_number": section["page_number"]
        })

    output_file_path = collection_path / "challenge1b_output.json"
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\n[SUCCESS] Complete analysis saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF analysis for a specific collection.")
    parser.add_argument("collection_path", type=Path, help="The full path to the target collection directory")
    args = parser.parse_args()

    if not args.collection_path.is_dir() or not (args.collection_path / "PDFS").is_dir():
        print(f"[FATAL] The provided path '{args.collection_path}' is not a valid collection directory.")
        sys.exit(1)

    run_challenge(args.collection_path)