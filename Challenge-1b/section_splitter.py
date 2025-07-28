import pdfplumber
from statistics import median, mean
from collections import defaultdict, Counter
import sys
from pathlib import Path
import json
import re
import os
import camelot
import unicodedata # Added for NFKC normalization


# --- Global Constants / Regex Patterns ---
DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s+\d{4})?\b"
)
VERSION_PATTERN = re.compile(
    r"\b(?:version|rev(?:ision)?)[\s:]\d+(\.\d+)\b", re.IGNORECASE
)
PAGE_NUMBER_PATTERN = re.compile(r"^\s*PAGES?\s+\d+\s*$", re.IGNORECASE)

# Define common generic recipe sub-headings (case-insensitive)
# Moved here to be accessible globally for more robust pattern matching
GENERIC_SUB_HEADING_PATTERNS = [
    r"ingredients:?", r"instructions:?", r"directions:?",
    r"preparation:?", r"method:?", r"cook time:?", r"prep time:?"
]
COMPILED_GENERIC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in GENERIC_SUB_HEADING_PATTERNS]


# --- Core Helper Functions ---


def normalize_text_for_lookup(text: str) -> str:
    """
    Normalizes text for consistent lookup by replacing common ligatures
    or problematic characters with their standard ASCII equivalents,
    and using NFKC normalization for broader compatibility.
    Also handles common PDF text extraction artifacts.
    """
    # Apply NFKC normalization first to handle many Unicode compatibility issues (like ligatures)
    text = unicodedata.normalize('NFKC', text)

    # Replace common ligatures that NFKC might miss or that are common PDF artifacts
    # These are often already handled by NFKC, but explicit replacement adds robustness
    text = text.replace('ﬃ', 'ffi')
    text = text.replace('ﬀ', 'ff')
    text = text.replace('ﬁ', 'fi')

    # Clean up common bullet/list remnants if they appear mid-text
    text = re.sub(r'\s+[•o*\-]\s*', ' ', text)

    # Attempt to correct common OCR/extraction errors like 'f' for 'fo', 'n' for 'on', 'rn' for 'm' etc.
    # These are heuristics and might introduce new errors. Use with caution.
    # Only apply if 'f' or 'n' is a standalone word or at the start of a word followed by space,
    # as these were observed in the output.
    text = re.sub(r'\bf\s', 'fo ', text) # Heuristic for "f " becoming "fo " (e.g., 'f assign' -> 'fo assign')
    text = re.sub(r'\bn\s', 'on ', text) # Heuristic for "n " becoming "on " (e.g., 'n the document' -> 'on the document')
    text = re.sub(r'\bil', 'oil', text)  # Heuristic for "il" becoming "oil" (e.g., 'sesame il' -> 'sesame oil')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_bold(fontname):
    """Checks if a fontname string implies a bold weight."""
    return any(x in fontname.lower() for x in ['bold', 'black', 'heavy', 'ultrabold', 'extrabold'])


def is_fake_bold(word):
    """Checks if a word is likely 'fake bold' (e.g., "HHeelloo")."""
    matches = re.findall(r'(.)\1', word)
    return len(matches) >= 2 and len(matches) >= (len(word) // 2)


def collapse_fake_bold(text):
    """Collapses 'fake bold' text into normal text."""
    words = text.split()
    collapsed_words = [re.sub(r'(.)\1+', r'\1', w) if is_fake_bold(w) else w for w in words]
    return " ".join(collapsed_words)


def deduplicate_title_text(text):
    """
    Cleans up titles by finding and removing repeated prefixes.
    e.g., "Title A Title A Subtitle B" becomes "Title A Subtitle B".
    """
    text = text.strip()
    for i in range(len(text) // 2, 3, -1):
        prefix = text[:i]
        next_chunk = text[i:i*2]
        if prefix.strip() == next_chunk.strip() and prefix.strip():
            rest_of_string = text[i*2:].strip()
            return f"{prefix.strip()} {rest_of_string}".strip()
    return text


def get_line_props(chars):
    """Calculates properties for a line of text based on its characters."""
    if not chars:
        return 0, '', 0, 0, 0, 0, 0, 0

    size = round(median([c["size"] for c in chars]), 1)
    fontnames = [c.get("fontname", "") for c in chars]
    fontname = Counter(fontnames).most_common(1)[0][0] if fontnames else ''
    x0 = min(c['x0'] for c in chars)
    x1 = max(c['x1'] for c in chars)
    top = min(c['top'] for c in chars)
    bottom = max(c['bottom'] for c in chars)

    # Calculate initial x0 of first word for indentation checks
    first_word_x0 = x0
    for c in chars:
        if not c["text"].isspace():
            first_word_x0 = c["x0"]
            break

    return size, fontname, x0, x1, top, bottom, first_word_x0, len(chars)


# --- Paragraph & Stats Functions ---


def calculate_dynamic_params(lines):
    """Analyzes lines to determine dynamic parameters for paragraph detection."""
    longer_line_word_counts = [line['words'] for line in lines if line['words'] > 3]
    if longer_line_word_counts:
        median_word_count = median(longer_line_word_counts)
        dynamic_min_words = max(4, round(median_word_count * 0.8))
    else:
        dynamic_min_words = 7 # Fallback


    gap_factors = []
    for i in range(1, len(lines)):
        prev_line, curr_line = lines[i-1], lines[i]
        if prev_line['page'] != curr_line['page']:
            continue
        gap = curr_line['top'] - prev_line['bottom']
        if gap > 0 and curr_line['avg_size'] > 0:
            factor = gap / curr_line['avg_size']
            if 0.5 < factor < 3.0: # Filter out extreme outliers
                gap_factors.append(factor)


    if gap_factors:
        median_gap_factor = median(gap_factors)
        dynamic_y_gap_factor = max(1.5, median_gap_factor * 1.5)
    else:
        dynamic_y_gap_factor = 1.7 # Fallback (typical single line spacing factor)


    return {"min_words": dynamic_min_words, "y_gap_factor": dynamic_y_gap_factor}


def flush_para_buffer(buffer, target_list, min_lines=2):
    """Appends buffer to target list if it meets the minimum line count."""
    if len(buffer) >= min_lines:
        target_list.extend(buffer)
    return []


def collect_lines_and_paras(pdf_path: Path, n_pages: int = 10000):
    """
    Extracts all lines from a PDF, calculates their properties,
    and identifies paragraph lines using dynamic parameters.
    Applies text normalization during line collection.
    """
    all_lines, lines_by_page = [], defaultdict(list)

    with pdfplumber.open(str(pdf_path)) as pdf:
        for idx, page in enumerate(pdf.pages):
            if idx >= n_pages: break

            page_h, page_w = page.height, page.width
            line_groups = defaultdict(list)
            for ch in page.chars:
                line_groups[round(ch["top"], 1)].append(ch)


            page_lines = []
            for y, group in sorted(line_groups.items()):
                group.sort(key=lambda c: c["x0"])
                text = "".join(c["text"] for c in group).strip()
                if not text: continue

                # === NEW: Normalize text immediately after extraction ===
                text = normalize_text_for_lookup(text)
                # === END NEW ===


                avg_size, fontname, x0, x1, top, bottom, x0_first_word, num_chars = get_line_props(group)
                line = {
                    "page": idx, "text": text, "words": len(text.split()),
                    "chars_raw": num_chars, # Raw character count before cleaning/normalization
                    "avg_size": avg_size, "fontname": fontname,
                    "x0": x0, "x1": x1, "x0_first_word": x0_first_word, "top": top,
                    "bottom": bottom, "y": y, "page_h": page_h, "page_w": page_w
                }
                page_lines.append(line)

            # Calculate above/below whitespace AFTER all lines for the page are collected
            for i, line in enumerate(page_lines):
                line['above_ws'] = line['top'] - page_lines[i-1]['bottom'] if i > 0 else line['top']
                line['below_ws'] = page_lines[i+1]['top'] - line['bottom'] if i < len(page_lines)-1 else page_h - line['bottom']
                lines_by_page[idx].append(line)
            all_lines.extend(page_lines)


    dynamic_params = calculate_dynamic_params(all_lines)
    PARA_MIN_WORDS = dynamic_params['min_words']
    PARA_Y_GAP_FACTOR = dynamic_params['y_gap_factor']

    para_lines = []
    for page_idx, lines in lines_by_page.items():
        para_buffer = []
        prev_line = None
        for line in lines:
            if len(line["text"]) < 3 or is_numbery(line["text"]): continue # filter very short or numeric lines early


            # Condition to break paragraph: short line, large vertical gap, or significant font/indent change
            is_para_break = False
            if len(line["text"].split()) < PARA_MIN_WORDS:
                is_para_break = True
            elif prev_line:
                dy = line["y"] - prev_line["y"]
                size_diff_abs = abs(line["avg_size"] - prev_line["avg_size"])
                x0_diff_abs = abs(line["x0"] - prev_line["x0"])

                if dy > PARA_Y_GAP_FACTOR * line["avg_size"]:
                    is_para_break = True # Large vertical gap
                elif size_diff_abs > 1.5 and (line['avg_size'] > prev_line['avg_size'] or is_bold(line['fontname'])): # Significant font size/bold change (potential heading)
                    is_para_break = True
                elif x0_diff_abs > 30: # Significant indentation change (potential new block)
                     is_para_break = True


            if is_para_break:
                para_buffer = flush_para_buffer(para_buffer, para_lines)
                prev_line = None # Reset prev_line after flushing


            para_buffer.append(line)
            prev_line = line

        flush_para_buffer(para_buffer, para_lines) # Flush any remaining buffer at end of page


    return all_lines, para_lines


def analyze_paragraph_fonts(para_lines, all_lines_fallback):
    source_lines = para_lines if para_lines else all_lines_fallback
    if not source_lines: return {
        "n_para_lines": 0, "para_median": 12, "para_mean": 12, "para_range": (12,12),
        "para_hist": [], "para_words_mean": 10, "is_fallback": True,
        "para_fontname_primary": "ArialMT", "para_x0_median": 0
    }


    para_sizes = [l['avg_size'] for l in source_lines]
    para_words = [l['words'] for l in source_lines]
    para_fontnames = [l['fontname'] for l in source_lines if l['fontname']]
    para_x0s = [l['x0'] for l in source_lines]


    primary_font = Counter(para_fontnames).most_common(1)[0][0] if para_fontnames else "ArialMT"
    median_x0 = median(para_x0s) if para_x0s else 0


    return {
        "n_para_lines": len(source_lines),
        "para_median": median(para_sizes) if para_sizes else 12,
        "para_mean": mean(para_sizes) if para_sizes else 12,
        "para_range": (min(para_sizes), max(para_sizes)) if para_sizes else (12, 12),
        "para_hist": Counter(para_sizes).most_common(5),
        "para_words_mean": mean(para_words) if para_words else 10,
        "para_fontname_primary": primary_font,
        "para_x0_median": median_x0,
        "is_fallback": not bool(para_lines)
    }


def calculate_dynamic_zones(all_lines, pages_count):
    header_candidates, footer_candidates = [], []
    CANDIDATE_ZONE_RATIO = 0.15 # Reduced ratio, common for headers/footers to be within 15%


    for line in all_lines:
        normalized_text = re.sub(r'[\d\s]+', '', line['text']).strip() # Remove digits and spaces for better pattern matching
        if len(normalized_text) < 4: continue # Too short to be a reliable header/footer content


        if line['top'] < CANDIDATE_ZONE_RATIO * line['page_h']:
            header_candidates.append({'normalized': normalized_text, 'bottom': line['bottom']})

        if line['bottom'] > (1 - CANDIDATE_ZONE_RATIO) * line['page_h']: # Use bottom for footer zone check
            footer_candidates.append({'normalized': normalized_text, 'top': line['top']})


    header_text_counts = Counter(cand['normalized'] for cand in header_candidates)
    footer_text_counts = Counter(cand['normalized'] for cand in footer_candidates)


    # A repeating header/footer should appear on more than 50% of the pages
    repeating_header_texts = {text for text, count in header_text_counts.items() if count > pages_count / 2 and text not in ['of']} # 'of' can be part of page 'x of y'
    repeating_footer_texts = {text for text, count in footer_text_counts.items() if count > pages_count / 2 and text not in ['of']}


    header_y, footer_y = None, None
    if repeating_header_texts:
        # Find the lowest bottom coordinate among repeating headers to set the exclusion line
        lowest_header_bottom = max(cand['bottom'] for cand in header_candidates if cand['normalized'] in repeating_header_texts)
        header_y = lowest_header_bottom + 5 # Add a small buffer


    if repeating_footer_texts:
        # Find the highest top coordinate among repeating footers to set the exclusion line
        highest_footer_top = min(cand['top'] for cand in footer_candidates if cand['normalized'] in repeating_footer_texts)
        footer_y = highest_footer_top - 5 # Subtract a small buffer

    return {"header_y": header_y, "footer_y": footer_y}


# --- Heading Classification Logic ---


def is_numbery(text):
    s = text.strip()
    return bool(re.fullmatch(r'^\d+([,.]\d+)*$', s) or # Pure numbers (e.g., "1", "1,000")
                re.fullmatch(r'^[-\d]+$', s) or # Negative numbers "-123"
                re.fullmatch(r'^\d{1,2}/\d{1,2}/\d{2,4}$', s) or # Dates "MM/DD/YYYY"
                re.fullmatch(r'^\d+(\.\d+)*[a-zA-Z]$', s) or # "1.A", "2.1B"
                re.fullmatch(r'^\(?\d{1,3}\)?([.-]\d{3})?[.-]\d{4}$', s) or # Phone numbers
                re.fullmatch(r'^\d+(st|nd|rd|th)$', s, re.IGNORECASE) or # Ordinal numbers "1st"
                re.fullmatch(r'^[A-Z]\d+$', s) # "A1", "B2"
               )


def is_likely_list_item(line, page_lines, para_stats):
    """
    Checks if a line is likely a list item based on indentation, font similarity,
    and length relative to paragraph text.
    """
    # Reject if it's explicitly a heading by style (bold and larger than para)
    if (is_bold(line.get('fontname', '')) and not is_bold(para_stats.get("para_fontname_primary", ""))) or \
       (line['avg_size'] > para_stats['para_median'] * 1.1):
        return False


    # Check for bullet points or numbering at the start of the line
    if re.match(r'^\s*([•*\-–—]\s+|\d+\.\s+|\([a-zA-Z]\)\s+)', line['text']):
        return True


    # Contextual check: if similar font/size/indent lines are nearby
    line_x0_tolerance = 15 # x0 can vary slightly for list items
    size_tolerance = 0.5 # font size can vary slightly


    current_index = -1
    for idx, l in enumerate(page_lines):
        if l['page'] == line['page'] and l['top'] == line['top']: # Find the line by its unique coordinates
            current_index = idx
            break
    if current_index == -1: return False


    similar_count = 0
    # Check lines before and after for similar indentation and style
    for offset in [-2, -1, 1, 2]:
        neighbor_index = current_index + offset
        if 0 <= neighbor_index < len(page_lines):
            neighbor = page_lines[neighbor_index]
            if abs(line['x0_first_word'] - neighbor['x0_first_word']) <= line_x0_tolerance and \
               abs(line['avg_size'] - neighbor['avg_size']) <= size_tolerance and \
               is_bold(line['fontname']) == is_bold(neighbor['fontname']): # Check bold match
                similar_count += 1

    return similar_count >= 1 # If at least one similar neighbor is found


def heading_format_ok(text):
    """Enhanced check to reject noisy or non-semantic headings."""
    text = text.strip()

    # NEW: Blacklist common uninformative words if they are very short headings.
    UNINFORMATIVE_WORDS = ["next", "all", "table", "content", "page", "chapter", "section", "figure", "appendix", "index", "password"]
    if text.lower() in UNINFORMATIVE_WORDS and len(text.split()) <= 2: # Allow slightly longer uninformative phrases
        return False # Reject single or two-word uninformative headings

    # Rule 1: Basic format checks
    if not text: return False
    # If the line starts with a common list/bullet character or a number sequence, it's likely a list item or body text.
    if re.match(r'^\s*([•*\-–—]|\d+\.|\([a-zA-Z]\))', text): return False
    # If the line ends with common punctuation for regular sentences or incomplete thoughts
    # NEW: Allow colon for headings like "Summary:" but exclude other typical sentence enders.
    if text.endswith(('.', ';', ',', '-')): return False
    # If line contains very little alphabetic content compared to numbers/symbols
    if sum(c.isalpha() for c in text) < len(text) * 0.2: return False


    # Rule 2: Reject if all lowercase (and contains at least one letter) and not a common generic heading
    is_generic_pattern = False
    for pattern in COMPILED_GENERIC_PATTERNS:
        if pattern.fullmatch(text): # Use fullmatch for strictness
            is_generic_pattern = True
            break
    if text.islower() and any(c.isalpha() for c in text) and not is_generic_pattern:
        return False

    # Rule 3: Reject if it looks like a simple number or code (more refined than is_numbery)
    if is_numbery(text): return False


    # Rule 4: Reject if the entire line is enclosed in brackets and very short
    if (re.fullmatch(r'\(.+\)', text) or re.fullmatch(r'\[.+\]', text)) and len(text) < 15:
        return False

    # Rule 5: Reject if the line is likely just a date or version number or page number
    if DATE_PATTERN.search(text) or VERSION_PATTERN.search(text) or PAGE_NUMBER_PATTERN.fullmatch(text):
        return False

    return True


def classify_line(line, para_stats, page_lines, dynamic_zones, table_bboxes_by_page, prev_line_was_heading=False):
    """
    Applies a sequence of rules with precedence to classify a line as a heading.
    """
    text = line['text'] # Text is already normalized by collect_lines_and_paras
    if not text: return False, "Empty line"


    # --- SECTION 1: ABSOLUTE PRE-CHECKS (Strong Rejections) ---
    if is_in_table(line, table_bboxes_by_page):
        return False, "Rejected: Located inside a detected table"

    header_y, footer_y = dynamic_zones.get("header_y"), dynamic_zones.get("footer_y")
    if header_y and line['bottom'] <= header_y: # Use bottom to ensure entire line is above header line
        return False, f"In dynamic header zone (bottom <= {header_y:.0f})"
    if footer_y and line['top'] >= footer_y: # Use top to ensure entire line is below footer line
        return False, f"In dynamic footer zone (top >= {footer_y:.0f})"


    # If format is clearly bad, reject early (except for specific generic headings like 'Ingredients:')
    is_generic_heading_text = any(pattern.fullmatch(text) for pattern in COMPILED_GENERIC_PATTERNS)
    if not heading_format_ok(text) and not is_generic_heading_text:
        return False, "Rejected: Bad format (e.g., list item, all lowercase, numeric, uninformative, etc.)"

    # Very short lines are typically not headings, unless they are styled.
    if len(text.split()) < 2 and not (is_bold(line['fontname']) or line['avg_size'] > para_stats['para_median'] * 1.1):
        return False, "Rejected: Too short and not styled"


    # --- SECTION 2: STRONG ACCEPTANCE RULES ---
    # Rule: Significantly larger font size than average paragraph text
    if line['avg_size'] > para_stats['para_median'] * 1.2: # 20% larger
        return True, f"Accepted: Significantly larger font ({line['avg_size']:.1f} > {para_stats['para_median']:.1f})"

    # Rule: Bold font, and paragraph text is not bold (or is default font)
    is_line_bold, is_para_font_bold = is_bold(line['fontname']), is_bold(para_stats.get("para_fontname_primary", ""))
    if is_line_bold and not is_para_font_bold:
        return True, f"Accepted: Bold font ('{line['fontname']}')"


    # Rule: Large vertical gap above and below, suggesting it's isolated
    if (line['above_ws'] + line['below_ws']) > line['avg_size'] * 3.0: # Significant vertical isolation
        return True, "Accepted: High vertical whitespace isolation"


    # Rule: Generic heading pattern, and not a list item
    if is_generic_heading_text and not is_likely_list_item(line, page_lines, para_stats):
        return True, f"Accepted: Matched generic heading pattern ('{text}')"


    # --- SECTION 3: CONTEXT-SENSITIVE RULES ---
    # Rule: Short line that is likely not a paragraph based on typical line length
    if line['words'] < para_stats['para_words_mean'] * 0.8: # Shorter than 80% of avg paragraph line
        # If it's a short line and it's not a list item by features
        if not is_likely_list_item(line, page_lines, para_stats):
            # Check for changes in x0 relative to body text
            if abs(line['x0'] - para_stats['para_x0_median']) > 5: # Slightly different indentation
                return True, "Accepted: Short line, not list item, slightly indented"
            # If the previous line was a heading, and this one is visually similar (but maybe not bold/large enough to be accepted directly)
            if prev_line_was_heading and abs(line['avg_size'] - para_stats['para_median']) < 2.0:
                return True, "Accepted: Short line, follows a heading, similar font size"

    return False, "Rejected: Resembles body text or unclassified content"


# --- Main Processing and Grouping ---


def is_in_table(line, table_bboxes_by_page):
    """Checks if a line is within any of the detected table bounding boxes."""
    page_num = line['page']

    # --- CRITICAL FIX: Handle cases where no tables are detected on the page ---
    if not table_bboxes_by_page.get(page_num):
        return False
    # --- END FIX ---

    if page_num in table_bboxes_by_page:
        # A line is a horizontal strip, check if its vertical span overlaps the table's
        # and its horizontal span is roughly within the table's bounds

        for bbox in table_bboxes_by_page[page_num]:
            line_v_overlap = max(0, min(line['bottom'], bbox[3]) - max(line['top'], bbox[1]))
            line_h_overlap = max(0, min(line['x1'], bbox[2]) - max(line['x0'], bbox[0]))

            # Consider a line 'in table' if a significant part of it is within the table bbox
            if line_v_overlap > (line['bottom'] - line['top']) * 0.5 and \
               line_h_overlap > (line['x1'] - line['x0']) * 0.5:
                return True
    return False


def detect_headings(all_lines, para_stats, dynamic_zones, table_bboxes_by_page): # Removed debug parameter
    accepted, rejected = [], []
    lines_by_page = defaultdict(list)
    for l in all_lines: lines_by_page[l['page']].append(l)


    # Need to keep track of the previous line's classification for contextual rules
    last_accepted_heading_line = None


    for page_idx in sorted(lines_by_page.keys()):
        page_lines = lines_by_page[page_idx]
        for line_idx, line in enumerate(page_lines):
            # Determine if the immediately preceding line on the same page was an accepted heading
            prev_line_was_accepted_heading = False
            if last_accepted_heading_line and last_accepted_heading_line['page'] == line['page']:
                # Simple check for immediate vertical proximity for multiline headings
                # Or if the current line is a potential subtitle (smaller, but still heading-like)
                if (line['top'] - last_accepted_heading_line['bottom']) < (line['avg_size'] * 2.0) : # 2 linespaces
                    prev_line_was_accepted_heading = True


            is_heading, reason = classify_line(line, para_stats, page_lines, dynamic_zones, table_bboxes_by_page, prev_line_was_accepted_heading)

            line_info = {**line, "why": reason, "text": line['text']} # line['text'] is already collapsed/normalized

            if is_heading:
                accepted.append(line_info)
                last_accepted_heading_line = line_info # Update for next iteration
            else:
                rejected.append(line_info)
                last_accepted_heading_line = None # Reset if not a heading

            # Removed: if debug: print(f"[{'ACCEPTED' if is_heading else 'REJECTED'}] Pg{line['page']}: {line_info['text'][:60]!r} | {reason}")


    return accepted, rejected


def group_multiline_headings(headings, all_lines):
    """
    Groups lines that are likely part of the same multi-line heading.
    Relies on visual proximity, similar styling (font, size), and lack of
    intervening non-heading content.
    """
    if not headings: return []

    grouped_headings = []
    used_indices = set()

    # Sort headings by their position in the document
    sorted_headings = sorted(headings, key=lambda h: (h['page'], h['top']))

    # Create a quick lookup for all detected heading coordinates
    all_heading_coords = {(h['page'], h['top']) for h in sorted_headings}


    # Pre-calculate an average whitespace for the document (heuristic for line spacing)
    all_ws_values = [h['above_ws'] for h in sorted_headings if h['above_ws'] is not None] + \
                    [h['below_ws'] for h in sorted_headings if h['below_ws'] is not None]
    avg_ws = mean(all_ws_values) if all_ws_values else 0

    for i, curr in enumerate(sorted_headings):
        if i in used_indices:
            continue

        current_group = [curr]
        used_indices.add(i)

        for j in range(i + 1, len(sorted_headings)):
            candidate = sorted_headings[j]

            # Stop if candidate is already used
            if j in used_indices:
                break

            # Conditions for breaking a multi-line heading group:
            # 1. Different page
            if candidate['page'] != curr['page']:
                break

            # 2. Significant style mismatch (different font, size, or too much x0 difference)
            if candidate['fontname'] != curr['fontname'] or \
               abs(candidate['avg_size'] - curr['avg_size']) > 0.5 or \
               abs(candidate['x0'] - curr['x0']) > 20: # Allow slight x0 variance for wrapped text
                break

            # 3. Intervening non-heading lines or large vertical gap between current and candidate
            # Check lines between the bottom of the current line and the top of the candidate line
            intervening_lines = [
                l for l in all_lines
                if l['page'] == curr['page'] and curr['bottom'] < l['top'] and l['top'] < candidate['top']
            ]

            # If there are any intervening lines that are NOT themselves headings, break.
            if intervening_lines and any((l['page'], l['top']) not in all_heading_coords for l in intervening_lines):
                break

            # If the vertical whitespace to the candidate is too large (more than 1.5x average line spacing)
            vertical_gap = candidate['top'] - curr['bottom']
            if vertical_gap > (curr['avg_size'] * 1.5): # Heuristic for normal line spacing
                break


            # If all conditions pass, add to the current group
            current_group.append(candidate)
            used_indices.add(j)
            curr = candidate # Update current to last line in group for next comparison


        # Form the final grouped heading
        if current_group:
            first_line = current_group[0]
            last_line = current_group[-1]
            combined_text = " ".join(h['text'] for h in current_group) # Text already normalized

            grouped_headings.append({
                "page": first_line['page'],
                "text": combined_text,
                "avg_size": first_line['avg_size'],
                "fontname": first_line['fontname'],
                "x0": first_line['x0'],
                "x1": last_line['x1'], # Extend x1 to cover all lines
                "top": first_line['top'],
                "bottom": last_line['bottom'], # Extend bottom to cover all lines
                "page_h": first_line['page_h'],
                "page_w": first_line['page_w'],
                "above_ws": first_line['above_ws'], # Keep original above_ws for first line
                "below_ws": last_line['below_ws'], # Keep original below_ws for last line
                "x0_first_word": first_line['x0_first_word']
            })

    return grouped_headings



def clean_headings_for_output(grouped_headings):
    """
    Cleans up the grouped headings and ensures all necessary keys are present
    for the final structuring step.
    """
    cleaned_list = []
    for h in grouped_headings:
        cleaned_list.append({
            "page": h.get('page'),
            "text": h.get('text', ''),
            "words": len(h.get('text', '').split()),
            "chars": len(h.get('text', '')),
            "avg_size": h.get('avg_size'),
            "fontname": h.get('fontname'),
            "above_ws": h.get('above_ws'), # Added
            "below_ws": h.get('below_ws'), # Added
            "x0": h.get('x0'),
            "x1": h.get('x1'),
            "top": h.get('top'),
            "bottom": h.get('bottom'),
            "page_h": h.get('page_h'),
            "page_w": h.get('page_w'),
            "x0_first_word": h.get('x0_first_word') # Added for better indentation analysis
        })
    return cleaned_list



def structure_outline_with_final_rules(headings):
    """
    This is the definitive structuring function. It now more aggressively enforces
    the hierarchy for recipe documents, ensuring generic sub-headings are demoted.
    """
    if not headings:
        return []


    # 1. Sort all headings by their position in the document
    outline_headings = sorted(headings, key=lambda h: (h['page'], h['top']))


    # === NEW: Normalize heading text immediately after sorting ===
    for h in outline_headings:
        h['text'] = normalize_text_for_lookup(h['text'])
    # === END NEW ===


    # 2. Establish a font-size-to-rank mapping
    all_heading_sizes = sorted(list(set(h.get('avg_size', 0) for h in outline_headings)), reverse=True)
    font_size_level_map = {size: i + 1 for i, size in enumerate(all_heading_sizes)}


    # === REVAMPED PRIMARY LEVELING: Prioritize x0, then font size, then a strong default ===
    unique_x0s = sorted(list(set(h['x0_first_word'] for h in outline_headings)))
    use_x0_for_initial_leveling = len(unique_x0s) > 1 and (max(unique_x0s) - min(unique_x0s)) > 5.0
    x0_to_level_map = {}
    if use_x0_for_initial_leveling:
        for i, x0_val in enumerate(unique_x0s):
            x0_to_level_map[x0_val] = i + 1 # Assign based on x0


    for h in outline_headings:
        # Default level (can be adjusted by rules later)
        h['level'] = 99 # High number, indicating unassigned or low priority


        if use_x0_for_initial_leveling:
            h['level'] = x0_to_level_map.get(h['x0_first_word'], 99)

        # If x0-based level is not assigned or is too high (means it didn't align to main x0s)
        # or if not using x0, fallback to font size.
        if h['level'] == 99 or not use_x0_for_initial_leveling:
             h['level'] = font_size_level_map.get(h.get('avg_size', 0), 99)


        # Ensure a base level is always set
        if h['level'] == 99:
            h['level'] = 1 # Default to H1 if no strong cues, then demote by rules


    # Removed: print(f"[DEBUG] Assigned levels (after initial x0/font-size pass, before rules):")
    # Removed loop: for h in outline_headings: print(f"{h['text']} - x0: {h['x0_first_word']:.2f} - level: H{h['level']}")
    # === END REVAMPED PRIMARY LEVELING ===


    # 4. Perform a final correction pass using refinement rules
    corrected_headings = []
    if not outline_headings:
        return []

    corrected_headings.append(outline_headings[0])


    for i in range(1, len(outline_headings)):
        prev_h = corrected_headings[-1]
        curr_h = outline_headings[i]

        # --- REVISED RULE 0: Aggressive Demotion for Generic Headings ---
        is_generic_sub_heading = False
        for pattern in COMPILED_GENERIC_PATTERNS:
            if pattern.match(curr_h.get('text', '').strip()):
                is_generic_sub_heading = True
                break

        if is_generic_sub_heading:
            # If a generic heading, force it to be H2 or H3 relative to its parent,
            # regardless of its initial assigned level.
            # This rule will take precedence for generic headings.
            if prev_h.get('level') == 1:
                curr_h['level'] = 2 # Directly force to H2
            elif prev_h.get('level') == 2:
                curr_h['level'] = 3 # Directly force to H3
            else: # If previous was already H3 or lower, keep it consistent
                curr_h['level'] = max(3, prev_h.get('level', 3) + 1) # Cap at H4 max
                curr_h['level'] = min(curr_h['level'], 4)
            # Add it to corrected_headings and skip further rules for this heading.
            corrected_headings.append(curr_h)
            continue # This ensures no other rules interfere with this specific generic heading
        # --- END REVISED RULE 0 ---


        # --- RULE 1: Force Numbered Headings (e.g., "1. Section", "2.1 Subsection") ---
        # Only if it's not a generic sub-heading that was already handled by Rule 0
        # (This check is now redundant due to 'continue' above, but harmless)
        if re.match(r'^\d+(\.\d+)*[\.\s]', curr_h.get('text', '')):
            curr_h['level'] = min(curr_h['level'], 3) # Cap to H3, common for numbered sections


        # --- RULE 2: H1 Subtitle Promotion ---
        # This rule promotes a line to H1 if it's visually very close to a preceding H1
        # and has similar visual properties (alignment or top-tier font size).
        # This is for multi-line main titles or subtitles that are effectively H1.
        if prev_h.get('level') == 1 and curr_h.get('page') == prev_h.get('page'):
            is_close_vertically = (curr_h.get('top', 0) - prev_h.get('bottom', 0)) < (prev_h.get('avg_size', 12) * 3)
            is_aligned_or_slightly_indented = abs(curr_h.get('x0_first_word', 0) - prev_h.get('x0_first_word', 0)) < 20
            is_top_tier_font = len(all_heading_sizes) > 0 and curr_h.get('avg_size', 0) >= (all_heading_sizes[0] * 0.85)


            if is_close_vertically and (is_aligned_or_slightly_indented or is_top_tier_font):
                curr_h['level'] = 1 # Promote to H1 (subtitle)

        # --- Final Hierarchy Rule (Crucial): Prevents large jumps in hierarchy ---
        # Ensures that a heading's level doesn't jump too far from its parent's level.
        # This is vital for x0-based leveling where a small x0 difference could otherwise
        # imply a large level jump. Caps level to be at most 1 level deeper than previous.
        if curr_h.get('level', 99) > prev_h.get('level', 99) + 1:
            curr_h['level'] = prev_h.get('level', 99) + 1

        corrected_headings.append(curr_h) # This line executes only if `continue` above was NOT hit


    # 5. Final Pass: Apply the overall H4 Cap and Format for Output
    structured_outline = []
    for h in corrected_headings:
        final_level = min(h['level'], 4) # Cap the level at H4
        structured_outline.append({
            "level": f"H{final_level}",
            "text": h['text'],
            "page": h['page']
        })

    return structured_outline


def process_pdf(pdf_path, output_dir, debug_print=False): # Retained debug_print parameter in process_pdf
    """Main pipeline to process a single PDF file."""
    # Removed: print(f"\n[INFO] Processing: {pdf_path}")

    table_bboxes_by_page = defaultdict(list)
    pages_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_count = len(pdf.pages)

            # Use camelot for table detection. Lattice mode is good for structured tables.
            tables = camelot.read_pdf(str(pdf_path), pages=f'1-{pages_count}', flavor='lattice', suppress_stdout=True)
            # Removed: print(f"[INFO] Camelot found {tables.n} tables in total.")

            for table in tables:
                page_num = table.page - 1 # Camelot pages are 1-indexed, pdfplumber/our pages are 0-indexed
                if page_num < pages_count:
                    x0, y1_camelot, x1, y2_camelot = table._bbox
                    # Camelot's y-coordinates are from bottom-left origin.
                    # pdfplumber uses top-left origin, increasing downwards.
                    # So, y1_camelot is the bottom y-coord, y2_camelot is the top y-coord.
                    # To convert to pdfplumber's 'top' and 'bottom':
                    page_height = pdf.pages[page_num].height # Get page height for coordinate conversion
                    top = page_height - y2_camelot # top is page_height - camelot_top_y
                    bottom = page_height - y1_camelot # bottom is page_height - camelot_bottom_y
                    table_bboxes_by_page[page_num].append((x0, top, x1, bottom))

    except Exception as e:
        # Removed: print(f"[WARN] Camelot failed or PDF is unscannable: {e}. Continuing without table detection.")
        if pages_count == 0: # Ensure pages_count is set even if camelot fails
            with pdfplumber.open(pdf_path) as pdf:
                pages_count = len(pdf.pages)

    all_lines, para_lines = collect_lines_and_paras(pdf_path, n_pages=pages_count)
    dynamic_zones = calculate_dynamic_zones(all_lines, pages_count)
    # Removed: print(f"[INFO] Dynamic zones found: {dynamic_zones}")

    para_stats = analyze_paragraph_fonts(para_lines, all_lines)
    # Removed: if para_stats['is_fallback']: print(f"[WARN] No strong paragraph blocks found; using all_lines for stats.")
    # Removed: else: print(f"Found {para_stats['n_para_lines']} paragraph lines for stats.")

    # --- FINAL TITLE & HEADING FLOW (with nuanced Page 0 rules) ---

    # 1. Perform nuanced title detection on page 0
    title_text = ""
    title_block_lines = []
    page_0_lines = [line for line in all_lines if line['page'] == 0]

    if page_0_lines:
        # Define relaxed thresholds for grouping on Page 0 to capture main titles/subtitles
        MAX_Y_GAP_FACTOR = 2.5 # Allow slightly larger vertical gaps for titles/subtitles
        PAGE_0_MAX_SIZE_DROP_RATIO = 0.6 # Font size can't shrink to less than 60% of the line above
        PAGE_0_MIN_SIZE_THRESHOLD = para_stats['para_median'] * 1.2 # Must be at least 20% larger than body text
        TITLE_ZONE_Y_LIMIT = page_0_lines[0].get('page_h', 792) * 0.6 # Look in top 60% of the page

        # Candidates are lines within the title zone and not too short
        candidate_lines = [l for l in page_0_lines if l['top'] < TITLE_ZONE_Y_LIMIT and l['words'] > 0]

        if candidate_lines:
            # Start with the largest font size line as the potential main title
            candidate_lines.sort(key=lambda l: (-l["avg_size"], l["top"]))
            top_line = candidate_lines[0]

            # Ensure we're working with lines sorted by their actual position
            page_0_lines_by_pos = sorted([l for l in page_0_lines if l['words'] > 0], key=lambda l: l['top'])

            try:
                start_index = page_0_lines_by_pos.index(top_line)
            except ValueError:
                start_index = -1

            if start_index != -1:
                title_block_lines = [top_line]
                prev_line = top_line

                # Iterate through subsequent lines to build the title block
                for i in range(start_index + 1, len(page_0_lines_by_pos)):
                    current_line = page_0_lines_by_pos[i]

                    # --- Rules to stop accumulating title lines ---
                    # 1. Stop if the vertical gap is too large (indicates a new section)
                    dy_from_prev = current_line['top'] - prev_line['bottom']
                    if dy_from_prev > prev_line['avg_size'] * MAX_Y_GAP_FACTOR:
                        break

                    # 2. Stop if the font size is not 'title-like' (i.e., it's dropped to body text size or below threshold)
                    if current_line['avg_size'] < PAGE_0_MIN_SIZE_THRESHOLD:
                        break

                    # 3. Stop if the font size shrinks too drastically from the previous line (e.g., from title to body)
                    size_ratio = current_line['avg_size'] / prev_line['avg_size'] if prev_line['avg_size'] > 0 else 0
                    if size_ratio < PAGE_0_MAX_SIZE_DROP_RATIO:
                        break

                    # If all checks pass, this line is likely part of the title block
                    title_block_lines.append(current_line)
                    prev_line = current_line

        # Clean up unwanted text (dates, versions) from the end of the title block
        if title_block_lines:
            while title_block_lines:
                last_text = title_block_lines[-1]["text"]
                if (DATE_PATTERN.search(last_text) or VERSION_PATTERN.search(last_text)) and len(last_text.split()) < 5:
                    title_block_lines.pop()
                else:
                    break

            raw_title = " ".join(line["text"] for line in title_block_lines)
            title_text = deduplicate_title_text(raw_title) # Text is already normalized and collapsed

    # Removed: print(f"[INFO] Extracted Title: '{title_text}'")

    # 2. Prepare lines for heading detection (all lines MINUS the title block)
    title_line_coords = {(l['page'], l['top']) for l in title_block_lines}
    lines_for_heading_detection = [
        line for line in all_lines
        if (line['page'], line['top']) not in title_line_coords
    ]

    # 3. Run heading detection on the remaining lines from ALL pages
    accepted, rejected = detect_headings(
        lines_for_heading_detection,
        para_stats,
        dynamic_zones,
        table_bboxes_by_page
        # debug_print=debug_print # Removed
    )

    # Removed: print(f"\n[INFO] Initial classification on remaining lines: {len(accepted)} accepted, {len(rejected)} rejected.")

    grouped_headings = group_multiline_headings(accepted, all_lines)
    # Removed: print(f"[INFO] Grouped into {len(grouped_headings)} final headings.")

    cleaned_headings = clean_headings_for_output(grouped_headings)

    structured_outline = structure_outline_with_final_rules(cleaned_headings)

    final_structured_output = {
        "title": title_text,
        "outline": structured_outline
    }

    output_json = output_dir / f"{pdf_path.stem}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_structured_output, f, indent=4, ensure_ascii=False)
    # Removed: print(f"[SUCCESS] Saved structured headings to: {output_json}")


if __name__ == "__main__":
    # Define fixed input and output directories as per the challenge rules
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")

    # Ensure the output directory exists. This is crucial inside the container.
    output_dir.mkdir(exist_ok=True)

    # Set debug mode directly. For submission, you might set this to False.
    debug_mode = False # Keep this as False for submission

    # Find all PDF files in the mandatory input directory
    files_to_process = sorted(list(input_dir.rglob("*.pdf")))
    if not files_to_process:
        # Removed: print(f"[WARN] No PDF files found in {input_dir}")
        sys.exit(0)

    # Removed: print(f"[INFO] Found {len(files_to_process)} PDF(s) to process.")
    for idx, pdf_path in enumerate(files_to_process, 1):
        # Removed: print(f"\n--- [{idx}/{len(files_to_process)}] ---")
        try:
            # Pass the correct output directory to your processing function
            # Removed debug_print=debug_mode from here as process_pdf no longer uses it
            process_pdf(pdf_path, output_dir)
        except Exception as e:
            # Removed: print(f"[CRITICAL ERROR] Failed to process {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
