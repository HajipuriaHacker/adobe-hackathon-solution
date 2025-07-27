import pdfplumber
from statistics import median, mean
from collections import defaultdict, Counter
import sys
from pathlib import Path
import json
import re
import os
import camelot


# === Date/Version Patterns for Title Cleanup ===
DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s+\d{4})?\b"
)
VERSION_PATTERN = re.compile(
    r"\b(?:version|rev(?:ision)?)[\s:]\d+(\.\d+)\b", re.IGNORECASE
)


# --- Core Helper Functions ---


def is_bold(fontname):
    """Checks if a fontname string implies a bold weight."""
    return any(x in fontname.lower() for x in ['bold', 'black', 'heavy', 'ultrabold'])


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
    # Iterate from the center of the string outwards to find the longest possible repeated prefix
    for i in range(len(text) // 2, 3, -1): # Start from the middle, down to a min prefix of 4 chars
        prefix = text[:i]
        next_chunk = text[i:i*2]
        if prefix.strip() == next_chunk.strip() and prefix.strip():
            # Found a repeated prefix. Return the single prefix + the rest of the string.
            rest_of_string = text[i*2:].strip()
            return f"{prefix.strip()} {rest_of_string}".strip()
    return text # No repeated prefix found


def get_line_props(chars):
    """Calculates properties for a line of text based on its characters."""
    if not chars:
        return 0, '', 0, 0, 0, 0, [0, 0]
    
    size = round(median([c["size"] for c in chars]), 1)
    fontnames = [c.get("fontname", "") for c in chars]
    fontname = Counter(fontnames).most_common(1)[0][0] if fontnames else ''
    x0 = min(c['x0'] for c in chars)
    x1 = max(c['x1'] for c in chars)
    top = min(c['top'] for c in chars)
    bottom = max(c['bottom'] for c in chars)
    
    word_boundaries = []
    in_word = False
    for c in chars:
        if not c["text"].isspace():
            if not in_word:
                word_boundaries.append(c["x0"])
                if len(word_boundaries) == 2:
                    break
            in_word = True
        else:
            in_word = False
    while len(word_boundaries) < 2:
        word_boundaries.append(x0)
        
    return size, fontname, x0, x1, top, bottom, word_boundaries


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
            if 0.5 < factor < 3.0:
                gap_factors.append(factor)

    if gap_factors:
        median_gap_factor = median(gap_factors)
        dynamic_y_gap_factor = max(1.5, median_gap_factor * 1.5)
    else:
        dynamic_y_gap_factor = 1.7 # Fallback

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
                
                avg_size, fontname, x0, x1, top, bottom, x0_words = get_line_props(group)
                line = {
                    "page": idx, "text": text, "words": len(text.split()),
                    "chars": len(text), "avg_size": avg_size, "fontname": fontname,
                    "x0": x0, "x1": x1, "x0_words": x0_words, "top": top,
                    "bottom": bottom, "y": y, "page_h": page_h, "page_w": page_w
                }
                page_lines.append(line)
            
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
            if len(line['text'].strip()) <= 2: continue

            if len(line["text"].split()) < PARA_MIN_WORDS:
                para_buffer = flush_para_buffer(para_buffer, para_lines)
                prev_line = None
                continue

            if prev_line:
                dy = line["y"] - prev_line["y"]
                size_gap = abs(line["avg_size"] - prev_line["avg_size"])
                avg_x0_diff = abs(line["x0_words"][0] - prev_line["x0_words"][0])
                
                if dy > PARA_Y_GAP_FACTOR * line["avg_size"] or size_gap > 1.0 or avg_x0_diff > 40:
                    para_buffer = flush_para_buffer(para_buffer, para_lines)
            
            para_buffer.append(line)
            prev_line = line
            
        flush_para_buffer(para_buffer, para_lines)

    return all_lines, para_lines


def analyze_paragraph_fonts(para_lines, all_lines_fallback):
    source_lines = para_lines if para_lines else all_lines_fallback
    if not source_lines: return {
        "n_para_lines": 0, "para_median": 12, "para_mean": 12, "para_range": (12,12),
        "para_hist": [], "para_words_mean": 10, "is_fallback": True,
        "para_fontname_primary": "ArialMT"
    }

    para_sizes = [l['avg_size'] for l in source_lines]
    para_words = [l['words'] for l in source_lines]
    para_fontnames = [l['fontname'] for l in source_lines if l['fontname']]
    primary_font = Counter(para_fontnames).most_common(1)[0][0] if para_fontnames else "ArialMT"

    return {
        "n_para_lines": len(source_lines),
        "para_median": median(para_sizes) if para_sizes else 12,
        "para_mean": mean(para_sizes) if para_sizes else 12,
        "para_range": (min(para_sizes), max(para_sizes)) if para_sizes else (12, 12),
        "para_hist": Counter(para_sizes).most_common(5),
        "para_words_mean": mean(para_words) if para_words else 10,
        "para_fontname_primary": primary_font,
        "is_fallback": not bool(para_lines)
    }


def calculate_dynamic_zones(all_lines, pages_count):
    header_candidates, footer_candidates = [], []
    CANDIDATE_ZONE_RATIO = 0.20

    for line in all_lines:
        normalized_text = re.sub(r'\d+', '', line['text']).strip()
        if len(normalized_text) < 4: continue

        if line['top'] < CANDIDATE_ZONE_RATIO * line['page_h']:
            header_candidates.append({'normalized': normalized_text, 'bottom': line['bottom']})
        
        if line['top'] > (1 - CANDIDATE_ZONE_RATIO) * line['page_h']:
            footer_candidates.append({'normalized': normalized_text, 'top': line['top']})

    header_text_counts = Counter(cand['normalized'] for cand in header_candidates)
    footer_text_counts = Counter(cand['normalized'] for cand in footer_candidates)

    repeating_header_texts = {text for text, count in header_text_counts.items() if count > pages_count / 2}
    repeating_footer_texts = {text for text, count in footer_text_counts.items() if count > pages_count / 2}

    header_y, footer_y = None, None
    if repeating_header_texts:
        lowest_header_bottom = max(cand['bottom'] for cand in header_candidates if cand['normalized'] in repeating_header_texts)
        header_y = lowest_header_bottom + 5

    if repeating_footer_texts:
        highest_footer_top = min(cand['top'] for cand in footer_candidates if cand['normalized'] in repeating_footer_texts)
        footer_y = highest_footer_top - 5
        
    return {"header_y": header_y, "footer_y": footer_y}


# --- Heading Classification Logic ---


def is_numbery(text):
    s = text.strip()
    return any([
        s.isdigit(), re.fullmatch(r'\d+[,.]?\d*', s), re.fullmatch(r'[-\d]+', s),
        re.fullmatch(r'\d{1,2}/\d{1,2}/\d{2,4}', s), re.fullmatch(r'[A-Z]\d+', s),
        re.fullmatch(r'\d+(st|nd|rd|th)', s)
    ])


def is_likely_list_item(line, page_lines, para_stats):
    """
    Checks if a line is likely a list item. This version is more careful
    not to misclassify styled headings.
    """
    # --- NEW EXCEPTION RULE ---
    # If the line is bold and the paragraph font is not, it's a heading, not a list item.
    is_line_bold = is_bold(line.get('fontname', ''))
    is_para_font_bold = is_bold(para_stats.get("para_fontname_primary", ""))
    if is_line_bold and not is_para_font_bold:
        return False

    # Original logic remains, but is now secondary to the style check
    if line['avg_size'] > para_stats['para_range'][1] * 1.1 or line['words'] >= para_stats['para_words_mean']:
        return False
    try:
        current_index = page_lines.index(line)
    except (ValueError, KeyError):
        return False # Cannot determine context if not found
             
    def is_similar(l1, l2):
        if abs(l1['avg_size'] - l2['avg_size']) > 0.5: return False
        if abs(l1['x0'] - l2['x0']) > 15: return False
        if l2['words'] >= para_stats['para_words_mean']: return False
        return True
    
    similar_before, similar_after = 0, 0
    for i in range(1, 3):
        if current_index - i >= 0 and is_similar(line, page_lines[current_index - i]):
            similar_before += 1
        if current_index + i < len(page_lines) and is_similar(line, page_lines[current_index + i]):
            similar_after += 1
            
    return (similar_before >= 1 and similar_after >= 1) or similar_before >= 2 or similar_after >= 2


def heading_format_ok(text):
    """Enhanced check to reject noisy or non-semantic headings."""
    text = text.strip()
    
    # Rule 1: Basic format checks (empty, starts with list-like chars, ends with punctuation)
    if not text or text.startswith(('•', '-', '*')) or text.endswith(('.', ';', ',')):
        return False
        
    # Rule 2: Reject if all lowercase (and contains at least one letter)
    if text.islower() and any(c.isalpha() for c in text):
        return False
        
    # Rule 3: Reject if it looks like a number or simple numeric code
    if re.fullmatch(r'\d+', text) or re.match(r'^\d+\.\d+', text):
        return False

    # Rule 4: Reject if the entire line is enclosed in brackets
    if re.fullmatch(r'\(.\)', text) or re.fullmatch(r'\[.\]', text):
        return False
        
    # Rule 5: Reject if the line is likely just a date
    # Check if the line is very short and contains a month name
    if len(text.split()) <= 3 and DATE_PATTERN.search(text):
        return False
        
    return True


def classify_line(line, para_stats, page_lines, dynamic_zones, table_bboxes_by_page):
    """
    Applies a sequence of rules with precedence to classify a line as a heading.
    """
    text = collapse_fake_bold(line['text'].strip())
    if not text: return False, "Empty line"

    # --- SECTION 1: ABSOLUTE PRE-CHECKS ---
    if is_in_table(line, table_bboxes_by_page):
        return False, "Rejected: Located inside a detected table"
        
    header_y, footer_y = dynamic_zones.get("header_y"), dynamic_zones.get("footer_y")
    if header_y and line['top'] < header_y: return False, f"In dynamic header zone (y < {header_y:.0f})"
    if footer_y and line['bottom'] > footer_y: return False, f"In dynamic footer zone (y > {footer_y:.0f})"
    if is_numbery(text) and not re.match(r'^\d+(\.\d+)*\.\s', text): return False, f"Rejected: Is numbery ('{text}')"

    if line['top'] < line['page_h'] * 0.15: 
        if line['avg_size'] <= para_stats['para_range'][1]:
            is_line_bold = is_bold(line['fontname']) and not is_bold(para_stats.get("para_fontname_primary", ""))
            if not is_line_bold:
                return False, "Rejected: Normal font size at top of page"

    # --- SECTION 2: STRONG ACCEPTANCE RULES ---
    if line['avg_size'] > para_stats['para_range'][1] * 1.1:
        return (True, f"Accepted: Oversize font ({line['avg_size']:.1f} > {para_stats['para_range'][1]:.1f})") if heading_format_ok(text) else (False, "Rejected: Oversize but bad format")
    
    is_line_bold, is_para_font_bold = is_bold(line['fontname']), is_bold(para_stats.get("para_fontname_primary", ""))
    if is_line_bold and not is_para_font_bold:
        return (True, f"Accepted: Bold font ('{line['fontname']}')") if heading_format_ok(text) else (False, "Rejected: Bold but bad format")

    # --- SECTION 3: STRONG REJECTION RULES ---
    if ':' in text:
        parts = text.split(':', 1)
        if len(parts) > 1 and parts[1].strip() and (parts[1].strip()[0].islower() or parts[1].strip()[0].isdigit()):
            return False, "Rejected: Key-value pair"

    # --- SECTION 4: CONTEXT-SENSITIVE RULES ---
    if line['words'] <= para_stats['para_words_mean'] * 0.7:
        if is_likely_list_item(line, page_lines, para_stats): return False, "Rejected: Short line, but likely a list item"
        if (line['above_ws'] + line['below_ws']) > line['avg_size'] * 2.5 and heading_format_ok(text):
            return True, "Accepted: Short line with extra whitespace"

    return False, "Rejected: Resembles paragraph text"


# --- Main Processing and Grouping ---


def is_in_table(line, table_bboxes_by_page):
    """Checks if a line is within any of the detected table bounding boxes."""
    page_num = line['page']
    if page_num in table_bboxes_by_page:
        line_v_center = (line['top'] + line['bottom']) / 2
        line_h_center = (line['x0'] + line['x1']) / 2
        for bbox in table_bboxes_by_page[page_num]:
            is_vertically_inside = bbox[1] <= line_v_center <= bbox[3]
            is_horizontally_inside = bbox[0] <= line_h_center <= bbox[2]
            if is_vertically_inside and is_horizontally_inside:
                return True
    return False


def detect_headings(all_lines, para_stats, dynamic_zones, table_bboxes_by_page, debug=False):
    accepted, rejected = [], []
    lines_by_page = defaultdict(list)
    for l in all_lines: lines_by_page[l['page']].append(l)

    for page_idx, page_lines in lines_by_page.items():
        for line in page_lines:
            is_heading, reason = classify_line(line, para_stats, page_lines, dynamic_zones, table_bboxes_by_page)
            line_info = {**line, "why": reason, "text": collapse_fake_bold(line['text'])}
            (accepted if is_heading else rejected).append(line_info)
            if debug: print(f"[{'ACCEPTED' if is_heading else 'REJECTED'}] Pg{line['page']}: {line_info['text'][:60]!r} | {reason}")

    return accepted, rejected


def group_multiline_headings(headings, all_lines):
    if not headings: return []
    grouped, used_indices = [], set()
    heading_coords = {(h['page'], h['top']) for h in headings}
    all_ws = [h['above_ws'] for h in headings] + [h['below_ws'] for h in headings]
    avg_ws = mean(all_ws) if all_ws else 0

    headings.sort(key=lambda h: (h['page'], h['top'])) 

    for i, curr in enumerate(headings):
        if i in used_indices: continue
        group = [curr]
        for j in range(i + 1, len(headings)):
            if j in used_indices: continue
            candidate = headings[j]
            if any([candidate['page'] != curr['page'], candidate['fontname'] != curr['fontname'],
                    candidate['avg_size'] != curr['avg_size'], abs(candidate['x0'] - curr['x0']) > 10]): break
            
            intervening = [l for l in all_lines if l['page'] == curr['page'] and curr['bottom'] < l['top'] < candidate['top']]
            if intervening and any((l['page'], l['top']) not in heading_coords for l in intervening): break
            if candidate['above_ws'] > 0.7 * avg_ws and avg_ws > 1: break
            
            group.append(candidate)
            used_indices.add(j)

        final_heading = {**group[0], "text": " ".join(h['text'] for h in group), "bottom": group[-1]['bottom'], "x1": max(h['x1'] for h in group)}
        grouped.append(final_heading)
        used_indices.add(i)
    return grouped


def clean_headings_for_output(grouped_headings):
    """
    Cleans up the grouped headings and ensures all necessary keys,
    including 'page_w', are preserved for the final structuring step.
    """
    return [{
        "page": h.get('page'), 
        "text": h.get('text', ''), 
        "words": len(h.get('text', '').split()),
        "chars": len(h.get('text', '')), 
        "avg_size": h.get('avg_size'), 
        "fontname": h.get('fontname'),
        "avg_ws": round(mean([h.get('above_ws', 0), h.get('below_ws', 0)]), 2),
        "x0": h.get('x0'), 
        "x1": h.get('x1'), 
        "top": h.get('top'), 
        "bottom": h.get('bottom'),
        "page_h": h.get('page_h'),
        "page_w": h.get('page_w')  # <-- THE FIX: This key is now included.
    } for h in grouped_headings]



def structure_outline_with_final_rules(headings):
    """
    This is the definitive structuring function. It uses a base font ranking
    and then applies a series of specific, user-defined rules for H1 subtitles,
    numbered headings, and conservative H4 assignment.
    """
    if not headings:
        return []

    # 1. Sort all headings by their position in the document
    outline_headings = sorted(headings, key=lambda h: (h['page'], h['top']))

    # 2. Establish a font-size-to-rank mapping (uncapped).
    all_heading_sizes = sorted(list(set(h.get('avg_size', 0) for h in outline_headings)), reverse=True)
    level_map = {size: i + 1 for i, size in enumerate(all_heading_sizes)}

    # 3. Assign an initial, uncapped level to all headings
    for h in outline_headings:
        h['level'] = level_map.get(h.get('avg_size', 0), 99) # Default to a high number

    # 4. Perform a final correction pass using the new rules
    corrected_headings = []
    if not outline_headings:
        return []
        
    corrected_headings.append(outline_headings[0])

    for i in range(1, len(outline_headings)):
        prev_h = corrected_headings[-1]
        curr_h = outline_headings[i]
        
        # --- RULE 1: Force Numbered Headings to H3 (Strong Override) ---
        # THIS IS THE CORRECTED REGEX: It now accepts a space OR a dot after the number.
        is_numbered_heading = re.match(r'^\d+(\.\d+)*[\.\s]', curr_h.get('text', ''))
        if is_numbered_heading:
            curr_h['level'] = 3
        
        # --- RULE 2: H1 Subtitle Promotion ---
        if prev_h.get('level') == 1 and curr_h.get('page') == prev_h.get('page'):
            is_close_vertically = (curr_h.get('top', 0) - prev_h.get('bottom', 0)) < (prev_h.get('avg_size', 12) * 3)
            is_top_tier_font = len(all_heading_sizes) > 1 and curr_h.get('avg_size', 0) >= all_heading_sizes[1]
            if is_close_vertically and is_top_tier_font:
                curr_h['level'] = 1 # Promote to H1
        
        # --- RULE 3: Conservative H4 Assignment ---
        is_provisional_h4_or_lower = level_map.get(curr_h.get('avg_size', 0), 99) >= 4
        if is_provisional_h4_or_lower and prev_h.get('level') == 3:
            if curr_h.get('avg_size', 0) >= prev_h.get('avg_size', 0) * 0.85:
                curr_h['level'] = 3 # Promote to H3
        
        # --- Final Hierarchy Rule ---
        if curr_h.get('level', 99) > prev_h.get('level', 99) + 1:
            curr_h['level'] = prev_h.get('level', 99) + 1
            
        corrected_headings.append(curr_h)

    # 5. Final Pass: Apply the H4 Cap and Format for Output
    structured_outline = []
    for h in corrected_headings:
        final_level = min(h['level'], 4) # Cap the level at H4
        structured_outline.append({
            "level": f"H{final_level}",
            "text": h['text'].strip(),
            "page": h['page']
        })
    
    return structured_outline










def process_pdf(pdf_path, output_dir, debug_print=False):
    """Main pipeline to process a single PDF file."""
    print(f"\n[INFO] Processing: {pdf_path}")

    table_bboxes_by_page = defaultdict(list)
    pages_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_count = len(pdf.pages)
            
            tables = camelot.read_pdf(str(pdf_path), pages=f'1-{pages_count}', flavor='lattice', suppress_stdout=True)
            print(f"[INFO] Camelot found {tables.n} tables in total.")
            
            for table in tables:
                page_num = table.page - 1
                if page_num < pages_count:
                    x0, y1_camelot, x1, y2_camelot = table._bbox
                    page_height = pdf.pages[page_num].height
                    top, bottom = page_height - y2_camelot, page_height - y1_camelot
                    table_bboxes_by_page[page_num].append((x0, top, x1, bottom))
                    
    except Exception as e:
        print(f"[WARN] Camelot failed or PDF is unscannable: {e}. Continuing without table detection.")
        if pages_count == 0:
            with pdfplumber.open(pdf_path) as pdf:
                pages_count = len(pdf.pages)

    all_lines, para_lines = collect_lines_and_paras(pdf_path, n_pages=pages_count)
    dynamic_zones = calculate_dynamic_zones(all_lines, pages_count)
    print(f"[INFO] Dynamic zones found: {dynamic_zones}")
    
    para_stats = analyze_paragraph_fonts(para_lines, all_lines)
    if para_stats['is_fallback']:
        print(f"[WARN] No paragraph blocks found...")
    else:
        print(f"Found {para_stats['n_para_lines']} paragraph lines...")

    # --- FINAL TITLE & HEADING FLOW (with relaxed Page 0 rules) ---

    # 1. Perform nuanced title detection on page 0
    title_text = ""
    title_block_lines = []
    page_0_lines = [line for line in all_lines if line['page'] == 0]

    if page_0_lines:
        # Define relaxed thresholds for grouping on Page 0
        MAX_Y_GAP_FACTOR = 2.5  # Allow slightly larger vertical gaps for subtitles
        PAGE_0_MAX_SIZE_DROP_RATIO = 0.6 # Font size can't shrink to less than 60% of the line above
        PAGE_0_MIN_SIZE_THRESHOLD = para_stats['para_median'] * 1.2 # Must be at least 20% larger than body text
        TITLE_ZONE_Y_LIMIT = page_0_lines[0].get('page_h', 792) * 0.6 # Look in top 60%

        candidate_lines = [l for l in page_0_lines if l['top'] < TITLE_ZONE_Y_LIMIT and l['words'] > 0]
        
        if candidate_lines:
            candidate_lines.sort(key=lambda l: (-l["avg_size"], l["top"]))
            top_line = candidate_lines[0]
            
            page_0_lines_by_pos = sorted([l for l in page_0_lines if l['words'] > 0], key=lambda l: l['top'])
            
            try:
                start_index = page_0_lines_by_pos.index(top_line)
            except ValueError:
                start_index = -1

            if start_index != -1:
                title_block_lines = [top_line]
                prev_line = top_line

                for i in range(start_index + 1, len(page_0_lines_by_pos)):
                    current_line = page_0_lines_by_pos[i]
                    
                    # --- YOUR NEW, RELAXED PAGE 0 RULES ---
                    # 1. Stop if the vertical gap is too large (indicates a new section)
                    dy_from_prev = current_line['top'] - prev_line['bottom']
                    if dy_from_prev > prev_line['avg_size'] * MAX_Y_GAP_FACTOR:
                        break

                    # 2. Stop if the font size is not 'title-like' (i.e., it's body text size)
                    if current_line['avg_size'] < PAGE_0_MIN_SIZE_THRESHOLD:
                        break
                        
                    # 3. Stop if the font size shrinks too drastically from the previous line
                    size_ratio = current_line['avg_size'] / prev_line['avg_size'] if prev_line['avg_size'] > 0 else 0
                    if size_ratio < PAGE_0_MAX_SIZE_DROP_RATIO:
                        break

                    # If all checks pass, this line is part of the title block
                    title_block_lines.append(current_line)
                    prev_line = current_line
        
        if title_block_lines:
            while title_block_lines:
                last_text = title_block_lines[-1]["text"]
                if (DATE_PATTERN.search(last_text) or VERSION_PATTERN.search(last_text)) and len(last_text.split()) < 5:
                    title_block_lines.pop()
                else:
                    break
            
            raw_title = " ".join(line["text"] for line in title_block_lines)
            # Use the new helper function to clean the final string
            title_text = deduplicate_title_text(collapse_fake_bold(raw_title).strip())

    print(f"[INFO] Extracted Title: '{title_text}'")

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
        table_bboxes_by_page, 
        debug=debug_print
    )

    # This print statement now reflects that it processes lines from all pages
    print(f"\n[INFO] Initial classification on remaining lines: {len(accepted)} accepted, {len(rejected)} rejected.")
    
    grouped_headings = group_multiline_headings(accepted, all_lines)
    print(f"[INFO] Grouped into {len(grouped_headings)} final headings.")
    
    cleaned_headings = clean_headings_for_output(grouped_headings)
    
    # In process_pdf():
    structured_outline = structure_outline_with_final_rules(cleaned_headings)


    
    final_structured_output = {
        "title": title_text,
        "outline": structured_outline
    }
    
    output_json = output_dir / f"{pdf_path.stem}.json" 
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_structured_output, f, indent=4, ensure_ascii=False)
    print(f"[SUCCESS] Saved structured headings to: {output_json}")


if __name__ == "__main__":
    # Define fixed input and output directories as per the challenge rules
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")

    # Ensure the output directory exists. This is crucial inside the container.
    output_dir.mkdir(exist_ok=True)

    # Set debug mode directly. For submission, you might set this to False.
    # Alternatively, you could use an environment variable.
    debug_mode = False 

    # Find all PDF files in the mandatory input directory
    files_to_process = sorted(list(input_dir.rglob("*.pdf")))
    if not files_to_process:
        print(f"[WARN] No PDF files found in {input_dir}")
        sys.exit(0)

    print(f"[INFO] Found {len(files_to_process)} PDF(s) to process.")
    for idx, pdf_path in enumerate(files_to_process, 1):
        print(f"\n--- [{idx}/{len(files_to_process)}] ---")
        try:
            # Pass the correct output directory to your processing function
            process_pdf(pdf_path, output_dir, debug_print=debug_mode)
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to process {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()