"""
syllabus_parser.py — Campus-Wide GraphRAG Syllabus Ingestion Pipeline

Four-phase pipeline:
  Phase 1: Fast text-only pass → page-type classification (~1s for 300+ pages)
  Phase 2: Targeted table pass → only pages that need table parsing (~7-8s)
  Phase 3: Structural extraction → regex + rule-based, no LLM
  Phase 4: Neo4j MERGE writes → semantic + source layers (batched)
"""

import hashlib
import re
import time

import pymupdf as fitz  # PyMuPDF

# Page Type Constants

PAGE_COVER = "COVER"
PAGE_GENERAL_INFO = "GENERAL_INFO"
PAGE_SEMESTER_TABLE = "SEMESTER_TABLE"
PAGE_COURSE_DETAIL = "COURSE_DETAIL"
PAGE_ELECTIVE = "ELECTIVE_BASKET"
PAGE_RULE = "RULE"
PAGE_OTHER = "OTHER"

# Patterns compiled once at module level
_COURSE_HEADER_RE = re.compile(
    r"^(\d{2}[A-Z]{3}\d{3})\s+(.*?)\s+L-T-P-C\s*:\s*([\d\[\]-]+)-([\d\[\]-]+)-([\d\[\]-]+)-([\d\[\]-]+)",
    re.IGNORECASE | re.MULTILINE,
)
_COURSE_CODE_RE = re.compile(r"\b\d{2}[A-Z]{3}\d{3}\b", re.IGNORECASE)
_UNIT_RE = re.compile(r"(?:Syllabus\s*)?Unit\s*([IVX]+|\d+)", re.IGNORECASE)
_PO_LINE_RE = re.compile(r"^(PO\d+)\s*[:\-]\s*(.+)", re.IGNORECASE | re.MULTILINE)
_PSO_LINE_RE = re.compile(r"^(PSO\d+)\s*[:\-]\s*(.+)", re.IGNORECASE | re.MULTILINE)
_PEO_BULLET_RE = re.compile(r"^[•\-\*]\s*(.+)", re.MULTILINE)
_CO_LINE_RE = re.compile(r"^(CO\d+)\s*[:\-]\s*(.+)", re.IGNORECASE | re.MULTILINE)
_PREREQ_RE = re.compile(r"Pre[-\s]*Requisite[s]?\s*[:\-]\s*(.+)", re.IGNORECASE)
_LTTPC_RE = re.compile(r"([\d\[\]]+)[-\s]([\d\[\]]+)[-\s]([\d\[\]]+)[-\s]([\d\[\]]+)")
_EVAL_RATIO_RE = re.compile(r"(\d{2})\s*:\s*(\d{2})")
_SEMESTER_HDR_RE = re.compile(
    r"Semester\s*(I{1,3}V?|VI{0,3}|IX|X{0,2}I{0,3}|\d+)", re.IGNORECASE
)
_BASKET_HDR_RE = re.compile(
    r"(?:Electives?\s+in|Professional\s+Electives?\s*[-–]?|Science\s+Electives?|"
    r"Free\s+Electives?|General\s+Electives?|Amrita\s+Value|Mandatory\s+Course)",
    re.IGNORECASE,
)
_STOP_SECTION_RE = re.compile(
    r"^(?:TEXTBOOK|TEXT\s*BOOK|REFERENCE|ESSENTIAL\s+READING|Evaluation\s+Pattern|"
    r"Assessment|TEXTBOOKS|REFERENCES)\b",
    re.IGNORECASE | re.MULTILINE,
)
_JUNK_PATTERNS = [
    re.compile(r"\d{4}"),
    re.compile(r"publishers?", re.IGNORECASE),
    re.compile(r"\bedition\b", re.IGNORECASE),
    re.compile(r"\bpress\b", re.IGNORECASE),
    re.compile(r"\bISBN\b", re.IGNORECASE),
    re.compile(r"Text\s*Book", re.IGNORECASE),
    re.compile(r"\bReference\b", re.IGNORECASE),
    re.compile(r"Evaluation\s+Pattern", re.IGNORECASE),
    re.compile(r"\bAssessment\b", re.IGNORECASE),
    re.compile(r"McGraw|Pearson|Wiley|Springer|Elsevier|Tata|Jones", re.IGNORECASE),
]


def _roman_to_int(s: str) -> int:
    roman = {"I": 1, "V": 5, "X": 10}
    s = s.upper().strip()
    try:
        return int(s)
    except ValueError:
        pass
    result, prev = 0, 0
    for ch in reversed(s):
        val = roman.get(ch, 0)
        if val < prev:
            result -= val
        else:
            result += val
        prev = val
    return result if result > 0 else 1


def _is_junk_topic(text: str) -> bool:
    for pat in _JUNK_PATTERNS:
        if pat.search(text):
            return True
    return False


def _sv_id(dept: str, year: str) -> str:
    return f"{dept.upper()}_{year}"


# Phase 1 — Fast Page Pass


def _classify_page(page_num: int, text: str) -> str:
    if page_num == 0:
        return PAGE_COVER
    tu = text.upper()
    if any(
        k in tu
        for k in [
            "GENERAL INFORMATION",
            "ABBREVIATIONS USED",
            "PROGRAM OUTCOME",
            "PROGRAM EDUCATIONAL OBJECTIVE",
            "PROGRAM SPECIFIC OUTCOME",
            "GRADUATE WILL",
            "PROGRAM SPECIFIC OUTCOMES",
        ]
    ):
        return PAGE_GENERAL_INFO
    if _BASKET_HDR_RE.search(text):
        return PAGE_ELECTIVE
    if _COURSE_HEADER_RE.search(text):
        return PAGE_COURSE_DETAIL
    if _SEMESTER_HDR_RE.search(text) and _COURSE_CODE_RE.search(text):
        return PAGE_SEMESTER_TABLE
    if any(
        k in tu
        for k in [
            "NPTEL",
            "CGPA",
            "LIVE-IN",
            "LIVE IN LABS",
            "EXEMPTION",
            "ELIGIBILITY",
            "CREDIT LIMIT",
            "SPECIALIZATION",
        ]
    ):
        return PAGE_RULE
    return PAGE_OTHER


def phase1_fast_pass(doc):
    page_texts = []
    page_type_idx = {}
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        page_texts.append(text)
        page_type_idx[i] = _classify_page(i, text)
    return page_texts, page_type_idx


# Phase 2 — Targeted Table Pass

# Only run find_tables() on pages where structural tables are critical
# (not COURSE_DETAIL — those need per-course inline detection for CO-PO)
_TABLE_PAGES = {PAGE_SEMESTER_TABLE, PAGE_GENERAL_INFO}


def phase2_table_pass(doc, page_type_idx: dict):
    """
    Run find_tables() ONLY on pages that need table data for structural parsing:
    - SEMESTER_TABLE: for curriculum overview tables
    - GENERAL_INFO: for framework/PSO tables
    COURSE_DETAIL and ELECTIVE pages use inline per-course table detection.
    """
    page_tables = {}
    for page_num, ptype in page_type_idx.items():
        if ptype not in _TABLE_PAGES:
            continue
        try:
            found = doc[page_num].find_tables()
            if found and found.tables:
                page_tables[page_num] = [t.extract() for t in found.tables]
        except Exception as e:
            print(f"[Parser] Table detection failed on page {page_num + 1}: {e}")
    return page_tables


# Keep a reference to the live doc for inline CO-PO table detection
_live_doc = None
# Per-page table cache (reset each parse run to avoid stale data)
_page_table_cache: dict = {}


def set_live_doc(doc):
    global _live_doc, _page_table_cache
    _live_doc = doc
    _page_table_cache = {}  # Reset cache on new doc


def _get_page_tables_inline(page_num: int) -> list:
    """Get tables for a single page, using cache to avoid duplicate find_tables() calls."""
    if page_num in _page_table_cache:
        return _page_table_cache[page_num]
    result = []
    if _live_doc is not None:
        try:
            found = _live_doc[page_num].find_tables()
            if found and found.tables:
                result = [t.extract() for t in found.tables]
        except Exception:
            pass
    _page_table_cache[page_num] = result
    return result


# Phase 3 — Structural Extraction helpers


def _extract_section(text: str, start_markers: list, end_markers: list) -> str:
    start_idx = -1
    for marker in start_markers:
        idx = text.find(marker)
        if idx >= 0:
            start_idx = idx + len(marker)
            break
    if start_idx < 0:
        return ""
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker, start_idx)
        if 0 <= idx < end_idx:
            end_idx = idx
    return text[start_idx:end_idx].strip()


def _parse_general_info(page_texts, page_type_idx, page_tables) -> dict:
    result = {
        "abbreviations": {},
        "program_outcomes": [],
        "program_educational_objectives": [],
        "program_specific_outcomes": [],
    }
    gi_pages = [i for i, t in page_type_idx.items() if t == PAGE_GENERAL_INFO]
    combined = "\n".join(page_texts[i] for i in gi_pages)

    # Abbreviations
    abbrev_re = re.compile(r"^([A-Z]{2,6})\s*[:\-]\s*(.+)", re.MULTILINE)
    for m in abbrev_re.finditer(combined):
        abbr, desc = m.group(1).strip(), m.group(2).strip()
        if len(abbr) >= 2 and len(desc) > 3:
            result["abbreviations"][abbr] = desc

    # POs via "PO1: ..." pattern
    for m in _PO_LINE_RE.finditer(combined):
        result["program_outcomes"].append(
            {"id": m.group(1).upper(), "description": m.group(2).strip()}
        )

    # Fallback: numbered list under "PROGRAM OUTCOMES"
    if not result["program_outcomes"]:
        blocks = combined.split("\n")
        collecting, counter = False, 1
        for line in blocks:
            line = line.strip()
            if (
                "PROGRAM OUTCOMES" in line.upper()
                or "PROGRAM OUTCOME (PO)" in line.upper()
            ):
                collecting = True
                continue
            if collecting:
                if (
                    "PROGRAM EDUCATIONAL" in line.upper()
                    or "PROGRAM SPECIFIC" in line.upper()
                ):
                    collecting = False
                    continue
                nm = re.match(r"^\d+\.\s+(.+)", line)
                if nm:
                    result["program_outcomes"].append(
                        {"id": f"PO{counter}", "description": nm.group(1).strip()}
                    )
                    counter += 1

    # PSOs
    for m in _PSO_LINE_RE.finditer(combined):
        result["program_specific_outcomes"].append(
            {"id": m.group(1).upper(), "description": m.group(2).strip()}
        )

    # PEOs
    peo_block = ""
    if "PROGRAM EDUCATIONAL OBJECTIVES" in combined.upper():
        idx = combined.upper().find("PROGRAM EDUCATIONAL OBJECTIVES")
        end = combined.upper().find("PROGRAM SPECIFIC OUTCOME", idx)
        peo_block = combined[idx : end if end > 0 else idx + 2000]
    for i, m in enumerate(_PEO_BULLET_RE.finditer(peo_block), start=1):
        desc = m.group(1).strip()
        if len(desc) > 10:
            result["program_educational_objectives"].append(
                {"id": f"PEO{i}", "description": desc}
            )

    return result


# Semester row pattern: "CAT  CODE  TITLE  L-T-P  CREDIT  EVAL"
# The PDF uses whitespace-separated columns — course code is the reliable anchor
_SEM_COURSE_ROW_RE = re.compile(
    r"^([A-Z/]+)\s+(\d{2}[A-Z]{3}\d{3})\s+(.+?)\s+(\d+[\s\-]\d+[\s\-]\d+)\s+(\d+)\s*(\d+:\d+)?",
    re.MULTILINE,
)
_SEM_ELECTIVE_ROW_RE = re.compile(
    r"^([A-Z/]+)\s+Professional Elective|Open Elective|Science Elective|Free Elective",
    re.IGNORECASE | re.MULTILINE,
)
_TOTAL_CREDITS_RE = re.compile(
    r"Total\s*[\(\[]?.*?=\s*(\d+)\s*(?:hrs|credits)?", re.IGNORECASE
)


def _parse_semester_tables(page_texts, page_type_idx, page_tables) -> list:
    """
    Parse semester course lists from raw page text (not tables).
    The PDF curriculum table uses whitespace-separated columns so we parse
    directly from text: each line has Cat Code Title L-T-P Credits EvalPattern.
    """
    semesters = {}

    # Collect the full curriculum text (pages 3-15 roughly)
    # Only look on SEMESTER_TABLE pages
    sem_pages = sorted(
        [i for i, t in page_type_idx.items() if t == PAGE_SEMESTER_TABLE]
    )
    # Also include pages 2-12 as fallback (the curriculum is in the first ~15 pages)
    extra_pages = [
        i
        for i in range(2, min(15, len(page_texts)))
        if page_type_idx.get(i) not in {PAGE_GENERAL_INFO, PAGE_COURSE_DETAIL}
    ]
    check_pages = sorted(set(sem_pages + extra_pages))

    for page_num in check_pages:
        text = page_texts[page_num]
        lines = text.split("\n")

        current_sem = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Detect semester header
            sem_m = _SEMESTER_HDR_RE.match(line_stripped)
            if sem_m:
                sem_label = sem_m.group(1)
                sem_num = _roman_to_int(sem_label)
                if sem_num not in semesters:
                    semesters[sem_num] = {
                        "number": sem_num,
                        "label": f"Semester {sem_label.upper() if not sem_label.isdigit() else sem_label}",
                        "courses": [],
                        "elective_slots": [],
                        "total_credits": None,
                    }
                current_sem = semesters[sem_num]
                continue

            if current_sem is None:
                continue

            # Detect total credits line
            tot_m = _TOTAL_CREDITS_RE.search(line_stripped)
            if tot_m and current_sem["total_credits"] is None:
                try:
                    current_sem["total_credits"] = int(tot_m.group(1))
                except Exception:
                    pass
                continue

            # Detect course code in line
            code_m = _COURSE_CODE_RE.search(line_stripped)
            if code_m:
                code = code_m.group(0).upper()
                # Grab the category from start of line
                cat = line_stripped[: code_m.start()].strip()
                # Grab the rest of the line after the code
                rest = line_stripped[code_m.end() :].strip()

                # Title: everything before the L-T-P pattern
                ltp_m = re.search(r"(\d+[-\s]\d+[-\s]\d+)", rest)
                title = rest[: ltp_m.start()].strip() if ltp_m else rest
                # Multi-line title: sometimes next line continues title
                if ltp_m is None and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not _COURSE_CODE_RE.search(next_line):
                        ltp_m2 = re.search(r"(\d+[-\s]\d+[-\s]\d+)", next_line)
                        if ltp_m2:
                            title = title + " " + next_line[: ltp_m2.start()].strip()
                            ltp_m = ltp_m2
                            rest = next_line[ltp_m2.start() :]

                # Credits: number after L-T-P
                cred = ""
                eval_ = ""
                if ltp_m:
                    after_ltp = rest[ltp_m.end() :].strip() if ltp_m else ""
                    parts = after_ltp.split()
                    if parts:
                        cred = parts[0]
                    if len(parts) >= 2:
                        eval_ = parts[1]
                    # Also check for eval ratio
                    er = _EVAL_RATIO_RE.search(after_ltp)
                    if er:
                        eval_ = f"{er.group(1)}:{er.group(2)}"

                if not any(c["code"] == code for c in current_sem["courses"]):
                    current_sem["courses"].append(
                        {
                            "code": code,
                            "title": title,
                            "category": cat,
                            "credits": cred,
                            "evaluation_pattern_ratio": eval_,
                        }
                    )
                continue

            # Elective slot lines (no code)
            if any(
                k in line_stripped
                for k in [
                    "Professional Elective",
                    "Open Elective",
                    "Science Elective",
                    "Free Elective",
                    "Amrita Value Programme",
                ]
            ):
                current_sem["elective_slots"].append(
                    {
                        "label": line_stripped,
                        "category": "",
                        "credits": "",
                        "evaluation_pattern_ratio": "",
                    }
                )

    return sorted(semesters.values(), key=lambda s: s["number"])


def _parse_elective_baskets(page_texts, page_type_idx, page_tables) -> list:
    baskets = {}
    for page_num, ptype in page_type_idx.items():
        if ptype != PAGE_ELECTIVE:
            continue
        text = page_texts[page_num]
        tables = page_tables.get(page_num, [])

        current_basket = None
        for line in text.split("\n"):
            bm = _BASKET_HDR_RE.search(line)
            if bm:
                basket_name = line.strip()
                if basket_name not in baskets:
                    baskets[basket_name] = {"name": basket_name, "courses": []}
                current_basket = basket_name

        for table in tables:
            if not table or len(table) < 2:
                continue
            header = [str(c or "").strip().lower() for c in (table[0] or [])]
            code_col = next((i for i, h in enumerate(header) if "code" in h), None)
            title_col = next(
                (i for i, h in enumerate(header) if "subject" in h or "title" in h),
                None,
            )
            cat_col = next((i for i, h in enumerate(header) if "cat" in h), None)
            credit_col = next((i for i, h in enumerate(header) if "credit" in h), None)
            eval_col = next(
                (i for i, h in enumerate(header) if "eval" in h or "pattern" in h), None
            )

            if code_col is None and title_col is None:
                continue

            for row in table[1:]:
                if not row:
                    continue
                cells = [str(c or "").strip() for c in row]
                max_col = max(x for x in [code_col, title_col] if x is not None)
                if len(cells) <= max_col:
                    continue
                code = cells[code_col].strip() if code_col is not None else ""
                title = cells[title_col].strip() if title_col is not None else ""
                cat = cells[cat_col].strip() if cat_col is not None else ""
                cred = cells[credit_col].strip() if credit_col is not None else ""
                eval_ = cells[eval_col].strip() if eval_col is not None else ""

                if not _COURSE_CODE_RE.match(code):
                    continue

                entry = {
                    "code": code.upper(),
                    "title": title,
                    "category": cat,
                    "credits": cred,
                    "evaluation_pattern_ratio": eval_,
                }

                if current_basket and current_basket in baskets:
                    baskets[current_basket]["courses"].append(entry)
                elif baskets:
                    list(baskets.values())[-1]["courses"].append(entry)

    return list(baskets.values())


def _parse_copo_table(tables: list) -> list:
    mappings = []
    for table in tables:
        if not table or len(table) < 2:
            continue
        header_row_idx = None
        header_cols = {}
        for ri, row in enumerate(table):
            cells = [str(c or "").strip() for c in (row or [])]
            po_hits = [
                (ci, c)
                for ci, c in enumerate(cells)
                if re.match(r"^P[OS]\d+$", c.strip(), re.IGNORECASE)
            ]
            if len(po_hits) >= 3:
                header_row_idx = ri
                header_cols = {ci: c.upper() for ci, c in po_hits}
                break
        if header_row_idx is None:
            continue
        for row in table[header_row_idx + 1 :]:
            cells = [str(c or "").strip() for c in (row or [])]
            co_id = None
            for c in cells:
                if re.match(r"^CO\d+$", c.strip(), re.IGNORECASE):
                    co_id = c.strip().upper()
                    break
            if not co_id:
                continue
            for ci, po_label in header_cols.items():
                if ci < len(cells):
                    val = cells[ci].strip()
                    if val and val not in ("-", "", "None", "null"):
                        try:
                            weight = int(val)
                            if weight > 0:
                                mappings.append(
                                    {
                                        "co_id": co_id,
                                        "po_id": po_label,
                                        "weight": weight,
                                    }
                                )
                        except ValueError:
                            pass
    return mappings


def _parse_units(body: str) -> list:
    units = []
    stop_match = _STOP_SECTION_RE.search(body)
    safe_end = stop_match.start() if stop_match else len(body)
    body_trimmed = body[:safe_end]

    unit_matches = list(_UNIT_RE.finditer(body_trimmed))
    if not unit_matches:
        return units

    for idx, u_match in enumerate(unit_matches):
        unit_num = _roman_to_int(u_match.group(1))
        u_start = u_match.end()
        u_end = (
            unit_matches[idx + 1].start() if idx + 1 < len(unit_matches) else safe_end
        )

        unit_body = body_trimmed[u_start:u_end]
        inner_stop = _STOP_SECTION_RE.search(unit_body)
        if inner_stop:
            unit_body = unit_body[: inner_stop.start()]

        raw_topics = re.split(r"[,;\-–]", unit_body)
        topics = []
        for rt in raw_topics:
            t = re.sub(r"\s+", " ", rt.strip().replace("\n", " "))
            if len(t) > 3 and len(t) < 200 and not _is_junk_topic(t):
                if t.lower() not in {
                    "introduction",
                    "overview",
                    "summary",
                    "conclusion",
                    "and",
                    "the",
                    "of",
                    "in",
                }:
                    topics.append(t)

        if topics:
            units.append(
                {"number": unit_num, "title": f"Unit {unit_num}", "topics": topics}
            )

    return units


def _parse_evaluation_pattern(body: str, tables: list) -> dict | None:
    pattern = {"ratio": None, "internal": [], "external": []}

    ratio_m = _EVAL_RATIO_RE.search(body)
    if ratio_m:
        pattern["ratio"] = f"{ratio_m.group(1)}:{ratio_m.group(2)}"

    for table in tables:
        if not table or len(table) < 2:
            continue
        header = [str(c or "").strip().lower() for c in (table[0] or [])]
        has_assessment = any("assessment" in h for h in header)
        has_internal = any("internal" in h for h in header)
        has_external = any(
            k in " ".join(header) for k in ["external", "end semester", "end\nsemester"]
        )

        if not (has_assessment or (has_internal and has_external)):
            continue

        assessment_col = next((i for i, h in enumerate(header) if "assessment" in h), 0)
        internal_col = next((i for i, h in enumerate(header) if "internal" in h), None)
        external_col = next(
            (i for i, h in enumerate(header) if "external" in h or "end" in h), None
        )

        for row in table[1:]:
            cells = [str(c or "").strip() for c in (row or [])]
            if not cells or len(cells) <= assessment_col:
                continue
            comp_name = cells[assessment_col]
            if not comp_name or comp_name.lower() == "assessment":
                continue

            int_marks = (
                cells[internal_col].strip()
                if internal_col and internal_col < len(cells)
                else ""
            )
            ext_marks = (
                cells[external_col].strip()
                if external_col and external_col < len(cells)
                else ""
            )

            if int_marks and int_marks not in ("-", ""):
                try:
                    pattern["internal"].append(
                        {
                            "name": comp_name,
                            "marks": int(re.search(r"\d+", int_marks).group()),
                        }
                    )
                except Exception:
                    pattern["internal"].append({"name": comp_name, "marks": int_marks})

            if ext_marks and ext_marks not in ("-", ""):
                try:
                    pattern["external"].append(
                        {
                            "name": comp_name,
                            "marks": int(re.search(r"\d+", ext_marks).group()),
                        }
                    )
                except Exception:
                    pattern["external"].append({"name": comp_name, "marks": ext_marks})

        if pattern["internal"] or pattern["external"]:
            return pattern

    return (
        pattern
        if (pattern["internal"] or pattern["external"] or pattern["ratio"])
        else None
    )


def _parse_bibliography_section(body: str, markers: list) -> list:
    section = _extract_section(
        body, markers, ["Reference", "Evaluation Pattern", "Unit "]
    )
    if not section:
        return []
    entries = []
    for line in section.split("\n"):
        line = line.strip().lstrip("0123456789. ")
        if len(line) > 10:
            entries.append(line)
    return entries


_COPO_KEYWORD_RE = re.compile(r"CO-PO\s+Mapping|CO\s+PO\s+Mapping", re.IGNORECASE)


def _get_course_tables(page_texts: list, block: list) -> list:
    """
    Get tables from pages in a block that explicitly contain a CO-PO Mapping table.
    Inline detect (slower) — only called for blocks where the text mentions CO-PO Mapping.
    """
    tables = []
    for page_num in block:
        if _COPO_KEYWORD_RE.search(page_texts[page_num]):
            tables.extend(_get_page_tables_inline(page_num))
    return tables


def _parse_courses(page_texts, page_type_idx, page_tables) -> list:
    courses = []

    # Detail pages include COURSE_DETAIL and any contiguous OTHER pages (spillover)
    detail_pages = []
    for i, t in page_type_idx.items():
        if t == PAGE_COURSE_DETAIL or t == PAGE_OTHER and (i - 1) in detail_pages:
            detail_pages.append(i)

    # Group consecutive pages into blocks (one block per contiguous course section)
    course_blocks = []
    block = []
    for i in detail_pages:
        if not block:
            block = [i]
        elif i - block[-1] <= 1:
            block.append(i)
        else:
            course_blocks.append(block)
            block = [i]
    if block:
        course_blocks.append(block)

    for block in course_blocks:
        page_offsets = []
        full_text = ""
        for p in block:
            page_offsets.append((len(full_text), p))
            full_text += page_texts[p] + "\n\x0c\n"

        # CO-PO tables: detect only per-block where CO-PO is mentioned (saves time)

        for hdr_match in _COURSE_HEADER_RE.finditer(full_text):
            code = hdr_match.group(1).upper()
            name = hdr_match.group(2).strip()
            l_ = hdr_match.group(3)
            t_ = hdr_match.group(4)
            p_ = hdr_match.group(5)
            c_ = hdr_match.group(6)

            start = hdr_match.end()
            next_hdr = _COURSE_HEADER_RE.search(full_text, start)
            end = next_hdr.start() if next_hdr else len(full_text)
            body = full_text[start:end]

            # Determine which pages this course spans
            course_pages = []
            for i, (offset, p) in enumerate(page_offsets):
                next_offset = (
                    page_offsets[i + 1][0]
                    if i + 1 < len(page_offsets)
                    else len(full_text)
                )
                # Course spans from hdr_match.start() to end.
                if next_offset > hdr_match.start() and offset < end:
                    course_pages.append(p)

            # Only load tables for this course body if CO-PO is mentioned
            if _COPO_KEYWORD_RE.search(body):
                course_tables = _get_course_tables(page_texts, course_pages)
            else:
                course_tables = []

            course = {
                "code": code,
                "name": name,
                "L": l_,
                "T": t_,
                "P": p_,
                "credits": c_,
                "objectives": [],
                "prerequisites_courses": [],
                "prerequisites_knowledge": [],
                "outcomes": [],
                "co_po_mapping": [],
                "units": [],
                "textbooks": [],
                "references": [],
                "evaluation_pattern": None,
                "source_pages": block,
            }

            # Objectives
            obj_text = _extract_section(
                body,
                ["Course Objectives", "Course Objective"],
                ["Course Outcomes", "CO1", "Pre-Requisite"],
            )
            if obj_text:
                for line in obj_text.split("\n"):
                    line = line.strip().lstrip("•●-* ")
                    if len(line) > 15:
                        course["objectives"].append(line)

            # Prerequisites
            pre_m = _PREREQ_RE.search(body)
            if pre_m:
                for part in re.split(r"[,;]", pre_m.group(1).strip()):
                    part = part.strip()
                    cm = _COURSE_CODE_RE.search(part)
                    if cm:
                        course["prerequisites_courses"].append(cm.group(0).upper())
                    elif len(part) > 3:
                        course["prerequisites_knowledge"].append(part)

            # Course Outcomes
            course["outcomes"] = [
                {"id": m.group(1).upper(), "description": m.group(2).strip()}
                for m in _CO_LINE_RE.finditer(body)
            ]

            # CO-PO mapping (only from tables with CO-PO header)
            course["co_po_mapping"] = _parse_copo_table(course_tables)

            # Units & Topics
            course["units"] = _parse_units(body)

            # Textbooks & References
            course["textbooks"] = _parse_bibliography_section(
                body, ["Textbook", "TEXTBOOK", "TEXT BOOK"]
            )
            course["references"] = _parse_bibliography_section(
                body, ["Reference", "REFERENCE", "REFERENCE BOOK"]
            )

            # Evaluation Pattern (reuse already-loaded course_tables)
            # Also try to load eval tables if not already done
            if not course_tables and re.search(
                r"Evaluation Pattern", body, re.IGNORECASE
            ):
                course_tables = _get_course_tables(page_texts, block)
            course["evaluation_pattern"] = _parse_evaluation_pattern(
                body, course_tables
            )

            courses.append(course)

    return courses


def _parse_curriculum_rules(page_texts, page_type_idx) -> list:
    rules = []
    for page_num, ptype in page_type_idx.items():
        if ptype != PAGE_RULE:
            continue
        for para in page_texts[page_num].split("\n\n"):
            para = para.strip()
            if len(para) > 30:
                rules.append(
                    {
                        "text": para,
                        "text_hash": hashlib.md5(para.encode()).hexdigest(),
                        "source_page": page_num,
                    }
                )
    return rules


# Phase 4 — Neo4j MERGE Writes


def _write_neo4j(
    driver,
    dept,
    year,
    sv_id,
    general_info,
    semesters,
    baskets,
    courses,
    rules,
    page_texts,
    page_type_idx,
):

    with driver.session() as session:
        # 0. Wipe old subgraph for this dept+year
        print(f"[Parser] Wiping old graph for {dept} {year}...")
        session.run(
            """
            MATCH (sv:SyllabusVersion {dept: $dept, year: $year})
            OPTIONAL MATCH (sv)-[*0..12]->(n)
            DETACH DELETE n, sv
        """,
            dept=dept,
            year=year,
        )

        # 1. Department + SyllabusVersion
        session.run(
            """
            MERGE (d:Department {name: $dept})
            MERGE (sv:SyllabusVersion {dept: $dept, year: $year})
            SET sv.id = $sv_id, sv.created_at = datetime()
            MERGE (d)-[:HAS_SYLLABUS_VERSION]->(sv)
        """,
            dept=dept,
            year=year,
            sv_id=sv_id,
        )

        # 2. General Information
        session.run(
            """
            MATCH (sv:SyllabusVersion {dept: $dept, year: $year})
            MERGE (gi:GeneralInformation {syllabus_version_id: $sv_id})
            SET gi.abbreviations = $abbrevs
            MERGE (sv)-[:HAS_GENERAL_INFORMATION]->(gi)
        """,
            dept=dept,
            year=year,
            sv_id=sv_id,
            abbrevs=str(general_info.get("abbreviations", {})),
        )

        # 3. POs, PEOs, PSOs
        for po in general_info.get("program_outcomes", []):
            session.run(
                """
                MATCH (sv:SyllabusVersion {dept: $dept, year: $year})
                MERGE (po:ProgramOutcome {id: $id, syllabus_version_id: $sv_id})
                SET po.description = $desc
                MERGE (sv)-[:HAS_PO]->(po)
            """,
                dept=dept,
                year=year,
                sv_id=sv_id,
                id=po["id"],
                desc=po["description"],
            )

        for peo in general_info.get("program_educational_objectives", []):
            session.run(
                """
                MATCH (sv:SyllabusVersion {dept: $dept, year: $year})
                MERGE (peo:ProgramEducationalObjective {id: $id, syllabus_version_id: $sv_id})
                SET peo.description = $desc
                MERGE (sv)-[:HAS_PEO]->(peo)
            """,
                dept=dept,
                year=year,
                sv_id=sv_id,
                id=peo["id"],
                desc=peo["description"],
            )

        for pso in general_info.get("program_specific_outcomes", []):
            session.run(
                """
                MATCH (sv:SyllabusVersion {dept: $dept, year: $year})
                MERGE (pso:ProgramSpecificOutcome {id: $id, syllabus_version_id: $sv_id})
                SET pso.description = $desc
                MERGE (sv)-[:HAS_PSO]->(pso)
            """,
                dept=dept,
                year=year,
                sv_id=sv_id,
                id=pso["id"],
                desc=pso["description"],
            )

        # 4. Semesters + Courses + Elective Slots
        for sem in semesters:
            sem_num = sem["number"]
            session.run(
                """
                MATCH (sv:SyllabusVersion {dept: $dept, year: $year})
                MERGE (sem:Semester {number: $num, syllabus_version_id: $sv_id})
                SET sem.label = $label, sem.total_credits = $tc
                MERGE (sv)-[:HAS_SEMESTER]->(sem)
            """,
                dept=dept,
                year=year,
                sv_id=sv_id,
                num=sem_num,
                label=sem.get("label", f"Semester {sem_num}"),
                tc=sem.get("total_credits"),
            )

            for c in sem.get("courses", []):
                session.run(
                    """
                    MATCH (sem:Semester {number: $sem_num, syllabus_version_id: $sv_id})
                    MERGE (c:Course {code: $code, dept: $dept, year: $year})
                    SET c.title = $title, c.category = $cat,
                        c.L = $L, c.T = $T, c.P = $P,
                        c.credits = $cred,
                        c.evaluation_pattern_ratio = $ep
                    MERGE (sem)-[:OFFERS]->(c)
                """,
                    sv_id=sv_id,
                    sem_num=sem_num,
                    dept=dept,
                    year=year,
                    code=c["code"],
                    title=c.get("title", ""),
                    cat=c.get("category", ""),
                    L=c.get("L"),
                    T=c.get("T"),
                    P=c.get("P"),
                    cred=c.get("credits", ""),
                    ep=c.get("evaluation_pattern_ratio", ""),
                )

            for slot in sem.get("elective_slots", []):
                slot_id = f"{sv_id}_sem{sem_num}_{slot['label'][:30]}"
                session.run(
                    """
                    MATCH (sem:Semester {number: $sem_num, syllabus_version_id: $sv_id})
                    MERGE (es:ElectiveSlot {slot_id: $slot_id})
                    SET es.label = $label, es.credits = $cred
                    MERGE (sem)-[:HAS_ELECTIVE_SLOT]->(es)
                """,
                    sv_id=sv_id,
                    sem_num=sem_num,
                    slot_id=slot_id,
                    label=slot["label"],
                    cred=slot.get("credits", ""),
                )

        # 5. Elective Baskets
        for basket in baskets:
            bname = basket["name"]
            session.run(
                """
                MATCH (sv:SyllabusVersion {dept: $dept, year: $year})
                MERGE (eb:ElectiveBasket {name: $name, syllabus_version_id: $sv_id})
                MERGE (sv)-[:HAS_ELECTIVE_BASKET]->(eb)
            """,
                dept=dept,
                year=year,
                sv_id=sv_id,
                name=bname,
            )

            for c in basket.get("courses", []):
                session.run(
                    """
                    MERGE (c:Course {code: $code, dept: $dept, year: $year})
                    ON CREATE SET c.title = $title, c.credits = $cred
                    WITH c
                    MATCH (eb:ElectiveBasket {name: $bname, syllabus_version_id: $sv_id})
                    MERGE (eb)-[:CONTAINS]->(c)
                """,
                    code=c["code"],
                    dept=dept,
                    year=year,
                    title=c.get("title", ""),
                    cred=c.get("credits", ""),
                    bname=bname,
                    sv_id=sv_id,
                )

        # 6. Course Detail
        for course in courses:
            code = course["code"]
            session.run(
                """
                MERGE (c:Course {code: $code, dept: $dept, year: $year})
                ON CREATE SET c.title = $name
                SET c.name = $name, c.L = $L, c.T = $T, c.P = $P, c.credits = $cred
            """,
                code=code,
                dept=dept,
                year=year,
                name=course.get("name", ""),
                L=course.get("L"),
                T=course.get("T"),
                P=course.get("P"),
                cred=course.get("credits", ""),
            )

            for prereq_code in course.get("prerequisites_courses", []):
                session.run(
                    """
                    MATCH (c:Course {code: $code, dept: $dept, year: $year})
                    MERGE (pre:Course {code: $pre_code, dept: $dept, year: $year})
                    MERGE (c)-[:REQUIRES_PREREQUISITE]->(pre)
                """,
                    code=code,
                    dept=dept,
                    year=year,
                    pre_code=prereq_code,
                )

            for know in course.get("prerequisites_knowledge", []):
                if len(know) > 3:
                    session.run(
                        """
                        MATCH (c:Course {code: $code, dept: $dept, year: $year})
                        MERGE (pk:PrerequisiteKnowledge {description: $desc})
                        MERGE (c)-[:REQUIRES_KNOWLEDGE]->(pk)
                    """,
                        code=code,
                        dept=dept,
                        year=year,
                        desc=know,
                    )

            for i, obj in enumerate(course.get("objectives", []), start=1):
                session.run(
                    """
                    MATCH (c:Course {code: $code, dept: $dept, year: $year})
                    MERGE (o:CourseObjective {course_code: $code, dept: $dept, year: $year, order: $order})
                    SET o.description = $desc
                    MERGE (c)-[:HAS_OBJECTIVE]->(o)
                """,
                    code=code,
                    dept=dept,
                    year=year,
                    order=i,
                    desc=obj,
                )

            for co in course.get("outcomes", []):
                session.run(
                    """
                    MATCH (c:Course {code: $code, dept: $dept, year: $year})
                    MERGE (co:CourseOutcome {id: $co_id, course_code: $code, dept: $dept, year: $year})
                    SET co.description = $desc
                    MERGE (c)-[:HAS_OUTCOME]->(co)
                """,
                    code=code,
                    dept=dept,
                    year=year,
                    co_id=co["id"],
                    desc=co["description"],
                )

            for mp in course.get("co_po_mapping", []):
                co_id, po_id, weight = mp["co_id"], mp["po_id"], mp["weight"]
                target_label = (
                    "ProgramSpecificOutcome"
                    if po_id.startswith("PSO")
                    else "ProgramOutcome"
                )
                session.run(
                    f"""
                    MATCH (co:CourseOutcome {{id: $co_id, course_code: $code,
                                             dept: $dept, year: $year}})
                    MATCH (po:{target_label} {{id: $po_id, syllabus_version_id: $sv_id}})
                    MERGE (co)-[r:MAPS_TO {{po_id: $po_id}}]->(po)
                    SET r.weight = $weight
                """,
                    co_id=co_id,
                    code=code,
                    dept=dept,
                    year=year,
                    po_id=po_id,
                    sv_id=sv_id,
                    weight=weight,
                )

            for unit in course.get("units", []):
                u_num = unit["number"]
                session.run(
                    """
                    MATCH (c:Course {code: $code, dept: $dept, year: $year})
                    MERGE (u:Unit {number: $num, course_code: $code, dept: $dept, year: $year})
                    SET u.title = $title
                    MERGE (c)-[:HAS_UNIT]->(u)
                """,
                    code=code,
                    dept=dept,
                    year=year,
                    num=u_num,
                    title=unit.get("title", f"Unit {u_num}"),
                )

                for order, topic_text in enumerate(unit.get("topics", []), start=1):
                    topic_id = f"{code}_{dept}_{year}_u{u_num}_t{order}"
                    session.run(
                        """
                        MATCH (u:Unit {number: $num, course_code: $code, dept: $dept, year: $year})
                        MERGE (t:Topic {topic_id: $topic_id})
                        SET t.name = $name, t.order = $order
                        MERGE (u)-[:HAS_TOPIC]->(t)
                    """,
                        code=code,
                        dept=dept,
                        year=year,
                        num=u_num,
                        topic_id=topic_id,
                        name=topic_text,
                        order=order,
                    )

            for i, tb in enumerate(course.get("textbooks", []), start=1):
                if len(tb) > 5:
                    session.run(
                        """
                        MATCH (c:Course {code: $code, dept: $dept, year: $year})
                        MERGE (tb:Textbook {course_code: $code, dept: $dept, year: $year, order: $order})
                        SET tb.text = $text
                        MERGE (c)-[:USES_TEXTBOOK]->(tb)
                    """,
                        code=code,
                        dept=dept,
                        year=year,
                        order=i,
                        text=tb,
                    )

            for i, ref in enumerate(course.get("references", []), start=1):
                if len(ref) > 5:
                    session.run(
                        """
                        MATCH (c:Course {code: $code, dept: $dept, year: $year})
                        MERGE (ref:Reference {course_code: $code, dept: $dept, year: $year, order: $order})
                        SET ref.text = $text
                        MERGE (c)-[:HAS_REFERENCE]->(ref)
                    """,
                        code=code,
                        dept=dept,
                        year=year,
                        order=i,
                        text=ref,
                    )

            ep = course.get("evaluation_pattern")
            if ep:
                ep_id = f"{code}_{dept}_{year}_ep"
                session.run(
                    """
                    MATCH (c:Course {code: $code, dept: $dept, year: $year})
                    MERGE (ep:EvaluationPattern {eval_pattern_id: $ep_id})
                    SET ep.ratio = $ratio
                    MERGE (c)-[:HAS_EVALUATION_PATTERN]->(ep)
                """,
                    code=code,
                    dept=dept,
                    year=year,
                    ep_id=ep_id,
                    ratio=ep.get("ratio", ""),
                )

                if ep.get("internal"):
                    session.run(
                        """
                        MATCH (ep:EvaluationPattern {eval_pattern_id: $ep_id})
                        MERGE (int:Internal {eval_pattern_id: $ep_id})
                        MERGE (ep)-[:HAS_INTERNAL]->(int)
                    """,
                        ep_id=ep_id,
                    )
                    for comp in ep["internal"]:
                        session.run(
                            """
                            MATCH (int:Internal {eval_pattern_id: $ep_id})
                            MERGE (ac:AssessmentComponent {name: $name, eval_pattern_id: $ep_id, branch: 'internal'})
                            SET ac.marks = $marks
                            MERGE (int)-[:HAS_ASSESSMENT]->(ac)
                        """,
                            ep_id=ep_id,
                            name=comp["name"],
                            marks=comp.get("marks"),
                        )

                if ep.get("external"):
                    session.run(
                        """
                        MATCH (ep:EvaluationPattern {eval_pattern_id: $ep_id})
                        MERGE (ext:External {eval_pattern_id: $ep_id})
                        MERGE (ep)-[:HAS_EXTERNAL]->(ext)
                    """,
                        ep_id=ep_id,
                    )
                    for comp in ep["external"]:
                        session.run(
                            """
                            MATCH (ext:External {eval_pattern_id: $ep_id})
                            MERGE (ac:AssessmentComponent {name: $name, eval_pattern_id: $ep_id, branch: 'external'})
                            SET ac.marks = $marks
                            MERGE (ext)-[:HAS_ASSESSMENT]->(ac)
                        """,
                            ep_id=ep_id,
                            name=comp["name"],
                            marks=comp.get("marks"),
                        )

        # 7. Curriculum Rules
        for rule in rules:
            session.run(
                """
                MATCH (sv:SyllabusVersion {dept: $dept, year: $year})
                MERGE (r:CurriculumRule {text_hash: $hash, syllabus_version_id: $sv_id})
                SET r.text = $text, r.source_page = $page
                MERGE (sv)-[:HAS_RULE]->(r)
            """,
                dept=dept,
                year=year,
                sv_id=sv_id,
                hash=rule["text_hash"],
                text=rule["text"],
                page=rule.get("source_page"),
            )

    # 8. Source/Document Layer — Pages + DocumentElements (separate session per batch)
    print(f"[Parser] Writing source layer ({len(page_texts)} pages)...")
    batch_size = 30
    for batch_start in range(0, len(page_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(page_texts))
        with driver.session() as s2:
            for page_num in range(batch_start, batch_end):
                s2.run(
                    """
                    MATCH (sv:SyllabusVersion {dept: $dept, year: $year})
                    MERGE (pg:Page {page_num: $page_num, syllabus_version_id: $sv_id})
                    SET pg.page_type = $ptype
                    MERGE (sv)-[:HAS_PAGE]->(pg)
                """,
                    dept=dept,
                    year=year,
                    sv_id=sv_id,
                    page_num=page_num,
                    ptype=page_type_idx.get(page_num, PAGE_OTHER),
                )

                text_content = page_texts[page_num].strip()
                if text_content:
                    elem_id = f"{sv_id}_p{page_num}_e0"
                    s2.run(
                        """
                        MATCH (pg:Page {page_num: $page_num, syllabus_version_id: $sv_id})
                        MERGE (el:DocumentElement {element_id: $elem_id})
                        SET el.type = 'Text', el.raw_content = $content,
                            el.page_num = $page_num, el.reading_order = 0
                        MERGE (pg)-[:HAS_ELEMENT]->(el)
                        MERGE (el)-[:LOCATED_ON]->(pg)
                    """,
                        sv_id=sv_id,
                        page_num=page_num,
                        elem_id=elem_id,
                        content=text_content[:5000],
                    )


# Main Entry — SyllabusParser class


class SyllabusParser:
    """
    Campus-wide GraphRAG syllabus ingestion.
    Supports any department (CSE, ECE, EEE, MECH, etc.) and curriculum year.
    """

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def parse_and_ingest(self, file_path: str, dept: str, year: str) -> dict:
        t0 = time.time()
        dept = (dept or "UNKNOWN").upper().strip()
        year = (year or "UNKNOWN").strip()
        sv_id = _sv_id(dept, year)

        print(f"\n[SyllabusParser] ====== {dept} {year} ======")
        doc = fitz.open(file_path)
        total_pages = len(doc)
        print(f"[SyllabusParser] Pages: {total_pages}")

        t1 = time.time()
        page_texts, page_type_idx = phase1_fast_pass(doc)
        type_counts = {}
        for pt in page_type_idx.values():
            type_counts[pt] = type_counts.get(pt, 0) + 1
        print(f"[SyllabusParser] Phase 1: {time.time() - t1:.2f}s | {type_counts}")

        t2 = time.time()
        page_tables = phase2_table_pass(doc, page_type_idx)
        # Keep doc open for inline CO-PO table detection in Phase 3
        set_live_doc(doc)
        print(
            f"[SyllabusParser] Phase 2: {time.time() - t2:.2f}s | {len(page_tables)} pages with tables"
        )

        t3 = time.time()
        general_info = _parse_general_info(page_texts, page_type_idx, page_tables)
        semesters = _parse_semester_tables(page_texts, page_type_idx, page_tables)
        baskets = _parse_elective_baskets(page_texts, page_type_idx, page_tables)
        courses = _parse_courses(page_texts, page_type_idx, page_tables)
        rules = _parse_curriculum_rules(page_texts, page_type_idx)
        # Done with the PDF — release it
        set_live_doc(None)
        doc.close()
        print(
            f"[SyllabusParser] Phase 3: {time.time() - t3:.2f}s | "
            f"POs={len(general_info['program_outcomes'])} "
            f"PSOs={len(general_info['program_specific_outcomes'])} "
            f"Sems={len(semesters)} Baskets={len(baskets)} "
            f"Courses={len(courses)} Rules={len(rules)}"
        )

        t4 = time.time()
        _write_neo4j(
            self.driver,
            dept,
            year,
            sv_id,
            general_info,
            semesters,
            baskets,
            courses,
            rules,
            page_texts,
            page_type_idx,
        )
        print(f"[SyllabusParser] Phase 4: {time.time() - t4:.2f}s")

        total = time.time() - t0
        summary = {
            "dept": dept,
            "year": year,
            "pages": total_pages,
            "pos": len(general_info["program_outcomes"]),
            "peos": len(general_info["program_educational_objectives"]),
            "psos": len(general_info["program_specific_outcomes"]),
            "semesters": len(semesters),
            "elective_baskets": len(baskets),
            "courses": len(courses),
            "rules": len(rules),
            "elapsed_seconds": round(total, 2),
        }
        print(f"[SyllabusParser] ====== DONE {total:.2f}s | {summary} ======\n")
        return summary
