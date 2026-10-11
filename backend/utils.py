import base64
import math
import hashlib
import io
import json
import os

import httpx
from dotenv import load_dotenv
from embedder import (
    embed_text as _cached_embed_text,
    is_valid_embedding,
    cosine_similarity,
    safe_cosine_similarity,
)

# Load environment variables from the .env file at the project root.
# os.path.dirname(__file__) gives us the 'backend/' folder.
# '..' moves us one level up to the root folder where '.env' is stored.
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True
)

# PyMuPDF (imported as 'fitz') is the library we use to read text out of PDF files.
# We wrap it in a try/except block just in case it fails to install on some machines.
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

# -----------------------------------------------------------------------------
# Ollama Configuration
# -----------------------------------------------------------------------------
from config import settings

# Fetch the base URL where Ollama is running (defaults to localhost:11434).
OLLAMA_BASE_URL = settings.ollama_base_url

# This large model (12 billion parameters) is used ONLY for generating the final written answers.
OLLAMA_MODEL = settings.ollama_model

# This small, fast model (137 million parameters) is used ONLY to convert text into math vectors.
OLLAMA_EMBED_MODEL = settings.ollama_embed_model

# -----------------------------------------------------------------------------
# Cloudinary Configuration
# -----------------------------------------------------------------------------
try:
    import cloudinary
    import cloudinary.api
    import cloudinary.uploader

    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
    if CLOUDINARY_CLOUD_NAME:
        cloudinary.config(
            cloud_name=CLOUDINARY_CLOUD_NAME,
            api_key=CLOUDINARY_API_KEY,
            api_secret=CLOUDINARY_API_SECRET,
            secure=True,
        )
        print(f"[Cloudinary] Configured for cloud: {CLOUDINARY_CLOUD_NAME}")
    else:
        print("[Cloudinary] No credentials set — images will be saved locally.")
except ImportError:
    cloudinary = None
    CLOUDINARY_CLOUD_NAME = ""
    print("[Cloudinary] Package not installed — images will be saved locally.")


def upload_question_image_to_cloudinary(image_bytes: bytes, public_id: str) -> str:
    """
    Uploads a cropped question image to Cloudinary and returns its secure HTTPS URL.
    If Cloudinary is not configured, falls back to saving the image locally and
    returns a localhost URL instead — so the system never crashes.

    Args:
        image_bytes: Raw JPEG bytes of the cropped question image.
        public_id:   A unique identifier string used as the Cloudinary asset name.

    Returns:
        A publicly accessible HTTPS URL (Cloudinary) or a localhost URL (fallback).
    """
    if cloudinary and CLOUDINARY_CLOUD_NAME:
        try:
            buf = io.BytesIO(image_bytes)
            result = cloudinary.uploader.upload(
                buf,
                public_id=f"campus_ke/{public_id}",
                resource_type="image",
                overwrite=True,
                transformation=[{"fetch_format": "auto", "quality": "auto"}],
            )
            url = result.get("secure_url", "")
            print(f"[Cloudinary] Uploaded: {url}")
            return url
        except Exception as e:
            print(f"[Cloudinary] Upload failed ({e}), falling back to local storage.")

    # --- Local fallback ---
    filename = f"{public_id}.jpg"
    local_path = f"uploads/images/{filename}"
    os.makedirs("uploads/images", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(image_bytes)
    return f"http://localhost:8000/static/images/{filename}"


def delete_document_images_from_cloudinary(document_id: str):
    """
    Deletes all Cloudinary resources starting with the document prefix.
    """
    if cloudinary and CLOUDINARY_CLOUD_NAME:
        try:
            prefix = f"campus_ke/{document_id}"
            res = cloudinary.api.delete_resources_by_prefix(prefix)
            print(f"[Cloudinary] Deleted resources with prefix '{prefix}': {res}")
        except Exception as e:
            print(f"[Cloudinary] Failed to delete resources for doc {document_id}: {e}")


def delete_all_images_from_cloudinary():
    """
    Deletes all Cloudinary resources in the campus_ke/ folder.
    """
    if cloudinary and CLOUDINARY_CLOUD_NAME:
        try:
            prefix = "campus_ke/"
            res = cloudinary.api.delete_resources_by_prefix(prefix)
            print(f"[Cloudinary] Wiped all resources with prefix '{prefix}': {res}")
        except Exception as e:
            print(f"[Cloudinary] Failed to wipe all Cloudinary resources: {e}")


# 1. PDF Processing — Split a PDF file into small text chunks
def process_pdf(file_path: str) -> list[str]:
    """
    Opens a PDF file, reads every page, and splits the text into individual paragraphs.
    Returning small 'chunks' of text makes it easier for the AI to find specific answers later.
    """
    if not fitz:
        print("Error: PyMuPDF is not installed. Cannot process the PDF.")
        return []

    chunks = []
    try:
        # Open the PDF file
        doc = fitz.open(file_path)

        # Loop through every page in the document
        for page_num in range(len(doc)):
            # Extract all raw text from the current page
            page_text = doc[page_num].get_text("text")

            # Split the page text by double newlines to separate paragraphs
            paragraphs = page_text.split("\n\n")

            for p in paragraphs:
                cleaned = p.strip()
                # Only save chunks that have actual content (more than 20 characters)
                if len(cleaned) > 20:
                    chunks.append(cleaned)

        doc.close()
    except Exception as e:
        print(f"Error while reading PDF: {e}")

    return chunks


# 1b. Visual Content Extraction — Describe equations, diagrams, and circuits
async def describe_page_image(base64_image: str, page_num: int) -> str:
    """
    Sends a rendered PDF page image to the gemma4 vision model.
    The model identifies and describes all equations, formulas, circuit diagrams,
    block diagrams, and other visual elements on the page.
    Returns a plain-text description of the visual content found.
    """
    prompt = (
        "You are an expert academic content analyst. Examine this page image carefully. "
        "Identify and describe ALL of the following if present:\n"
        "- Mathematical equations and formulas (write them out in plain text notation)\n"
        "- Circuit diagrams (describe components, connections, and their purpose)\n"
        "- Block diagrams and flowcharts (describe the blocks and data flow)\n"
        "- Graphs and plots (describe axes, trends, and what they represent)\n"
        "- Tables with technical data\n\n"
        "For each item found, explain what it represents and which academic concept it belongs to. "
        "If no visual/technical content is found on this page, respond with exactly: NO_VISUAL_CONTENT"
    )

    try:
        async with httpx.AsyncClient(
            headers={"ngrok-skip-browser-warning": "true"}
        ) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "images": [base64_image],
                    "stream": False,
                    "options": {"num_ctx": 8192},
                },
                timeout=180.0,  # Vision processing can take longer than text
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()

            # If the model found nothing visual, return empty string
            if "NO_VISUAL_CONTENT" in result:
                return ""
            return result

    except Exception as e:
        print(f"[Vision] Failed to describe page {page_num + 1}: {e}")
        return ""


import re

from PIL import Image


def process_pyq_visuals(file_path: str) -> list[dict]:
    """
    Renders pages of a PYQ PDF and dynamically slices them into smaller chunks
    (one per question) based on the physical Y-coordinates of question headers.
    """
    if not fitz:
        print("Error: PyMuPDF is not installed. Cannot extract visuals.")
        return []

    chunks = []
    try:
        doc = fitz.open(file_path)
        zoom_matrix = fitz.Matrix(2, 2)

        # Regex to match question start markers like "1. ", "2(a). ", "Part A"
        # It must be at the very start of the text block to prevent false positives.
        q_pattern = re.compile(r"^(?:\d+[\.\)]\s+|Part\s+[A-Z])", re.IGNORECASE)

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")

            y_coords = []
            for b in blocks:
                _x0, y0, _x1, _y1, text, _block_no, block_type = b
                if block_type == 0:
                    clean_text = text.replace("\n", " ").strip()
                    if q_pattern.search(clean_text):
                        y_coords.append(y0)

            # Sort the y-coordinates
            y_coords = sorted(y_coords)

            # If no questions found, just use the whole page as one chunk
            if not y_coords:
                y_coords = [0]
            else:
                # Always start the first chunk at the top of the page if the first question isn't at the very top
                if y_coords[0] > 50:
                    y_coords.insert(0, 0)

            # Scale coordinates for 2x zoom and add the bottom of the page
            scaled_y = [int(y * 2) for y in y_coords]
            pixmap = page.get_pixmap(matrix=zoom_matrix)
            scaled_y.append(pixmap.height)

            # Convert PyMuPDF Pixmap to Pillow Image for easy cropping
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

            # Slice the image horizontally
            for i in range(len(scaled_y) - 1):
                top = scaled_y[i]
                # Pad slightly upward for safety, but don't go below 0
                crop_top = max(0, top - 20)
                crop_bottom = scaled_y[i + 1]

                # If chunk is too small (e.g., less than 50 pixels), skip it
                if crop_bottom - crop_top < 50:
                    continue

                chunk_img = img.crop((0, crop_top, pixmap.width, crop_bottom))

                # Convert back to base64
                img_byte_arr = io.BytesIO()
                chunk_img.save(img_byte_arr, format="JPEG", quality=85)
                b64_string = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

                chunks.append(
                    {"page": page_num, "chunk_index": i, "base64": b64_string}
                )

        doc.close()
    except Exception as e:
        import traceback

        print(f"[Vision] Error slicing PYQ pages: {e}")
        traceback.print_exc()

    return chunks


async def describe_uploaded_image(base64_image: str) -> str:
    """
    Takes a base64-encoded image uploaded by a student (e.g. a photo of an
    equation, formula, or circuit diagram) and asks the vision model to
    describe what it contains. The description is then used as a search query
    to find related curriculum content.
    """
    prompt = (
        "You are an expert academic content analyst. A student has uploaded an image. "
        "Examine it carefully and provide a detailed description of what you see.\n\n"
        "If it contains an equation or formula, write it out in plain text and explain "
        "what each variable represents and which topic or concept it belongs to.\n\n"
        "If it contains a circuit diagram, describe every component, their connections, "
        "and the overall purpose of the circuit.\n\n"
        "If it contains any other diagram (flowchart, block diagram, graph), describe "
        "the elements and the concept it illustrates.\n\n"
        "Be thorough. Your description will be used to search a curriculum database, "
        "so include all relevant academic terms and concept names."
    )

    try:
        async with httpx.AsyncClient(
            headers={"ngrok-skip-browser-warning": "true"}
        ) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "images": [base64_image],
                    "stream": False,
                    "options": {"num_ctx": 8192},
                },
                timeout=180.0,
            )
            response.raise_for_status()
            return response.json().get("response", "Could not analyze the image.")
    except Exception as e:
        return f"Error analyzing image: {e}"


# 2. Vector Embeddings — Convert text into a mathematical array
import asyncio


async def get_embedding(text: str) -> list:
    return await _cached_embed_text(text)


# 3. Knowledge Graph Extraction — Find concepts and their relationships
async def extract_knowledge_graph(course_code: str, text: str) -> list[dict]:
    """
    Sends a text chunk to the large LLM (gemma4) and asks it to identify academic
    concepts and how they relate to each other.
    Returns a list of Subject-Predicate-Object triplets in JSON format.
    """
    prompt = f"""
You are an expert at extracting Knowledge Graphs from academic text.
Extract key concepts and their relationships from the following text about {course_code}.

TEXT:
{text}

Return ONLY a JSON object containing a "triplets" key with an array of relationships. No markdown, no explanation. Use this exact format:
{{
  "triplets": [
    {{"subject": "Concept A", "predicate": "is related to", "object": "Concept B"}}
  ]
}}
"""

    try:
        async with httpx.AsyncClient(
            headers={"ngrok-skip-browser-warning": "true"}
        ) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",  # Forces Ollama to return valid JSON
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

            # Parse the string response into an actual Python dictionary/list
            result_str = data.get("response", "[]")
            triplets = json.loads(result_str)

            # Sometimes the LLM wraps the array inside an object like {"triplets": [...]}.
            # We unwrap it here if necessary.
            if isinstance(triplets, dict):
                triplets = triplets.get("triplets", [])

            # Ensure we are returning a list
            if not isinstance(triplets, list):
                return []

            # Filter out any invalid items (like floats or strings generated by the LLM by mistake)
            triplets = [t for t in triplets if isinstance(t, dict)]

            return triplets

    except Exception as e:
        print(f"Failed to extract knowledge graph: {e}")
        return []


# 4. Save to Neo4j — Write the triplets into the Graph Database
def save_to_neo4j(neo4j_driver, course_code: str, triplets: list[dict]):
    """
    Takes the list of extracted triplets and writes them into Neo4j using the Cypher query language.
    Nodes represent 'Concepts', and edges represent 'Predicates' (relationships).
    """
    if not triplets:
        return

    # Open a new database session
    with neo4j_driver.session() as session:
        # MERGE ensures the Course node exists. If it doesn't, it creates it.
        session.run("MERGE (c:Course {code: $code})", code=course_code)

        import re

        for triplet in triplets:
            subject = triplet.get("subject")

            # Neo4j relationships are typically uppercase with underscores (e.g., IS_RELATED_TO)
            raw_predicate = (
                triplet.get("predicate", "RELATED_TO").upper().replace(" ", "_")
            )
            # Strip out any characters that aren't A-Z, 0-9, or underscore to prevent Cypher syntax errors
            predicate = re.sub(r"[^A-Z0-9_]", "", raw_predicate)
            if not predicate:
                predicate = "RELATED_TO"

            obj = triplet.get("object")

            if not subject or not obj:
                continue

            # Ensure subject and object are scalar strings to prevent creating nodes with array properties
            if not isinstance(subject, str) or not isinstance(obj, str):
                continue

            # Cypher Query Logic:
            # 1. Ensure Course exists
            # 2. Ensure Subject Concept exists
            # 3. Ensure Object Concept exists
            # 4. Create a relationship from Subject to Object
            # 5. Create a relationship from the main Course to the Subject
            query = f"""
                MERGE (c:Course   {{code: $course_code}})
                MERGE (s:Concept  {{name: $subject}})
                MERGE (o:Concept  {{name: $obj}})
                MERGE (s)-[:{predicate}]->(o)
                MERGE (c)-[:HAS_CONCEPT]->(s)
            """
            try:
                session.run(query, course_code=course_code, subject=subject, obj=obj)
            except Exception as e:
                print(f"Failed to write triplet {triplet} to Neo4j: {e}")


def _format_history_block(history: list[dict] = None) -> str:
    """Formats the previous conversation turns for conversational memory."""
    if not history:
        return ""
    lines = ["Conversation History (Previous turns in this session):"]
    for msg in history:
        role = "Student" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            # Keep each historical message concise to save context window
            summary = content if len(content) <= 500 else content[:500] + "..."
            lines.append(f"{role}: {summary}")
    lines.append("")
    return "\n".join(lines) + "\n"


# 5. Answer Generation — Produce an answer based on retrieved context
async def generate_answer(question: str, context: str, history: list[dict] = None) -> str:
    """
    Takes the student's original question, the retrieved context from
    PostgreSQL/Neo4j, and previous conversation history, and asks the LLM for an answer.
    """
    history_block = _format_history_block(history)
    prompt = f"""
You are MACH AI (M.A.C.H. - Multimodel Academic Cognitive Hub - A Fine-tuned Graph-RAG framework for Campus Knowledge Systems), an academic AI assistant for university students.
Answer the student's question using ONLY the information provided in the Context below.
If the student asks for questions on a topic (e.g. PYQs or exam questions), list ALL available matching questions provided in the Context across all documents. Do not omit any question.
If the student refers to previous topics or questions using pronouns (like 'it', 'its', 'that', or 'the previous algorithm'), use the Conversation History to resolve what they are asking about.

SYLLABUS & CURRICULUM RULES:
1. EXACT MATCH: If the Context provides the syllabus for an exact matched course, output ONLY that course's syllabus. Clearly display the Course Code and Course Name as the heading.
2. ASCENDING ORDER: You MUST present all Units in STRICT ASCENDING NUMERICAL ORDER (Unit 1, Unit 2, Unit 3, Unit 4, Unit 5...). Never jumble unit numbers, never repeat units, and never mix different courses together.
3. MULTIPLE SIMILAR COURSES (NO EXACT MATCH): If the Context indicates that multiple similar courses exist (no single exact match), do NOT output full syllabus topics. Instead, list the available matching courses with their Course Codes and titles, and ask the student to specify which course they want the syllabus for.

Context:
{context}

{history_block}Question: {question}

Answer:
"""

    try:
        async with httpx.AsyncClient(
            headers={"ngrok-skip-browser-warning": "true"}
        ) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": 8192},
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json().get("response", "I could not generate an answer.")
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return f"ℹ️ *(Note: AI reformatting timed out. Showing direct database search results below)*\n\n{context}"


async def generate_answer_stream(question: str, context: str, history: list[dict] = None):
    """
    Streams the AI response chunk by chunk as it generates, with conversational memory.
    """
    history_block = _format_history_block(history)
    prompt = f"""
You are MACH AI (M.A.C.H. - Multimodel Academic Cognitive Hub - A Fine-tuned Graph-RAG framework for Campus Knowledge Systems), an academic AI assistant for university students.
Answer the student's question using ONLY the information provided in the Context below.
If the student asks for questions on a topic (e.g. PYQs or exam questions), list ALL available matching questions provided in the Context across all documents. Do not omit any question.
If the student refers to previous topics or questions using pronouns (like 'it', 'its', 'that', or 'the previous algorithm'), use the Conversation History to resolve what they are asking about.

SYLLABUS & CURRICULUM RULES:
1. EXACT MATCH: If the Context provides the syllabus for an exact matched course, output ONLY that course's syllabus. Clearly display the Course Code and Course Name as the heading.
2. ASCENDING ORDER: You MUST present all Units in STRICT ASCENDING NUMERICAL ORDER (Unit 1, Unit 2, Unit 3, Unit 4, Unit 5...). Never jumble unit numbers, never repeat units, and never mix different courses together.
3. MULTIPLE SIMILAR COURSES (NO EXACT MATCH): If the Context indicates that multiple similar courses exist (no single exact match), do NOT output full syllabus topics. Instead, list the available matching courses with their Course Codes and titles, and ask the student to specify which course they want the syllabus for.

IMPORTANT: If the Context contains Markdown image links for diagrams (e.g. `![Diagram for ...](http://...)`), you MUST include EVERY Markdown image link directly under its corresponding question in your final answer so the student can see the circuit/diagram images.

Context:
{context}

{history_block}Question: {question}

Answer:
"""
    try:
        async with httpx.AsyncClient(
            headers={"ngrok-skip-browser-warning": "true"}
        ) as client, client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"num_ctx": 8192},
            },
            timeout=300.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")
    except Exception as e:
        print(f"Ollama API Error in generate_answer_stream: {e}")
        yield f"*(AI reformatting unavailable: {type(e).__name__}). Showing database search results below:*\n\n{context}"


# 6. Syllabus Structural Extraction (New Architecture)


async def extract_syllabus_structure(text: str, dept: str, year: str) -> dict:
    """
    Extracts the syllabus structure deterministically using Regular Expressions.
    Bypasses the LLM for speed and reliability, parsing hundreds of pages instantly.
    """
    # Find all course headers
    # Example: 23ENG101 TECHNICAL COMMUNICATION L-T-P-C: 2-0-3-3
    course_pattern = re.compile(
        r"([0-9]{2}[a-zA-Z]{3}[0-9]{3})\s+(.*?)\s+L-T-P-C", re.IGNORECASE
    )

    courses_data = []

    # We find all matches, and iterate through them.
    # The text for a course is everything from the end of this match to the start of the next match.
    matches = list(course_pattern.finditer(text))

    for i, match in enumerate(matches):
        course_code = match.group(1).strip().upper()
        course_name = match.group(2).strip()

        # Get the text block for this course
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        course_text = text[start_idx:end_idx]

        units_data = []

        # Now find units inside course_text
        # Matches "Unit 1", "Unit 2", "SyllabusUnit 1", etc.
        unit_pattern = re.compile(r"(?:Syllabus)?Unit\s*(\d)", re.IGNORECASE)
        unit_matches = list(unit_pattern.finditer(course_text))

        for j, u_match in enumerate(unit_matches):
            unit_num = u_match.group(1)
            u_start = u_match.end()

            # Find the end of this unit (either next unit, or stopping keywords)
            if j + 1 < len(unit_matches):
                u_end = unit_matches[j + 1].start()
            else:
                # Find the first occurrence of stopping keywords
                stop_pattern = re.compile(
                    r"^(?:TEXTBOOK|REFERENCE|Evaluation Pattern)",
                    re.IGNORECASE | re.MULTILINE,
                )
                stop_match = stop_pattern.search(course_text, u_start)
                if stop_match:
                    u_end = stop_match.start()
                else:
                    u_end = len(course_text)

            unit_text = course_text[u_start:u_end].strip()

            # Extract topics by splitting the unit_text by punctuation
            raw_topics = re.split(r"[,;\.\-\:]", unit_text)

            topics_data = []
            for rt in raw_topics:
                t_name = rt.strip().replace("\n", " ")
                t_name = re.sub(r"\s+", " ", t_name)  # normalize spaces

                # Basic cleaning: remove empty topics and very long sentences
                if len(t_name) > 3 and len(t_name) < 150:
                    # Ignore generic words
                    if t_name.lower() not in [
                        "introduction",
                        "overview",
                        "summary",
                        "conclusion",
                    ]:
                        topics_data.append({"name": t_name, "subtopics": []})

            if topics_data:
                units_data.append(
                    {
                        "number": int(unit_num),
                        "title": f"Unit {unit_num}",
                        "topics": topics_data,
                    }
                )

        if units_data:
            courses_data.append(
                {"code": course_code, "name": course_name, "units": units_data}
            )

    print(
        f"[Syllabus Parser] Successfully extracted {len(courses_data)} courses from the text!"
    )
    return {"courses": courses_data}


def build_syllabus_kg(neo4j_driver, dept: str, year: str, courses: list):
    if not courses:
        return
    with neo4j_driver.session() as session:
        session.run("MERGE (d:Department {name: $dept})", dept=dept)
        for course in courses:
            c_code = course.get("code")
            c_name = course.get("name")
            if not c_code:
                continue

            session.run(
                """
                MERGE (d:Department {name: $dept})
                MERGE (c:Course {code: $c_code})
                ON CREATE SET c.name = $c_name, c.year = $year, c.dept = $dept
                MERGE (d)-[:OFFERS]->(c)
            """,
                dept=dept,
                c_code=c_code,
                c_name=c_name,
                year=year,
            )

            for unit in course.get("units", []):
                u_num = str(unit.get("number", ""))
                u_title = unit.get("title", "")
                if not u_title:
                    continue

                session.run(
                    """
                    MATCH (c:Course {code: $c_code})
                    MERGE (u:Unit {number: $u_num, title: $u_title, course_code: $c_code})
                    MERGE (c)-[:HAS_UNIT]->(u)
                """,
                    c_code=c_code,
                    u_num=u_num,
                    u_title=u_title,
                )

                for topic in unit.get("topics", []):
                    t_name = topic.get("name")
                    if not t_name:
                        continue

                    import hashlib
                    t_id = hashlib.md5(f"{c_code}_{u_num}_{t_name}".encode()).hexdigest()
                    session.run(
                        """
                        MATCH (u:Unit {number: $u_num, title: $u_title, course_code: $c_code})
                        MERGE (t:Topic {name: $t_name, course_code: $c_code})
                        ON CREATE SET t.topic_id = $t_id, t.id = $t_id
                        MERGE (u)-[:HAS_TOPIC]->(t)
                    """,
                        u_num=u_num,
                        u_title=u_title,
                        c_code=c_code,
                        t_name=t_name,
                        t_id=t_id,
                    )

                    for subtopic in topic.get("subtopics", []):
                        if not subtopic:
                            continue
                        session.run(
                            """
                            MATCH (t:Topic {name: $t_name, course_code: $c_code})
                            MERGE (st:SubTopic {name: $subtopic, course_code: $c_code})
                            MERGE (t)-[:HAS_SUBTOPIC]->(st)
                        """,
                            t_name=t_name,
                            c_code=c_code,
                            subtopic=subtopic,
                        )


# 7. PYQ Structural Extraction and Mapping
async def extract_pyq_questions(base64_image: str) -> list[dict]:
    prompt = """
You are an expert at extracting exam questions from PYQ (Past Year Question) pages.
Examine this page image carefully. Extract all questions found on the page into structured JSON.
Predict any formulas that the student needs to use to solve the question, even if not explicitly mentioned.

Return ONLY a JSON object with this exact format:
{
  "questions": [
    {
      "question_number": "Q1(a)",
      "text": "Find the transfer function...",
      "marks": 10,
      "formula_hints": ["transfer function", "Laplace transform"],
      "likely_topic": "Control Systems / Transfer Functions",
      "implicit_formulas": ["G(s)H(s) = 1"]
    }
  ]
}
If no questions are found, return {"questions": []}. No markdown, no explanation.
"""
    try:
        async with httpx.AsyncClient(
            headers={"ngrok-skip-browser-warning": "true"}
        ) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "images": [base64_image],
                    "format": "json",
                    "stream": False,
                    "options": {"num_ctx": 8192, "num_predict": 4096},
                },
                timeout=180.0,
            )
            response.raise_for_status()
            res_text = response.json().get("response", "{}")

            # Robust JSON extraction to handle markdown, control characters, and invalid JSON escaping
            import re

            def is_valid_question(q_text: str) -> bool:
                if not q_text or len(q_text) < 20:
                    return False
                lower_text = q_text.lower()
                junk_patterns = [
                    r"^answer all",
                    r"^part [a-z]",
                    r"^section [a-z]",
                    r"maximum marks",
                    r"^time:",
                    r"q\.p\. code",
                    r"^\d+\s*x\s*\d+\s*=\s*\d+",
                    r"^[a-z]\)",
                ]
                for pattern in junk_patterns:
                    if re.search(pattern, lower_text):
                        return False
                return True

            match = re.search(r"\{.*\}", res_text, re.DOTALL)
            clean_text = match.group(0) if match else res_text

            try:
                data = json.loads(clean_text)
                if isinstance(data, dict) and "questions" in data:
                    return [
                        q
                        for q in data["questions"]
                        if is_valid_question(q.get("text", ""))
                    ]
            except Exception:
                pass

            # Sanitize control characters (e.g. \u001e)
            try:
                sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", clean_text)
                data = json.loads(sanitized)
                if isinstance(data, dict) and "questions" in data:
                    return [
                        q
                        for q in data["questions"]
                        if is_valid_question(q.get("text", ""))
                    ]
            except Exception:
                pass

            # Regex fallback: extract "text": "..." fields directly
            extracted = []
            text_matches = re.findall(r'"text"\s*:\s*"([^"]+)"', clean_text)
            for q_t in text_matches:
                if is_valid_question(q_t):
                    extracted.append(
                        {
                            "question_number": "PYQ",
                            "text": q_t,
                            "likely_topic": "General",
                            "implicit_formulas": [],
                        }
                    )
            if extracted:
                return extracted

            # Final fallback: return clean text instead of raw JSON string, but only if it looks like a valid question
            clean_human_text = re.sub(r'[{}"\[\]]', "", clean_text).strip()
            if is_valid_question(clean_human_text):
                return [
                    {
                        "question_number": "PYQ",
                        "text": clean_human_text,
                        "likely_topic": "General",
                        "implicit_formulas": [],
                    }
                ]
            else:
                return []

    except Exception as e:
        print(f"Failed to extract PYQ questions: {e}")
        return []


def clean_formula_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\t", "\\t")
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    # Fix LaTeX commands lost during JSON unescaping (e.g. imes -> \times, rac -> \frac)
    text = re.sub(r"\bimes\b", r"\\times", text)
    text = re.sub(r"\brac\b", r"\\frac", text)
    # Sanitize repetitive OCR noise characters
    text = re.sub(r"[^\w\s\+\-\*\/\=\(\)\[\]\{\}\\\$\.\,\:\_\^\@\%]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class InvalidTopicMappingError(Exception):
    """Raised inside Neo4j transaction callbacks to force rollback when topic validation fails."""
    pass


def normalize_question_number(q_num: str | None) -> str:
    """
    Safely normalizes question number string while preserving distinctions
    between parent questions and subquestions (e.g. '1', '1(a)', 'Q1' vs 'Q1(a)').
    """
    if not q_num:
        return ""
    q_str = str(q_num).strip()
    if q_str.lower() in ("unknown", "none", "null", ""):
        return ""
    m = re.match(r'^(?:Question|Q\.?)\s*([0-9]+(?:\s*\(?[a-zA-Z0-9]+\)?)?.*)$', q_str, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return q_str


def generate_canonical_question_id(
    course_code: str,
    document_id: str | None = None,
    question_number: str | None = None,
    page_num: int | str | None = None,
    question_text: str | None = None,
) -> str:
    """
    Shared canonical question ID generator for all ingestion paths.
    Deterministic format: md5(f"{doc_part}_{course_code}_{q_num_part}_p{page_part}")
    with explicit safe fallbacks when metadata is missing.
    """
    c_code = str(course_code or "UNKNOWN").strip().upper()
    doc_part = str(document_id).strip() if document_id and str(document_id).strip() not in ("None", "", "nodoc") else "nodoc"
    
    norm_q = normalize_question_number(question_number)
    
    if page_num is not None and str(page_num).strip() not in ("None", "", "nopage"):
        p_str = str(page_num).strip().lower()
        if p_str.startswith("p"):
            p_str = p_str[1:]
        page_part = p_str
    else:
        page_part = "nopage"

    q_text_clean = str(question_text or "").strip()
    text_hash = hashlib.md5(q_text_clean.encode("utf-8")).hexdigest()[:8] if q_text_clean else "notext"

    if not norm_q and page_part == "nopage" and doc_part == "nodoc":
        id_str = f"nodoc_{c_code}_noqnum_{text_hash}"
    elif not norm_q:
        id_str = f"{doc_part}_{c_code}_noqnum_{text_hash}_p{page_part}"
    elif doc_part == "nodoc":
        id_str = f"nodoc_{c_code}_{norm_q}_p{page_part}_{text_hash}"
    else:
        id_str = f"{doc_part}_{c_code}_{norm_q}_p{page_part}"

    return hashlib.md5(id_str.encode("utf-8")).hexdigest()


def check_evidence_acceptance(cand: dict) -> tuple[bool, float]:
    """
    Shared evidence acceptance helper evaluating the deterministic acceptance policy:
    A. Both lexical and semantic evidence:
       - Semantic similarity >= 0.65
       - BM25 rank <= 3
       Returns (True, sem_score)
    B. Semantic-only evidence:
       - Semantic rank exists (and lexical does not)
       - Semantic similarity >= 0.70
       Returns (True, sem_score)
    C. Lexical-only evidence:
       - BM25 rank exists (and semantic does not)
       - BM25 score > 5.0
       Returns (True, 0.5)
    D. Otherwise:
       Returns (False, 0.0)
    """
    r_lex = cand.get("r_lex")
    r_sem = cand.get("r_sem")
    sem_score = float(cand.get("sem_score", 0.0))
    bm25_score = float(cand.get("bm25_score", 0.0))

    if r_lex is not None and r_sem is not None:
        if sem_score >= 0.65 and r_lex <= 3:
            return True, sem_score
    elif r_sem is not None and r_lex is None:
        if sem_score >= 0.70:
            return True, sem_score
    elif r_lex is not None and r_sem is None:
        if bm25_score > 5.0:
            return True, 0.5

    return False, 0.0


async def map_questions_to_kg(
    neo4j_driver,
    course_code: str,
    questions: list,
    document_id: str | None = None,
    image_url: str | None = None,
):
    if not questions:
        return

    syllabus_context = fetch_course_topics(neo4j_driver, course_code)
    print(
        f"[PYQ Vision] Fetched {len(syllabus_context.split(chr(10)))} topics from syllabus. Running semantic mapping via LLM..."
    )

    mapping = await semantic_map_questions_to_topics(
        course_code, syllabus_context, questions
    )
    print(f"[PYQ Vision] Semantic mapper returned {len(mapping)} matched questions.")

    def _update_graph_tx(tx, q_id, q_text, q_num, marks, implicit_formulas, doc_id, c_code, img_url, t_data_list):
        valid_topics = []
        if t_data_list:
            for t_data in t_data_list:
                if not isinstance(t_data, dict):
                    raise InvalidTopicMappingError(f"Topic item must be a dictionary, got {type(t_data)}")
                t_id = t_data.get("topic_id")
                if not t_id or not isinstance(t_id, str):
                    raise InvalidTopicMappingError(f"Missing or non-string topic_id: {t_id}")
                
                raw_conf = t_data.get("confidence", 0.8)
                if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
                    raise InvalidTopicMappingError(f"Non-numeric confidence {raw_conf!r} for topic {t_id}")
                conf = float(raw_conf)
                if not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
                    raise InvalidTopicMappingError(f"Out of bounds confidence {conf} for topic {t_id}")

                records = tx.run(
                    """
                    MATCH (t:Topic)
                    WHERE (t.topic_id = $t_id OR t.id = $t_id OR toString(id(t)) = $t_id)
                      AND (t.course_code = $c_code OR EXISTS { MATCH (:Course {code: $c_code})-[:HAS_UNIT]->(:Unit)-[:HAS_TOPIC]->(t) })
                    RETURN id(t) as t_internal_id
                    """,
                    t_id=t_id,
                    c_code=c_code
                ).data()

                if len(records) == 0:
                    raise InvalidTopicMappingError(f"Topic {t_id} not found in course {c_code}")
                if len(records) > 1:
                    raise InvalidTopicMappingError(f"Topic {t_id} is ambiguous ({len(records)} matches) in course {c_code}")

                valid_topics.append({"t_internal_id": records[0]["t_internal_id"], "conf": conf})

        # All validation passed! Mutate Question node inside this transaction
        q_result = tx.run(
            """
            MERGE (q:Question {id: $q_id})
            SET q.text = $q_text, 
                q.question_number = $q_num, 
                q.marks = $marks, 
                q.implicit_formulas = $implicit_formulas,
                q.document_id = $doc_id,
                q.course_code = $c_code,
                q.image_url = $image_url
            MERGE (c:Course {code: $c_code})
            RETURN id(q) as internal_id
            """,
            q_id=q_id,
            c_code=c_code,
            q_text=q_text,
            q_num=q_num,
            marks=marks,
            implicit_formulas=implicit_formulas,
            doc_id=doc_id,
            image_url=img_url,
        ).single()

        if not q_result:
            raise RuntimeError("Failed to merge Question node")
        internal_id = q_result["internal_id"]

        if valid_topics:
            # Safe replacement: delete ONLY TESTS_TOPIC and BELONGS_TO, preserving MAPPED_TO_CO and EXTRACTED_FROM
            tx.run(
                """
                MATCH (q) WHERE id(q) = $internal_id
                OPTIONAL MATCH (q)-[r1:TESTS_TOPIC]->()
                OPTIONAL MATCH (q)-[r2:BELONGS_TO]->(:Course)
                DELETE r1, r2
                """,
                internal_id=internal_id
            )
            for vt in valid_topics:
                tx.run(
                    """
                    MATCH (q) WHERE id(q) = $internal_id
                    MATCH (t) WHERE id(t) = $t_internal_id
                    MERGE (q)-[r:TESTS_TOPIC]->(t)
                    SET r.confidence = $conf, r.review_status = 'pending_review'
                    """,
                    internal_id=internal_id,
                    t_internal_id=vt["t_internal_id"],
                    conf=vt["conf"]
                )
        else:
            # Legitimate abstention -> course fallback
            tx.run(
                """
                MATCH (q) WHERE id(q) = $internal_id
                MATCH (c:Course {code: $c_code})
                OPTIONAL MATCH (q)-[r1:TESTS_TOPIC]->()
                OPTIONAL MATCH (q)-[r2:BELONGS_TO]->(:Course)
                DELETE r1, r2
                MERGE (q)-[:BELONGS_TO]->(c)
                """,
                internal_id=internal_id,
                c_code=c_code,
            )
        return True

    with neo4j_driver.session() as session:
        for i, q in enumerate(questions):
            if isinstance(q, str):
                q = {
                    "text": q,
                    "question_number": "Unknown",
                    "likely_topic": "General",
                    "implicit_formulas": [],
                }

            q_text = str(q.get("text", ""))

            raw_formulas = q.get("implicit_formulas", [])
            if not isinstance(raw_formulas, list):
                raw_formulas = [raw_formulas]
            implicit_formulas = [
                clean_formula_text(str(f))
                for f in raw_formulas
                if f is not None and len(clean_formula_text(str(f))) > 1
            ]

            q_num = str(q.get("question_number", ""))
            marks = str(q.get("marks", ""))

            # Canonical question ID generation
            q_id = q.get("id")
            if not q_id:
                page_part = q.get("page_num", "nopage")
                q_id = generate_canonical_question_id(
                    course_code=course_code,
                    document_id=document_id,
                    question_number=q_num,
                    page_num=page_part,
                    question_text=q_text,
                )

            mapped_data = mapping.get(i, [])
            try:
                session.execute_write(
                    _update_graph_tx,
                    q_id,
                    q_text,
                    q_num,
                    marks,
                    implicit_formulas,
                    str(document_id) if document_id else "",
                    course_code,
                    str(image_url) if image_url else "",
                    mapped_data
                )
            except InvalidTopicMappingError as e:
                print(f"[Graph] Invalid topic mapping proposal for question {q_id}: {e}. Aborted without modifying existing mappings.")
            except Exception as e:
                print(f"[Graph] Database error during mapping update for question {q_id}: {e}. Skipping question to isolate failure without aborting batch.")


import hashlib

from sqlalchemy import text

# 4b. Neo4j Prerequisite Mapping
PREREQUISITE_MAP = {
    "23CSE203": ["23MAT116"],  # Data Structures requires Discrete Math
    "23CSE211": ["23CSE203", "23MAT116"],  # Algorithms requires DSA & Discrete Math
    "23CSE301": ["23MAT117", "23MAT216"],  # ML requires Linear Algebra & Probability
    "23CSE314": ["23CSE303"],  # Compiler Design requires Theory of Computation
    "23CSE473": ["23CSE301"],  # Deep Learning requires ML
    "23CSE477": ["23CSE301"],  # Reinforcement Learning requires ML
}


def add_prerequisite_edges(neo4j_driver, prerequisite_map):
    """
    Executes parameterized Cypher to create REQUIRES edges between courses.
    """
    if not prerequisite_map:
        return

    with neo4j_driver.session() as session:
        for target_code, prereqs in prerequisite_map.items():
            for prereq_code in prereqs:
                session.run(
                    """
                    MERGE (c1:Course {code: $target_code})
                    MERGE (c2:Course {code: $prereq_code})
                    MERGE (c1)-[:REQUIRES]->(c2)
                """,
                    target_code=target_code,
                    prereq_code=prereq_code,
                )


def _render_page_image(page, scale: float = 2.0) -> "Image.Image":
    """
    Renders a PyMuPDF page into a Pillow Image at the given scale.
    """
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix)
    return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)


def _expand_bbox_for_graphics(
    page, y_top: float, y_bottom: float, page_height: float, hard_ceiling: float
) -> float:
    """
    Scans a page for raster images and vector drawings that start between
    y_top and y_bottom (plus a small look-ahead), and expands y_bottom to
    fully enclose them.

    CRITICAL: the result is capped at `hard_ceiling` so that graphics that
    belong to the NEXT question cannot pull this question's crop past that
    boundary.  Pass the next question's y_start as hard_ceiling; pass
    page_height when this is the last question on the page.
    """
    new_bottom = y_bottom
    # Look-ahead: only grab graphics that START within the current box.
    # Never peek more than 30 PDF pts below the current boundary.
    look_ahead = min(y_bottom + 30, hard_ceiling)

    # Check embedded raster images
    for img_info in page.get_image_info():
        bbox = img_info.get("bbox", None)
        if bbox:
            img_y0, img_y1 = bbox[1], bbox[3]
            if y_top - 5 <= img_y0 <= look_ahead:
                new_bottom = max(new_bottom, min(img_y1 + 5, hard_ceiling))

    # Check vector drawings (circuit diagram lines, boxes, arcs)
    for drawing in page.get_drawings():
        rect = drawing.get("rect", None)
        if rect:
            draw_y0, draw_y1 = rect.y0, rect.y1
            if y_top - 5 <= draw_y0 <= look_ahead:
                new_bottom = max(new_bottom, min(draw_y1 + 5, hard_ceiling))

    return min(new_bottom, hard_ceiling)


def extract_pyq_structured(
    file_path: str, course_code: str, document_id: str = ""
) -> list[dict]:
    """
    Extracts structured questions from PYQ PDFs.

    Two-pass bounding-box algorithm:
      Pass 1: Collect every question header's exact pixel y-coordinate on every page.
      Pass 2: For each question, expand the crop box to enclose all diagrams/graphics,
              stitch across page boundaries when a question continues to the next page,
              and upload the final cropped image to Cloudinary (or local fallback).
    """
    if not fitz:
        print("Error: PyMuPDF is not installed. Cannot extract PYQ structured text.")
        return []

    # Regex patterns
    CO_TAG_RE = re.compile(r"\[CO\s*0*(\d+)\]", re.IGNORECASE)
    BTL_TAG_RE = re.compile(r"\[BTL\s*(\d+)\]", re.IGNORECASE)
    MARKS_RE = re.compile(r"\[(\d+)\]|\((\d+)\s*[Mm]arks?\)", re.IGNORECASE)
    FIG_RE = re.compile(r"\bFig\.?\s*\d+\b|\bfigure\b", re.IGNORECASE)
    # TOP-LEVEL questions only: "1." "2." "10." "Q1." "Question 2."
    # Sub-questions like "1a)", "(i)", "a)" are intentionally excluded so
    # they are captured as part of the parent question text + crop.
    # Restrict leading whitespace to 5 chars to avoid matching deeply indented table cells
    Q_NUM_RE = re.compile(
        r"^[ \t]{0,5}((?:Q(?:uestion)?\s*)?\d{1,2})[\.\)]\s+",
        re.MULTILINE | re.IGNORECASE,
    )
    SCALE = 2.0  # Render pages at 2x resolution for crisp crops

    # -------------------------------------------------------------------------
    # PASS 1: Gather all QuestionSpan entries across pages
    # -------------------------------------------------------------------------
    # Each span: {q_num, page_num, y_start (PDF units), text, co_tag, btl_tag, marks, has_figure}
    all_spans = []  # ordered list of spans across all pages
    page_texts = []  # raw text per page (for text extraction)
    doc_separator = None  # Tracks if the exam uses '1)' or '1.' to reject mixed formats like table rows

    try:
        doc = fitz.open(file_path)
        num_pages = len(doc)

        for page_num in range(num_pages):
            page = doc[page_num]
            page_text = page.get_text("text")
            page_texts.append(page_text)
            matches = list(Q_NUM_RE.finditer(page_text))

            for idx, match in enumerate(matches):
                q_num = match.group(1).strip()

                # Skip obvious junk headers
                if q_num.lower() in ("", "q", "question"):
                    continue

                # Find exact pixel Y for this question header via text search
                search_str = match.group(0).strip()
                rects = page.search_for(search_str)
                y_start_pdf = rects[0].y0 if rects else 0.0

                # Determine raw text extent for this question
                start_char = match.end()
                end_char = (
                    matches[idx + 1].start()
                    if idx + 1 < len(matches)
                    else len(page_text)
                )
                raw_text = page_text[start_char:end_char].strip()

                if len(raw_text) < 8:
                    continue

                # Extract metadata tags
                # Extract metadata tags
                co_m = CO_TAG_RE.search(raw_text)
                btl_m = BTL_TAG_RE.search(raw_text)
                mk_m = MARKS_RE.search(raw_text)
                marks_val = int(mk_m.group(1) or mk_m.group(2)) if mk_m else None

                has_figure = bool(FIG_RE.search(raw_text))

                # Clean the display text
                clean_text = CO_TAG_RE.sub("", raw_text)
                clean_text = BTL_TAG_RE.sub("", clean_text)
                clean_text = MARKS_RE.sub("", clean_text).strip()

                # Filter out purely numeric table rows (no alphabetical characters)
                if not re.search(r"[a-zA-Z]", clean_text):
                    continue

                # Skip boilerplate
                if any(
                    junk in clean_text.lower()[:60]
                    for junk in [
                        "answer all",
                        "part a",
                        "part b",
                        "maximum marks",
                        "time:",
                        "course outcomes",
                        "co |",
                        "all questions",
                    ]
                ):
                    continue

                # Exam papers NEVER mix '1)' and '1.' for top-level questions.
                # If we lock in the separator from the first valid question, we can
                # safely ignore nested lists or table columns that happen to use the other format.
                separator = "." if "." in match.group(0) else ")"
                if doc_separator and separator != doc_separator:
                    continue
                if not doc_separator:
                    doc_separator = separator

                all_spans.append(
                    {
                        "q_num": q_num,
                        "page_num": page_num,
                        "y_start": y_start_pdf,  # PDF coordinate units
                        "q_text": clean_text,
                        "co_tag": f"CO{co_m.group(1)}" if co_m else None,
                        "btl_tag": f"BTL{btl_m.group(1)}" if btl_m else None,
                        "marks": marks_val,
                        "has_figure": has_figure,
                    }
                )

        # -------------------------------------------------------------------------
        # PASS 2: Compute crop regions, stitch cross-page questions, upload images
        # -------------------------------------------------------------------------
        structured_questions = []

        for span_idx, span in enumerate(all_spans):
            page_num = span["page_num"]
            page = doc[page_num]
            page_h = page.rect.height  # PDF units
            y_start_pdf = span["y_start"]

            # ── Strict boundary: next question's y_start is the hard ceiling ──
            # This MUST NOT be violated by graphics expansion.
            next_same_page = next(
                (s for s in all_spans[span_idx + 1 :] if s["page_num"] == page_num),
                None,
            )
            if next_same_page and next_same_page["y_start"] > y_start_pdf:
                # Leave a 3-pt gap above the next question header so we don't
                # accidentally clip its first line.
                hard_ceiling_this_page = next_same_page["y_start"] - 3
            else:
                hard_ceiling_this_page = (
                    page_h  # last question or multi-column: use full page
                )

            # Expand to include diagrams/drawings, but NEVER past hard ceiling
            y_end_pdf = _expand_bbox_for_graphics(
                page,
                y_start_pdf,
                hard_ceiling_this_page,
                page_h,
                hard_ceiling_this_page,
            )

            # Detect cross-page continuation:
            # If this is the last question on page N, check if the next page's
            # first question starts far enough down to suggest the current
            # question's content (e.g. a diagram) continues on the next page.
            CONTINUATION_THRESHOLD_PDF = 100  # PDF points (~35 mm)
            next_span = (
                all_spans[span_idx + 1] if span_idx + 1 < len(all_spans) else None
            )
            continues_to_next_page = (
                next_same_page is None  # last question on this page
                and next_span is not None  # there is a following span
                and next_span["page_num"] == page_num + 1  # immediately next page
                and next_span["y_start"] > CONTINUATION_THRESHOLD_PDF
            )

            # ---- Render and crop the image(s) ----
            image_url = ""
            if document_id:
                try:
                    part1_img = _render_page_image(page, SCALE)
                    # Strict pixel ceiling = hard_ceiling_this_page * SCALE
                    px_ceiling = int(hard_ceiling_this_page * SCALE)
                    crop_top = max(0, int((y_start_pdf - 10) * SCALE))
                    crop_bot = min(
                        part1_img.height, min(px_ceiling, int((y_end_pdf + 5) * SCALE))
                    )
                    part1_crop = part1_img.crop(
                        (0, crop_top, part1_img.width, crop_bot)
                    )

                    final_img = part1_crop

                    if continues_to_next_page:
                        # Stitch with the top region of the next page.
                        # The stitch ends STRICTLY before the next question header.
                        next_page = doc[page_num + 1]
                        next_page_h = next_page.rect.height
                        part2_img = _render_page_image(next_page, SCALE)

                        # Hard ceiling on next page = next question's y_start - 3
                        next_hard_ceiling = (
                            (next_span["y_start"] - 3) if next_span else next_page_h
                        )
                        cont_end_pdf = _expand_bbox_for_graphics(
                            next_page,
                            0,
                            next_hard_ceiling,
                            next_page_h,
                            next_hard_ceiling,
                        )
                        cont_bot_px = min(
                            part2_img.height, int((cont_end_pdf + 5) * SCALE)
                        )
                        part2_crop = part2_img.crop(
                            (0, 0, part2_img.width, cont_bot_px)
                        )

                        # Vertically concatenate part1 and part2
                        total_h = part1_crop.height + part2_crop.height
                        combined = Image.new(
                            "RGB", (part1_crop.width, total_h), (255, 255, 255)
                        )
                        combined.paste(part1_crop, (0, 0))
                        combined.paste(part2_crop, (0, part1_crop.height))
                        final_img = combined

                    # Encode to JPEG bytes and upload
                    if final_img.height > 30:
                        buf = io.BytesIO()
                        final_img.save(buf, format="JPEG", quality=88)
                        img_bytes = buf.getvalue()

                        safe_q = span["q_num"].replace(".", "_").replace(" ", "_")
                        public_id = f"{document_id}_p{page_num + 1}_q{safe_q}"
                        image_url = upload_question_image_to_cloudinary(
                            img_bytes, public_id
                        )

                except Exception as crop_err:
                    print(
                        f"[PYQ Crop] Failed for Q{span['q_num']} page {page_num + 1}: {crop_err}"
                    )

            # Build the canonical question ID
            q_id = generate_canonical_question_id(
                course_code=course_code,
                document_id=document_id,
                question_number=span["q_num"],
                page_num=page_num,
                question_text=span["q_text"],
            )

            structured_questions.append(
                {
                    "id": q_id,
                    "question_number": span["q_num"],
                    "question_text": span["q_text"],
                    "co_tag": span["co_tag"],
                    "btl_tag": span["btl_tag"],
                    "marks": span["marks"],
                    "has_figure": span["has_figure"],
                    "course_code": course_code,
                    "exam_name": "PYQ Exam",
                    "image_url": image_url,
                }
            )

        doc.close()
        return structured_questions

    except Exception as e:
        import traceback

        print(f"Error in extract_pyq_structured: {e}")
        traceback.print_exc()
        return []


async def map_pyq_structured_to_kg(
    neo4j_driver, structured_questions: list[dict], document_id: str | None = None
):
    if not structured_questions:
        return

    c_code_overall = structured_questions[0]["course_code"]
    syllabus_context = fetch_course_topics(neo4j_driver, c_code_overall)
    print(
        f"[PYQ] Fetched {len(syllabus_context.split(chr(10)))} topics from syllabus. Running semantic mapping via LLM..."
    )

    mapping = await semantic_map_questions_to_topics(
        c_code_overall, syllabus_context, structured_questions
    )
    print(f"[PYQ] Semantic mapper returned {len(mapping)} matched questions.")

    def _update_structured_graph_tx(tx, q_id, q_text, btl, marks, has_fig, img_url, doc_id, c_code, q_num, t_data_list, co_tag):
        valid_topics = []
        if t_data_list:
            for t_data in t_data_list:
                if not isinstance(t_data, dict):
                    raise InvalidTopicMappingError(f"Topic item must be a dictionary, got {type(t_data)}")
                t_id = t_data.get("topic_id")
                if not t_id or not isinstance(t_id, str):
                    raise InvalidTopicMappingError(f"Missing or non-string topic_id: {t_id}")
                
                raw_conf = t_data.get("confidence", 0.8)
                if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
                    raise InvalidTopicMappingError(f"Non-numeric confidence {raw_conf!r} for topic {t_id}")
                conf = float(raw_conf)
                if not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
                    raise InvalidTopicMappingError(f"Out of bounds confidence {conf} for topic {t_id}")

                records = tx.run(
                    """
                    MATCH (t:Topic)
                    WHERE (t.topic_id = $t_id OR t.id = $t_id OR toString(id(t)) = $t_id)
                      AND (t.course_code = $c_code OR EXISTS { MATCH (:Course {code: $c_code})-[:HAS_UNIT]->(:Unit)-[:HAS_TOPIC]->(t) })
                    RETURN id(t) as t_internal_id
                    """,
                    t_id=t_id,
                    c_code=c_code
                ).data()

                if len(records) == 0:
                    raise InvalidTopicMappingError(f"Topic {t_id} not found in course {c_code}")
                if len(records) > 1:
                    raise InvalidTopicMappingError(f"Topic {t_id} is ambiguous ({len(records)} matches) in course {c_code}")

                valid_topics.append({"t_internal_id": records[0]["t_internal_id"], "conf": conf})

        # All validation passed! Mutate Question node
        q_result = tx.run(
            """
            MERGE (q:Question {id: $q_id})
            SET q.text = $q_text, 
                q.btl = $btl, 
                q.marks = $marks, 
                q.has_figure = $has_fig, 
                q.image_url = $image_url,
                q.document_id = $doc_id,
                q.course_code = $c_code,
                q.question_number = $q_num
            MERGE (c:Course {code: $c_code})
            RETURN id(q) as internal_id
            """,
            q_id=q_id,
            c_code=c_code,
            q_text=q_text,
            btl=btl,
            marks=marks,
            has_fig=has_fig,
            doc_id=doc_id,
            image_url=img_url,
            q_num=q_num
        ).single()

        if not q_result:
            raise RuntimeError("Failed to merge Question node")
        internal_id = q_result["internal_id"]

        if valid_topics:
            tx.run(
                """
                MATCH (q) WHERE id(q) = $internal_id
                OPTIONAL MATCH (q)-[r1:TESTS_TOPIC]->()
                OPTIONAL MATCH (q)-[r2:BELONGS_TO]->(:Course)
                DELETE r1, r2
                """,
                internal_id=internal_id
            )
            for vt in valid_topics:
                tx.run(
                    """
                    MATCH (q) WHERE id(q) = $internal_id
                    MATCH (t) WHERE id(t) = $t_internal_id
                    MERGE (q)-[r:TESTS_TOPIC]->(t)
                    SET r.confidence = $conf, r.review_status = 'pending_review'
                    """,
                    internal_id=internal_id,
                    t_internal_id=vt["t_internal_id"],
                    conf=vt["conf"]
                )
        else:
            tx.run(
                """
                MATCH (q) WHERE id(q) = $internal_id
                MATCH (c:Course {code: $c_code})
                OPTIONAL MATCH (q)-[r1:TESTS_TOPIC]->()
                OPTIONAL MATCH (q)-[r2:BELONGS_TO]->(:Course)
                DELETE r1, r2
                MERGE (q)-[:BELONGS_TO]->(c)
                """,
                internal_id=internal_id,
                c_code=c_code,
            )

        if co_tag:
            tx.run(
                """
                MATCH (q) WHERE id(q) = $internal_id
                MATCH (c:Course {code: $c_code})
                MERGE (co:CourseOutcome {id: $co_id, course_code: $c_code})
                MERGE (q)-[:MAPPED_TO_CO]->(co)
                MERGE (co)-[:BELONGS_TO]->(c)
                """,
                internal_id=internal_id,
                c_code=c_code,
                co_id=co_tag
            )
        return True

    with neo4j_driver.session() as session:
        for i, q in enumerate(structured_questions):
            q_id = q.get("id")
            if not q_id:
                q_id = generate_canonical_question_id(
                    course_code=q.get("course_code", c_code_overall),
                    document_id=document_id,
                    question_number=q.get("question_number"),
                    page_num=q.get("page_num"),
                    question_text=q.get("question_text"),
                )
            q_text = q["question_text"]
            btl = q["btl_tag"]
            marks = q["marks"]
            has_fig = q["has_figure"]
            co_tag = q["co_tag"]
            c_code = q["course_code"]
            image_url = q.get("image_url", "")
            q_num = q.get("question_number", "")

            mapped_data = mapping.get(i, [])
            try:
                session.execute_write(
                    _update_structured_graph_tx,
                    q_id,
                    q_text,
                    btl,
                    marks,
                    has_fig,
                    image_url,
                    str(document_id) if document_id else "",
                    c_code,
                    q_num,
                    mapped_data,
                    co_tag
                )
            except InvalidTopicMappingError as e:
                print(f"[Graph] Invalid topic mapping proposal for structured question {q_id}: {e}. Aborted without modifying existing mappings.")
            except Exception as e:
                print(f"[Graph] Database error during structured mapping update for question {q_id}: {e}. Skipping question to isolate failure without aborting batch.")


def execute_neo4j_pyq_search(neo4j_driver, question: str) -> list:
    """Search for PYQ questions in Neo4j by keyword matching on question text."""
    stop_words = {
        "get",
        "me",
        "a",
        "the",
        "all",
        "questions",
        "question",
        "problems",
        "problem",
        "solve",
        "solved",
        "solutions",
        "solution",
        "calculate",
        "write",
        "analyse",
        "analysis",
        "determine",
        "derive",
        "sketch",
        "draw",
        "obtain",
        "evaluate",
        "on",
        "about",
        "find",
        "show",
        "list",
        "give",
        "related",
        "are",
        "there",
        "any",
        "is",
        "what",
        "how",
        "why",
        "who",
        "where",
        "can",
        "you",
        "tell",
        "explain",
        "describe",
        "provide",
        "in",
        "of",
        "to",
        "for",
        "with",
        "and",
        "or",
        "not",
        "this",
        "that",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "would",
        "could",
        "should",
        "some",
        "from",
        "by",
        "an",
        "it",
        "they",
        "we",
        "he",
        "she",
        "which",
    }
    words = [
        w.strip(".,!?-'\"")
        for w in question.lower().split()
        if len(w) > 2 and w not in stop_words
    ]

    if not words:
        return []

    with neo4j_driver.session() as session:
        records = session.run(
            """
            MATCH (q:Question)
            WHERE all(word IN $words WHERE replace(toLower(q.text), "'", "") CONTAINS word)
            RETURN q.text AS q_text, q.btl AS btl, q.marks AS marks,
                   q.image_url AS image_url, coalesce(q.course_code, "") AS course_code,
                   q.question_number AS q_num
            LIMIT 20
        """,
            words=words,
        ).data()
    return records


# 8. Hybrid Search & Query Classification


async def hybrid_search_rrf(
    db, query_text: str, query_embedding: list[float], k=5, rrf_k=60
):
    """
    Performs Reciprocal Rank Fusion (RRF) using PostgreSQL pgvector (semantic)
    and tsvector (keyword) on the document_chunks table.
    """
    if not query_embedding:
        return []

    # Convert embedding list to string format for pgvector
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    # We use a FULL OUTER JOIN to combine ranks and calculate RRF score
    sql_query = text("""
        WITH vector_ranked AS (
            SELECT id, content, course_code,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> :embedding) AS rank
            FROM document_chunks
            WHERE embedding IS NOT NULL
            LIMIT 50
        ),
        text_ranked AS (
            SELECT id, content, course_code,
                   ROW_NUMBER() OVER (ORDER BY ts_rank(tsv_content, query) DESC) AS rank
            FROM document_chunks,
                 plainto_tsquery('english', :query_text) query
            WHERE tsv_content @@ query
            LIMIT 50
        ),
        rrf AS (
            SELECT
                COALESCE(v.id, t.id) AS id,
                COALESCE(v.content, t.content) AS content,
                COALESCE(v.course_code, t.course_code) AS course_code,
                (COALESCE(1.0/(:rrf_k + v.rank), 0.0) + COALESCE(1.0/(:rrf_k + t.rank), 0.0)) AS rrf_score
            FROM vector_ranked v
            FULL OUTER JOIN text_ranked t ON v.id = t.id
        )
        SELECT id, content, course_code, rrf_score
        FROM rrf
        ORDER BY rrf_score DESC
        LIMIT :k
    """)

    try:
        result = await db.execute(
            sql_query,
            {
                "embedding": embedding_str,
                "query_text": query_text,
                "rrf_k": rrf_k,
                "k": k,
            },
        )

        chunks = []
        for row in result:
            chunks.append(
                {
                    "id": row.id,
                    "content": row.content,
                    "course_code": row.course_code,
                    "rrf_score": row.rrf_score,
                }
            )
        return chunks
    except Exception as e:
        print(f"Hybrid search error: {e}")
        return []


def fetch_course_topics(neo4j_driver, course_code: str) -> str:
    with neo4j_driver.session() as session:
        records = session.run(
            """
            MATCH (c:Course {code: $c_code})-[:HAS_UNIT]->(u:Unit)-[:HAS_TOPIC]->(t:Topic)
            RETURN u.title AS unit, coalesce(t.topic_id, t.id, toString(id(t))) AS topic_id, t.name AS topic_name
            ORDER BY u.title, t.order
        """,
            c_code=course_code,
        ).data()

    if not records:
        return "No syllabus found."

    syllabus = []
    for r in records:
        syllabus.append(f"[{r['topic_id']}] {r['unit']} - {r['topic_name']}")
    return "\n".join(syllabus)


async def semantic_map_questions_to_topics(
    course_code: str, syllabus_context: str, questions: list, batch_size: int = 2
) -> dict:
    if not syllabus_context or syllabus_context == "No syllabus found.":
        return {}

    final_mapping = {}
    import re
    import json
    import httpx
    import asyncio
    from embedder import is_valid_embedding, cosine_similarity, safe_cosine_similarity
    
    try:
        from bm25 import BM25Okapi
    except ImportError:
        print("[Mapper] Warning: bm25.py not found. Attempting to proceed without it.")
        BM25Okapi = None

    def tokenize(text: str) -> list:
        return re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())

    parsed_topics = []
    for line in syllabus_context.split("\n"):
        match = re.match(r"\[(.*?)\]\s*(.*)", line.strip())
        if match:
            tid = match.group(1).strip()
            name = match.group(2).strip()
            parsed_topics.append({
                "tid": tid, 
                "name": name,
                "tokens": tokenize(name)
            })

    if not parsed_topics:
        return {}

    bm25_index = None
    if BM25Okapi:
        bm25_corpus = [t["tokens"] for t in parsed_topics]
        bm25_index = BM25Okapi(bm25_corpus)

    for t in parsed_topics:
        try:
            t["embedding"] = await _cached_embed_text(t["name"])
        except Exception as e:
            print(f"[Mapper] Topic '{t['name']}' embedding error: {e}")
            t["embedding"] = []

    sem = asyncio.Semaphore(batch_size)

    async def process_single_question(i: int, q: dict):
        try:
            q_text = q.get("question_text", q.get("text", ""))
            if not q_text:
                return i, []

            q_tokens = tokenize(q_text)
            
            # A. Lexical Retrieval (BM25)
            bm25_scores = [0.0] * len(parsed_topics)
            has_lexical = False
            try:
                if bm25_index:
                    bm25_scores = bm25_index.get_scores(q_tokens)
                    if any(s > 0 for s in bm25_scores):
                        has_lexical = True
            except Exception as e:
                print(f"[Mapper] Q{i} Lexical retrieval error: {e}")
                has_lexical = False

            # B. Semantic Retrieval (Isolated so failure preserves BM25 evidence)
            has_semantic = False
            semantic_scores = [(0.0, idx) for idx in range(len(parsed_topics))]
            try:
                q_emb = await _cached_embed_text(q_text)
                if is_valid_embedding(q_emb):
                    valid_cands = 0
                    for idx, t in enumerate(parsed_topics):
                        t_emb = t.get("embedding")
                        sim = safe_cosine_similarity(q_emb, t_emb)
                        if sim is not None:
                            semantic_scores[idx] = (sim, idx)
                            if sim > 0.3:
                                valid_cands += 1
                    if valid_cands > 0:
                        has_semantic = True
            except Exception as e:
                print(f"[Mapper] Q{i} Semantic retrieval error ({type(e).__name__}): {e}. Preserving BM25 evidence.")
                has_semantic = False

            if not has_lexical and not has_semantic:
                return i, []

            # C. Bounded Candidate Generation & RRF
            bm25_candidates = sorted([(score, idx) for idx, score in enumerate(bm25_scores) if score > 0], key=lambda x: x[0], reverse=True)[:20]
            semantic_candidates = sorted([(score, idx) for score, idx in semantic_scores if score > 0.3], key=lambda x: x[0], reverse=True)[:20]
            
            bm25_ranks = {idx: rank + 1 for rank, (score, idx) in enumerate(bm25_candidates)}
            semantic_ranks = {idx: rank + 1 for rank, (score, idx) in enumerate(semantic_candidates)}
            
            k = 60
            rrf_scores = []
            candidate_indices = set(bm25_ranks.keys()).union(set(semantic_ranks.keys()))
            for idx in candidate_indices:
                t = parsed_topics[idx]
                r_lex = bm25_ranks.get(idx)
                r_sem = semantic_ranks.get(idx)
                
                rrf_val = 0.0
                if r_lex:
                    rrf_val += 1.0 / (k + r_lex)
                if r_sem:
                    rrf_val += 1.0 / (k + r_sem)
                    
                rrf_scores.append({
                    "tid": t["tid"],
                    "name": t["name"],
                    "idx": idx,
                    "bm25_score": bm25_scores[idx],
                    "sem_score": semantic_scores[idx][0],
                    "rrf_score": rrf_val,
                    "r_lex": r_lex,
                    "r_sem": r_sem
                })
                
            rrf_scores.sort(key=lambda x: (x["rrf_score"], -x["idx"]), reverse=True)
            top_candidates = rrf_scores[:5]
            
            if not top_candidates:
                return i, []
                
            best_candidate = top_candidates[0]
            second_best = top_candidates[1] if len(top_candidates) > 1 else None
            
            # Deterministic Fallback Policy function using shared evidence acceptance helper
            def fallback_decision():
                ok, conf = check_evidence_acceptance(best_candidate)
                if ok:
                    print(f"[Mapper] Q{i} Evidence accepted: '{best_candidate['name']}' (conf={conf})")
                    return [{"topic_id": best_candidate["tid"], "confidence": conf}]
                print(f"[Mapper] Q{i} Insufficient evidence -> Course Fallback")
                return []

            rrf_margin = best_candidate["rrf_score"] - (second_best["rrf_score"] if second_best else 0.0)
            RRF_EARLY_EXIT_MARGIN = 0.0001
            
            acc_ok, acc_conf = check_evidence_acceptance(best_candidate)

            is_unanimous = (best_candidate.get("r_lex") == 1 and best_candidate.get("r_sem") == 1)
            is_high_margin = (
                rrf_margin >= RRF_EARLY_EXIT_MARGIN 
                and (best_candidate.get("r_lex") is not None and best_candidate["r_lex"] <= 3) 
                and (best_candidate.get("r_sem") is not None and best_candidate["r_sem"] <= 3)
            )

            # Early exit path MUST satisfy evidence acceptance policy and retrieval ranking consensus/margin
            if acc_ok and (is_unanimous or is_high_margin):
                print(f"[Mapper] Q{i} RRF Consensus (Skipping LLM) - Unanimous: {is_unanimous}, Margin: {rrf_margin:.5f}, SemScore: {best_candidate['sem_score']:.3f}")
                return i, [{"topic_id": best_candidate["tid"], "confidence": acc_conf}]
                
            candidate_topics_formatted = "\n".join([f"[{c['tid']}] {c['name']}" for c in top_candidates])

            prompt = f"""You are an expert curriculum alignment system.
Map the following exam question to the single most relevant topic ID from the candidate list.

Candidate Topics:
{candidate_topics_formatted}

Exam Question:
"{q_text}"

Rules:
1. Output MUST be valid JSON.
2. "topic_id" must be the exact ID string of the most relevant topic, or null if none are relevant.
3. "reason" must be a brief justification.
4. "is_ambiguous" must be a true/false boolean (true if multiple topics apply strongly or if evidence is conflicting).
5. "confidence" must be a numeric float value between 0.0 and 1.0 representing your confidence in this decision.

Example Output:
{{"topic_id": "T1", "reason": "Question explicitly asks for binary search.", "is_ambiguous": false, "confidence": 0.9}}
"""
            
            async with sem:
                try:
                    async with httpx.AsyncClient(headers={"ngrok-skip-browser-warning": "true"}, timeout=120.0) as client:
                        resp = await client.post(
                            f"{OLLAMA_BASE_URL}/api/generate",
                            json={
                                "model": OLLAMA_MODEL,
                                "prompt": prompt,
                                "stream": False,
                                "format": "json",
                                "options": {"temperature": 0.0, "num_predict": 512},
                            },
                        )
                        resp.raise_for_status()
                        res_text = resp.json().get("response", "{}")
                        
                        res_text_clean = re.sub(r"<think>.*?</think>", "", res_text, flags=re.DOTALL).strip()
                        match = re.search(r"\{.*?\}", res_text_clean, re.DOTALL)
                        clean_json = match.group(0) if match else res_text_clean
                        if not clean_json.strip():
                            clean_json = "{}"
                            
                        parsed = json.loads(clean_json)
                        llm_selected_tid = parsed.get("topic_id")
                        raw_ambiguous = parsed.get("is_ambiguous")
                        
                        # PRIORITY 5: STRICT LLM OUTPUT VALIDATION
                        if not isinstance(raw_ambiguous, bool):
                            print(f"[Mapper] Q{i} LLM returned invalid non-boolean is_ambiguous -> Applying fallback")
                            return i, fallback_decision()
                            
                        if raw_ambiguous is True:
                            print(f"[Mapper] Q{i} LLM returned is_ambiguous=true -> Applying fallback")
                            return i, fallback_decision()
                            
                        if not isinstance(llm_selected_tid, str) or not llm_selected_tid.strip():
                            print(f"[Mapper] Q{i} LLM returned missing/non-string topic_id -> Applying fallback")
                            return i, fallback_decision()
                            
                        if llm_selected_tid not in [c["tid"] for c in top_candidates]:
                            print(f"[Mapper] Q{i} LLM returned invalid topic_id {llm_selected_tid} not in top candidates -> Applying fallback")
                            return i, fallback_decision()
                            
                        if "confidence" not in parsed:
                            print(f"[Mapper] Q{i} LLM returned missing confidence -> Applying fallback")
                            return i, fallback_decision()
                            
                        raw_conf = parsed.get("confidence")
                        if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
                            print(f"[Mapper] Q{i} LLM returned non-numeric confidence {raw_conf!r} -> Applying fallback")
                            return i, fallback_decision()
                            
                        confidence = float(raw_conf)
                        if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
                            print(f"[Mapper] Q{i} LLM returned out of bounds confidence {confidence} -> Applying fallback")
                            return i, fallback_decision()
                            
                        LLM_ACCEPTANCE_THRESHOLD = 0.6
                        if confidence < LLM_ACCEPTANCE_THRESHOLD:
                            print(f"[Mapper] Q{i} LLM confidence too low ({confidence}) -> Applying fallback")
                            return i, fallback_decision()
                            
                        selected_cand = next((c for c in top_candidates if c["tid"] == llm_selected_tid), None)
                        if not selected_cand:
                            return i, fallback_decision()
                            
                        print(f"[Mapper] Q{i} Selective LLM Accepted: '{selected_cand['name']}' (conf: {confidence})")
                        return i, [{"topic_id": llm_selected_tid, "confidence": confidence}]

                except Exception as e:
                    print(f"[Mapper] Q{i} LLM failed ({type(e).__name__}) -> Applying fallback")
                    return i, fallback_decision()

        except Exception as e:
            print(f"[Mapper] Q{i} Unexpected failure ({type(e).__name__}): {e}. Abstaining for this question to isolate failure.")
            return i, []

    tasks = [process_single_question(i, q) for i, q in enumerate(questions)]
    results = await asyncio.gather(*tasks)

    for i, res in results:
        final_mapping[i] = res

    return final_mapping

