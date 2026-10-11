import json
import logging
from embedder import embed_text, cosine_similarity

logger = logging.getLogger(__name__)

async def fetch_problems_by_topic_graph(neo4j_driver, topic: str, course_code: str = None) -> list[dict]:
    """
    PRIMARY Graph RAG retrieval: semantic Topic embedding -> TESTS_TOPIC traversal.
    """
    topic_ids_to_query = []
    retrieval_status = "initialized"
    
    # 1. Try semantic matching first
    topic_emb = await embed_text(topic)
    if not topic_emb:
        retrieval_status = "embedding_unavailable"
        print(f"[Retrieval] Status: {retrieval_status}")
    else:
        with neo4j_driver.session() as session:
            # Course bounding
            where_clause = "WHERE t.embedding IS NOT NULL"
            if course_code:
                where_clause += " AND t.course_code = $course_code"
                
            query = f"MATCH (t:Topic) {where_clause} RETURN coalesce(t.topic_id, t.id, toString(id(t))) AS id, t.embedding AS emb"
            topics_result = session.run(query, course_code=course_code)
            
            best_topics = []
            for r in topics_result:
                try:
                    t_emb_raw = r["emb"]
                    if isinstance(t_emb_raw, str):
                        t_emb = json.loads(t_emb_raw)
                    else:
                        t_emb = t_emb_raw
                        
                    if len(t_emb) == len(topic_emb):
                        sim = cosine_similarity(topic_emb, t_emb)
                        if sim >= 0.65:
                            best_topics.append((sim, r["id"]))
                except Exception as e:
                    print(f"[Retrieval] Embedding decode error for topic {r['id']}: {e}")
                    
            if best_topics:
                best_topics.sort(key=lambda x: x[0], reverse=True)
                topic_ids_to_query = [t_id for _, t_id in best_topics[:3]]
                retrieval_status = "semantic_success"
            else:
                retrieval_status = "no_sufficient_similarity"

    with neo4j_driver.session() as session:
        if topic_ids_to_query:
            result = session.run("""
                MATCH (t:Topic)
                WHERE (t.topic_id IN $topic_ids OR t.id IN $topic_ids OR toString(id(t)) IN $topic_ids)
                MATCH (q:Question)-[r:TESTS_TOPIC]->(t)
                WHERE r.review_status IN ['approved', 'pending_review']
                  AND ($course_code IS NULL OR q.course_code = $course_code)
                OPTIONAL MATCH (q)-[:MAPPED_TO_CO]->(co:CourseOutcome)
                OPTIONAL MATCH (q)-[:EXTRACTED_FROM]->(doc:Document)
                WITH q, doc, collect(DISTINCT co.id) AS course_outcomes, collect(DISTINCT t.name) AS topics, max(r.confidence) AS max_confidence
                WITH q.text AS text, collect(q.id)[0] AS id, collect(q.question_number)[0] AS question_number, collect(q.marks)[0] AS marks, collect(q.btl)[0] AS btl, collect(q.image_url)[0] AS image_url, collect(q.course_code)[0] AS course_code, collect(course_outcomes[0])[0] AS co_id, collect(topics[0])[0] AS topic_name, max(max_confidence) AS mapping_confidence, collect(coalesce(doc.id, ''))[0] AS source_document
                RETURN id, question_number, text, marks, btl, image_url, course_code, co_id, topic_name, mapping_confidence, source_document
                ORDER BY mapping_confidence DESC, toInteger(question_number) ASC
            """, topic_ids=topic_ids_to_query, course_code=course_code)
            data = result.data()
            if data:
                print(f"[Retrieval] Status: {retrieval_status} (found {len(data)} questions)")
                return data
            else:
                retrieval_status = "topic_found_no_questions_or_excluded"

        # Fallback to pure string matching
        where_clause_fallback = ""
        if course_code:
            where_clause_fallback = "AND t.course_code = $course_code"
            
        result = session.run(f"""
            MATCH (t:Topic)
            WHERE (toLower(t.name) CONTAINS toLower($topic) OR toLower(t.normalized_name) CONTAINS toLower($topic))
            {where_clause_fallback}
            MATCH (q:Question)-[r:TESTS_TOPIC]->(t)
            WHERE r.review_status IN ['approved', 'pending_review']
              AND ($course_code IS NULL OR q.course_code = $course_code)
            OPTIONAL MATCH (q)-[:MAPPED_TO_CO]->(co:CourseOutcome)
            OPTIONAL MATCH (q)-[:EXTRACTED_FROM]->(doc:Document)
            WITH q, doc, collect(DISTINCT co.id) AS course_outcomes, collect(DISTINCT t.name) AS topics, max(r.confidence) AS max_confidence
            WITH q.text AS text, collect(q.id)[0] AS id, collect(q.question_number)[0] AS question_number, collect(q.marks)[0] AS marks, collect(q.btl)[0] AS btl, collect(q.image_url)[0] AS image_url, collect(q.course_code)[0] AS course_code, collect(course_outcomes[0])[0] AS co_id, collect(topics[0])[0] AS topic_name, max(max_confidence) AS mapping_confidence, collect(coalesce(doc.id, ''))[0] AS source_document
            RETURN id, question_number, text, marks, btl, image_url, course_code, co_id, topic_name, mapping_confidence, source_document
            ORDER BY mapping_confidence DESC, toInteger(question_number) ASC
        """, topic=topic, course_code=course_code)
        
        data = result.data()
        if data:
            print(f"[Retrieval] Status: keyword_fallback_success (found {len(data)} questions)")
            return data
            
        print(f"[Retrieval] Status: no_matching_topic")
        return []


# =============================================================================
# Keyword fallback — existing logic preserved
# =============================================================================

def fetch_all_problems_by_topic(neo4j_driver, topic: str, course_code: str = None) -> list[dict]:
    """
    FALLBACK retrieval path — pure keyword search on Question.text.
    Used when no approved TESTS_TOPIC relationships exist for the topic.
    """
    stop_words = {
        "get", "me", "a", "the", "all", "questions", "question", "problems",
        "problem", "solve", "solved", "solutions", "solution", "calculate",
        "write", "analyse", "analysis", "determine", "derive", "sketch",
        "draw", "obtain", "evaluate", "on", "about", "find", "show", "list",
        "give", "related", "are", "there", "any", "is", "what", "how", "why",
        "who", "where", "can", "you", "tell", "explain", "describe", "provide",
        "in", "of", "to", "for", "with", "and", "or", "not", "this", "that",
        "do", "does", "did", "have", "has", "had", "would", "could", "should",
        "some", "from", "by", "an", "it", "they", "we", "he", "she", "which",
        "rule", "rules", "law", "laws", "theorem", "theorems", "method",
        "methods", "principle", "principles", "formula", "formulas",
        "technique", "techniques", "circuit", "circuits", "system", "systems",
        "example", "examples", "using",
    }

    words = [
        w.strip(".,!?-'\"")
        for w in topic.lower().split()
        if len(w) > 2 and w not in stop_words
    ]

    if not words:
        words = [topic.lower()]

    def expand_term(w: str) -> list[str]:
        w_clean = w.lower()
        if w_clean in ["division", "divider", "dividing"]:
            return ["division", "divider", "divid"]
        if w_clean in ["nodal", "node", "nodes"]:
            return ["nodal", "node"]
        if w_clean in ["mesh", "loop", "loops"]:
            return ["mesh", "loop"]
        if w_clean in ["thevenin", "thevenins"]:
            return ["thevenin"]
        if w_clean in ["norton", "nortons"]:
            return ["norton"]
        return [w_clean]

    word_groups = [expand_term(w) for w in words]

    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (q:Question)
            WHERE ($course_code IS NULL OR q.course_code = $course_code)
              AND (
                ALL(group IN $word_groups
                    WHERE ANY(term IN group
                              WHERE replace(toLower(q.text), "'", "") CONTAINS term))
                OR replace(toLower(q.text), "'", "") CONTAINS toLower($topic)
              )
            OPTIONAL MATCH (q)-[:BELONGS_TO]->(c:Course)
            RETURN q.id AS id, 
                   q.question_number AS question_number, 
                   q.text AS text, 
                   q.image_url AS image_url, 
                   coalesce(q.course_code, c.code, "") AS course_code
        """,
            word_groups=word_groups,
            topic=topic,
            course_code=course_code,
        )
        return result.data()


# =============================================================================
# Bidirectional graph visualisation query
# =============================================================================

def fetch_graph_for_course(neo4j_driver, course_code: str, limit: int = 200) -> dict:
    """
    Returns all nodes and relationships needed to visualise the full course graph.
    Includes BOTH outgoing (syllabus structure) and incoming (questions) relationships.

    Returns {"nodes": [...], "edges": [...]}
    """
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (c:Course {code: $code})
            OPTIONAL MATCH (c)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (u)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (q:Question)
                WHERE q.course_code = $code
            OPTIONAL MATCH (q)-[br:BELONGS_TO]->(c)
            OPTIONAL MATCH (q)-[:MAPPED_TO_CO]->(co:CourseOutcome)
            OPTIONAL MATCH (q)-[tr:TESTS_TOPIC]->(t2:Topic)
            WITH c, u, t, q, br, co, t2, tr
            LIMIT $limit
            RETURN c, u, t, q, br, co, t2, tr
        """, code=course_code, limit=limit)

        nodes = {}
        edges = []

        def add_node(node_id, label, props):
            if node_id and node_id not in nodes:
                nodes[node_id] = {"id": node_id, "label": label, **props}

        for record in result:
            r = dict(record)
            c  = r.get("c")
            u  = r.get("u")
            t  = r.get("t")
            q  = r.get("q")
            co = r.get("co")
            t2 = r.get("t2")

            if c:
                add_node(c.get("code"), "Course",
                         {"name": c.get("name", ""), "code": c.get("code", "")})
            if u:
                uid = u.get("id") or u.get("title", "")
                add_node(uid, "Unit",
                         {"title": u.get("title", ""), "number": u.get("number", "")})
                if c:
                    edges.append({"from": c.get("code"), "to": uid, "type": "HAS_UNIT"})
            if t:
                tid = t.get("topic_id") or t.get("id") or (str(t.element_id) if hasattr(t, 'element_id') else str(t.id))
                add_node(tid, "Topic",
                         {"name": t.get("name", ""), "approved": t.get("approved", False)})
                if u:
                    uid = u.get("id") or u.get("title", "")
                    edges.append({"from": uid, "to": tid, "type": "HAS_TOPIC"})
            if q:
                qid = q.get("id", "")
                add_node(qid, "Question",
                         {"text": (q.get("text") or "")[:80],
                          "question_number": q.get("question_number", ""),
                          "marks": q.get("marks"),
                          "btl": q.get("btl", ""),
                          "image_url": q.get("image_url", "")})
                if r.get("br") and c:
                    edges.append({"from": qid, "to": c.get("code"), "type": "BELONGS_TO"})
            if co and q:
                qid = q.get("id", "")
                add_node(co.get("id"), "CourseOutcome", {"id": co.get("id", "")})
                edges.append({"from": qid, "to": co.get("id"), "type": "MAPPED_TO_CO"})
            if t2 and q:
                qid = q.get("id", "")
                tid2 = t2.get("topic_id") or t2.get("id") or (str(t2.element_id) if hasattr(t2, 'element_id') else str(t2.id))
                add_node(tid2, "Topic",
                         {"name": t2.get("name", ""), "approved": t2.get("approved", False)})
                edges.append({
                    "from": qid,
                    "to": tid2,
                    "type": "TESTS_TOPIC",
                })

        return {"nodes": list(nodes.values()), "edges": edges}


def extract_semester_number(text: str) -> int | None:
    """Extract semester number from text (e.g. '3rd semester', 'semester 4', 'sem 5')."""
    sems = extract_semester_numbers(text)
    return sems[0] if sems else None


def extract_semester_numbers(text: str) -> list[int]:
    """Extract semester numbers from text including year expressions ('first year' -> [1, 2])."""
    import re
    t = text.lower()
    year_map = {
        "1st year": [1, 2], "first year": [1, 2], "year 1": [1, 2],
        "2nd year": [3, 4], "second year": [3, 4], "year 2": [3, 4],
        "3rd year": [5, 6], "third year": [5, 6], "year 3": [5, 6],
        "4th year": [7, 8], "fourth year": [7, 8], "final year": [7, 8], "year 4": [7, 8]
    }
    for k, sems in year_map.items():
        if k in t:
            return sems

    m = re.search(r'\b(?:sem(?:ester)?\s*(\d)|(\d)(?:st|nd|rd|th)?\s*sem(?:ester)?)\b', t)
    if m:
        val = m.group(1) or m.group(2)
        try:
            return [int(val)]
        except ValueError:
            pass
    # Words to numbers
    word_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}
    for word, num in word_map.items():
        if f"{word} semester" in t or f"{word} sem" in t:
            return [num]
    return []


def execute_graph_semester_query(neo4j_driver, sem_num) -> list[dict]:
    """Retrieves all courses offered in the requested semester(s). Accepts int or list[int]."""
    sems = [sem_num] if isinstance(sem_num, int) else list(sem_num)
    with neo4j_driver.session() as session:
        return session.run("""
            MATCH (s:Semester)-[:OFFERS]->(c:Course)
            WHERE s.number IN $sems
            RETURN s.number AS semester,
                   c.code AS code,
                   coalesce(nullif(c.name, ''), c.title, c.code) AS name,
                   c.credits AS credits,
                   c.category AS category
            ORDER BY s.number ASC, c.code ASC
        """, sems=sems).data()


def execute_graph_elective_query(neo4j_driver, stream_hint: str) -> list[dict]:
    """Retrieves courses under matching elective baskets."""
    with neo4j_driver.session() as session:
        return session.run("""
            MATCH (eb:ElectiveBasket)-[:CONTAINS]->(c:Course)
            WHERE toLower(eb.name) CONTAINS toLower($hint)
               OR any(word IN split(toLower($hint), ' ') WHERE toLower(eb.name) CONTAINS word AND size(word) > 2)
            RETURN eb.name AS basket,
                   c.code AS code,
                   coalesce(nullif(c.name, ''), c.title, c.code) AS name,
                   c.credits AS credits
            ORDER BY eb.name, c.code ASC
            LIMIT 30
        """, hint=stream_hint).data()


def resolve_course_code_or_name(neo4j_driver, text: str) -> dict | None:
    """Resolves a query text to a Course node by exact code, acronym, or name matching."""
    import re
    # 1. Check exact course code (e.g. 23CSE211, 23EEE104, 22ADM111)
    code_m = re.search(r'\b(\d{2}[A-Z]{3}\d{3}[A-Z]?)\b', text, re.IGNORECASE)
    with neo4j_driver.session() as session:
        if code_m:
            rec = session.run("""
                MATCH (c:Course {code: $code})
                RETURN c.code AS code, coalesce(nullif(c.name, ''), c.title, c.code) AS name, c.credits AS credits LIMIT 1
            """, code=code_m.group(1).upper()).single()
            if rec:
                return dict(rec)

        # 2. Known common acronyms
        acronyms = {
            "daa": "23CSE211", "dsa": "23CSE203", "os": "23CSE214", "coa": "23CSE213",
            "toc": "23CSE303", "cn": "23CSE302", "ml": "23CSE301", "dbms": "23CSE202",
            "crypto": "23CSE331", "cryptography": "23CSE331", "se": "23CSE311", "ai": "23CSE361"
        }
        for token in re.findall(r'\b[a-zA-Z]+\b', text.lower()):
            if token in acronyms:
                rec = session.run("""
                    MATCH (c:Course {code: $code})
                    RETURN c.code AS code, coalesce(nullif(c.name, ''), c.title, c.code) AS name, c.credits AS credits LIMIT 1
                """, code=acronyms[token]).single()
                if rec:
                    return dict(rec)

        # 3. Direct Substring / Phrase Match against all Course names
        all_courses = session.run("""
            MATCH (c:Course)
            RETURN c.code AS code, coalesce(nullif(c.name, ''), c.title, c.code) AS name, c.credits AS credits
        """).data()
        
        q_clean = " " + re.sub(r'[^a-z0-9\s]', ' ', text.lower()) + " "
        
        # Sort courses by length of course name descending so "DISTRIBUTED SYSTEMS" matches before "SYSTEMS"
        phrase_matches = []
        for c in all_courses:
            c_name = c["name"]
            if not c_name:
                continue
            c_norm = " " + re.sub(r'[^a-z0-9\s]', ' ', c_name.lower()) + " "
            if c_norm in q_clean or c_name.lower() in text.lower():
                phrase_matches.append(c)
        if phrase_matches:
            phrase_matches.sort(key=lambda x: len(x["name"]), reverse=True)
            return phrase_matches[0]

        # 4. Token-based matching with comprehensive stop words
        STOP = {
            "what", "are", "the", "prerequisites", "prerequisite", "for", "course", "subject",
            "before", "taking", "choosing", "is", "of", "in", "to", "and", "references", "reference",
            "textbooks", "textbook", "books", "syllabus", "topics", "units", "unit", "evaluation",
            "pattern", "marks", "credits", "code", "codes", "name", "names", "title", "full",
            "tell", "me", "about", "give", "show", "list", "get", "prescribed", "recommended",
            "read", "reading", "study", "taken", "take", "first", "second", "third", "year",
            "which", "who", "will"
        }
        words = [w for w in re.findall(r'[a-zA-Z0-9]+', text.lower()) if len(w) > 2 and w not in STOP]
        if words:
            # Rank candidates by how many query keywords they match
            scored_candidates = []
            for c in all_courses:
                c_name_lower = (c["name"] or "").lower()
                c_tokens = set(re.findall(r'[a-zA-Z0-9]+', c_name_lower))
                matched_cnt = sum(1 for w in words if w in c_name_lower or w in c_tokens)
                if matched_cnt > 0:
                    scored_candidates.append((matched_cnt, len(c_name_lower), c))
            
            if scored_candidates:
                # Prioritize: most matched words first, then closest name length
                scored_candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
                return scored_candidates[0][2]

    return None


def find_all_matching_courses(neo4j_driver, text: str) -> list[dict]:
    """Returns all courses matching keywords in query for disambiguation."""
    import re
    with neo4j_driver.session() as session:
        # Check direct code
        code_m = re.search(r'\b(\d{2}[A-Z]{3}\d{3}[A-Z]?)\b', text, re.IGNORECASE)
        if code_m:
            c = session.run("""
                MATCH (c:Course {code: $code})
                RETURN c.code AS code, coalesce(nullif(c.name, ''), c.title, c.code) AS name,
                       c.credits AS credits, c.L AS L, c.T AS T, c.P AS P
            """, code=code_m.group(1).upper()).single()
            if c:
                return [dict(c)]

        STOP = {
            "what", "are", "the", "for", "course", "subject", "is", "of", "in", "to", "and",
            "code", "codes", "name", "names", "tell", "me", "get", "give", "show", "list"
        }
        words = [w for w in re.findall(r'[a-zA-Z0-9]+', text.lower()) if len(w) > 2 and w not in STOP]
        if not words:
            return []
        
        return session.run("""
            MATCH (c:Course)
            WHERE all(w IN $words WHERE toLower(c.name) CONTAINS w OR toLower(c.code) CONTAINS w)
            RETURN c.code AS code, coalesce(nullif(c.name, ''), c.title, c.code) AS name,
                   c.credits AS credits, c.L AS L, c.T AS T, c.P AS P
            ORDER BY size(c.name) ASC LIMIT 5
        """, words=words).data()


def execute_graph_prereq_chain_query(neo4j_driver, target_code: str) -> list[dict]:
    """Finds all multi-hop recursive prerequisites for a course."""
    with neo4j_driver.session() as session:
        return session.run("""
            MATCH (target:Course {code: $code})
            OPTIONAL MATCH path = (target)-[:REQUIRES*1..5]->(pre:Course)
            WITH target, pre, length(path) AS depth
            WHERE pre IS NOT NULL
            RETURN DISTINCT pre.code AS code,
                            coalesce(nullif(pre.name, ''), pre.title, pre.code) AS name,
                            depth
            ORDER BY depth DESC
        """, code=target_code).data()


def execute_graph_course_details_query(neo4j_driver, course_code: str) -> dict:
    """Retrieves textbooks, references, evaluation pattern, units, and credits."""
    with neo4j_driver.session() as session:
        res = session.run("""
            MATCH (c:Course {code: $code})
            OPTIONAL MATCH (c)-[:USES_TEXTBOOK]->(tb:Textbook)
            WITH c, tb ORDER BY tb.order ASC
            WITH c, collect(DISTINCT coalesce(tb.text, tb.title)) AS textbooks
            OPTIONAL MATCH (c)-[:HAS_REFERENCE]->(ref:Reference)
            WITH c, textbooks, ref ORDER BY ref.order ASC
            WITH c, textbooks, collect(DISTINCT coalesce(ref.text, ref.title)) AS references
            OPTIONAL MATCH (c)-[:HAS_EVALUATION_PATTERN]->(ep:EvaluationPattern)
            OPTIONAL MATCH (ep)-[:HAS_INTERNAL]->(it:Internal)
            OPTIONAL MATCH (ep)-[:HAS_EXTERNAL]->(et:External)
            OPTIONAL MATCH (c)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (u)-[:HAS_TOPIC]->(t:Topic)
            RETURN c.code AS code,
                   coalesce(nullif(c.name, ''), c.title, c.code) AS name,
                   c.credits AS credits,
                   c.L AS L, c.T AS T, c.P AS P,
                   [x IN textbooks WHERE x IS NOT NULL AND size(x) > 0] AS textbooks,
                   [x IN references WHERE x IS NOT NULL AND size(x) > 0] AS references,
                   ep.ratio AS eval_ratio,
                   collect(DISTINCT {name: it.name, marks: it.marks}) AS internal_assessments,
                   collect(DISTINCT {name: et.name, marks: et.marks}) AS external_assessments,
                   collect(DISTINCT {unit_num: u.number, unit_title: u.title, topic: t.name}) AS unit_topics
        """, code=course_code).single()
        return dict(res) if res else {}


def execute_graph_references_and_textbooks_query(neo4j_driver, course_code: str) -> dict:
    """Explicitly fetches ordered textbooks and references for a course."""
    with neo4j_driver.session() as session:
        tbs = session.run("""
            MATCH (c:Course {code: $code})-[:USES_TEXTBOOK]->(tb:Textbook)
            RETURN coalesce(tb.text, tb.title) AS text
            ORDER BY tb.order ASC
        """, code=course_code).data()

        refs = session.run("""
            MATCH (c:Course {code: $code})-[:HAS_REFERENCE]->(ref:Reference)
            RETURN coalesce(ref.text, ref.title) AS text
            ORDER BY ref.order ASC
        """, code=course_code).data()

        c_info = session.run("""
            MATCH (c:Course {code: $code})
            RETURN c.code AS code, coalesce(nullif(c.name, ''), c.title, c.code) AS name
        """, code=course_code).single()

        return {
            "code": c_info["code"] if c_info else course_code,
            "name": c_info["name"] if c_info else course_code,
            "textbooks": [t["text"] for t in tbs if t.get("text")],
            "references": [r["text"] for r in refs if r.get("text")]
        }

