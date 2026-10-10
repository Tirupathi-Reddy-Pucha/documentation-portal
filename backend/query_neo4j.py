import json
from embedder import embed_text, cosine_similarity

async def fetch_problems_by_topic_graph(neo4j_driver, topic: str) -> list[dict]:
    """
    PRIMARY Graph RAG retrieval: semantic Topic embedding -> TESTS_TOPIC traversal.
    Falls back to string-match on Topic.name if no embedding hits found.
    Returns [] when no TESTS_TOPIC edges exist yet (topic mapper not run).
    """
    topic_ids = []
    topic_emb = await embed_text(topic)
    if topic_emb:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (t:Topic) WHERE t.embedding IS NOT NULL "
                "RETURN t.id AS id, t.embedding AS emb"
            )
            candidates = []
            for r in result:
                try:
                    t_emb = json.loads(r["emb"])
                    sim = cosine_similarity(topic_emb, t_emb)
                    if sim >= 0.65:
                        candidates.append((sim, r["id"]))
                except Exception:
                    pass
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                topic_ids = [tid for _, tid in candidates[:3]]

    with neo4j_driver.session() as session:
        if topic_ids:
            result = session.run(
                """
                MATCH (t:Topic) WHERE t.id IN $topic_ids
                MATCH (q:Question)-[r:TESTS_TOPIC]->(t)
                WHERE r.review_status IN ['approved', 'pending_review']
                RETURN q.id AS id, q.question_number AS question_number,
                       q.text AS text, q.marks AS marks, q.btl AS btl,
                       q.image_url AS image_url, q.course_code AS course_code,
                       t.name AS topic_name, r.confidence AS mapping_confidence
                ORDER BY mapping_confidence DESC, toInteger(question_number) ASC
                """,
                topic_ids=topic_ids,
            )
            data = result.data()
            if data:
                return data

        # String-match fallback on Topic.name
        result = session.run(
            """
            MATCH (t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($topic)
               OR toLower(t.normalized_name) CONTAINS toLower($topic)
            MATCH (q:Question)-[r:TESTS_TOPIC]->(t)
            WHERE r.review_status IN ['approved', 'pending_review']
            RETURN q.id AS id, q.question_number AS question_number,
                   q.text AS text, q.marks AS marks, q.btl AS btl,
                   q.image_url AS image_url, q.course_code AS course_code,
                   t.name AS topic_name, r.confidence AS mapping_confidence
            ORDER BY mapping_confidence DESC
            """,
            topic=topic,
        )
        return result.data()



import json
from embedder import embed_text, cosine_similarity

# =============================================================================
# Graph RAG — primary retrieval path via TESTS_TOPIC
# =============================================================================

async def fetch_problems_by_topic_graph(neo4j_driver, topic: str) -> list[dict]:
    """
    PRIMARY retrieval path for topic-based PYQ lookup using Semantic Search.
    
    Traverses: Topic <-[TESTS_TOPIC]- Question
    Returns only auto-approved mappings (review_status = "approved").
    """
    topic_ids_to_query = []
    
    # 1. Try semantic matching first
    topic_emb = await embed_text(topic)
    if topic_emb:
        with neo4j_driver.session() as session:
            topics_result = session.run("MATCH (t:Topic) WHERE t.embedding IS NOT NULL RETURN t.id AS id, t.embedding AS emb")
            best_topics = []
            for r in topics_result:
                try:
                    t_emb = json.loads(r["emb"])
                    sim = cosine_similarity(topic_emb, t_emb)
                    if sim >= 0.65:  # threshold for similarity
                        best_topics.append((sim, r["id"]))
                except Exception:
                    pass
            
            if best_topics:
                best_topics.sort(key=lambda x: x[0], reverse=True)
                topic_ids_to_query = [t_id for _, t_id in best_topics[:3]]

    with neo4j_driver.session() as session:
        if topic_ids_to_query:
            result = session.run("""
                MATCH (t:Topic)
                WHERE t.id IN $topic_ids
                MATCH (q:Question)-[r:TESTS_TOPIC]->(t)
                WHERE r.review_status IN ['approved', 'pending_review']
                OPTIONAL MATCH (q)-[:MAPPED_TO_CO]->(co:CourseOutcome)
                OPTIONAL MATCH (q)-[:EXTRACTED_FROM]->(doc:Document)
                WITH q, doc, collect(DISTINCT co.id) AS course_outcomes, collect(DISTINCT t.name) AS topics, max(r.confidence) AS max_confidence
                WITH q.text AS text, collect(q.id)[0] AS id, collect(q.question_number)[0] AS question_number, collect(q.marks)[0] AS marks, collect(q.btl)[0] AS btl, collect(q.image_url)[0] AS image_url, collect(q.course_code)[0] AS course_code, collect(course_outcomes[0])[0] AS co_id, collect(topics[0])[0] AS topic_name, max(max_confidence) AS mapping_confidence, collect(coalesce(doc.id, ''))[0] AS source_document
                RETURN id, question_number, text, marks, btl, image_url, course_code, co_id, topic_name, mapping_confidence, source_document
                ORDER BY mapping_confidence DESC, toInteger(question_number) ASC
            """, topic_ids=topic_ids_to_query)
            data = result.data()
            if data: return data

        # Fallback to pure string matching
        result = session.run("""
            MATCH (t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($topic)
               OR toLower(t.normalized_name) CONTAINS toLower($topic)
            MATCH (q:Question)-[r:TESTS_TOPIC]->(t)
            WHERE r.review_status IN ['approved', 'pending_review']
            OPTIONAL MATCH (q)-[:MAPPED_TO_CO]->(co:CourseOutcome)
            OPTIONAL MATCH (q)-[:EXTRACTED_FROM]->(doc:Document)
            WITH q, doc, collect(DISTINCT co.id) AS course_outcomes, collect(DISTINCT t.name) AS topics, max(r.confidence) AS max_confidence
            WITH q.text AS text, collect(q.id)[0] AS id, collect(q.question_number)[0] AS question_number, collect(q.marks)[0] AS marks, collect(q.btl)[0] AS btl, collect(q.image_url)[0] AS image_url, collect(q.course_code)[0] AS course_code, collect(course_outcomes[0])[0] AS co_id, collect(topics[0])[0] AS topic_name, max(max_confidence) AS mapping_confidence, collect(coalesce(doc.id, ''))[0] AS source_document
            RETURN id, question_number, text, marks, btl, image_url, course_code, co_id, topic_name, mapping_confidence, source_document
            ORDER BY mapping_confidence DESC, toInteger(question_number) ASC
        """, topic=topic)
        return result.data()


# =============================================================================
# Keyword fallback — existing logic preserved
# =============================================================================

def fetch_all_problems_by_topic(neo4j_driver, topic: str) -> list[dict]:
    """
    FALLBACK retrieval path — pure keyword search on Question.text.
    Used when no approved TESTS_TOPIC relationships exist for the topic.
    """
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
        "rule",
        "rules",
        "law",
        "laws",
        "theorem",
        "theorems",
        "method",
        "methods",
        "principle",
        "principles",
        "formula",
        "formulas",
        "technique",
        "techniques",
        "circuit",
        "circuits",
        "system",
        "systems",
        "example",
        "examples",
        "using",
    }

    # Extract clean search terms
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
        # Match Question nodes where every keyword group has at least one term matching in question text.
        # No LIMIT clause: fetches 100% of matching questions across all uploaded documents.
        result = session.run(
            """
            MATCH (q:Question)-[:BELONGS_TO]->(c:Course)
            WHERE ALL(group IN $word_groups
                      WHERE ANY(term IN group
                                WHERE replace(toLower(q.text), "'", "") CONTAINS term))
               OR replace(toLower(q.text), "'", "") CONTAINS toLower($topic)
            RETURN q.id AS id, 
                   q.question_number AS question_number, 
                   q.text AS text, 
                   q.image_url AS image_url, 
                   c.code AS course_code
        """,
            word_groups=word_groups,
            topic=topic,
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
            OPTIONAL MATCH (q:Question)-[:BELONGS_TO]->(c)
            OPTIONAL MATCH (q)-[:MAPPED_TO_CO]->(co:CourseOutcome)
            OPTIONAL MATCH (q)-[tr:TESTS_TOPIC]->(t2:Topic)
            WITH c, u, t, q, co, t2, tr
            LIMIT $limit
            RETURN c, u, t, q, co, t2,
                   tr.confidence AS topic_confidence,
                   tr.review_status AS topic_review_status
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
                add_node(t.get("id"), "Topic",
                         {"name": t.get("name", ""), "approved": t.get("approved", False)})
                if u:
                    uid = u.get("id") or u.get("title", "")
                    edges.append({"from": uid, "to": t.get("id"), "type": "HAS_TOPIC"})
            if q:
                qid = q.get("id", "")
                add_node(qid, "Question",
                         {"text": (q.get("text") or "")[:80],
                          "question_number": q.get("question_number", ""),
                          "marks": q.get("marks"),
                          "btl": q.get("btl", ""),
                          "image_url": q.get("image_url", "")})
                if c:
                    edges.append({"from": qid, "to": c.get("code"), "type": "BELONGS_TO"})
            if co and q:
                qid = q.get("id", "")
                add_node(co.get("id"), "CourseOutcome", {"id": co.get("id", "")})
                edges.append({"from": qid, "to": co.get("id"), "type": "MAPPED_TO_CO"})
            if t2 and q:
                qid = q.get("id", "")
                add_node(t2.get("id"), "Topic",
                         {"name": t2.get("name", ""), "approved": t2.get("approved", False)})
                edges.append({
                    "from":          qid,
                    "to":            t2.get("id"),
                    "type":          "TESTS_TOPIC",
                    "confidence":    r.get("topic_confidence"),
                    "review_status": r.get("topic_review_status"),
                })

        return {"nodes": list(nodes.values()), "edges": edges}
