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


def fetch_all_problems_by_topic(neo4j_driver, topic: str) -> list[dict]:
    """
    Executes a Cypher query to retrieve all Question nodes associated with a specific Topic.
    Returns the data as a clean Python list of dictionaries.
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

    # Expand keywords to cover common academic variations (e.g. division vs divider, nodal vs node)
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
            WHERE ALL(group IN $word_groups WHERE ANY(term IN group WHERE replace(toLower(q.text), "'", "") CONTAINS term))
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
