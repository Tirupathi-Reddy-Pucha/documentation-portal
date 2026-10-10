"""
intent_router.py — Semantic intent classification using embedding cosine similarity.
Singleton IntentRouter initialises once at startup.
"""
from embedder import embed_text, cosine_similarity

INTENT_DESCRIPTIONS = {
    "PROBLEM_LIST": (
        "Ask for a list of problems, practice questions, or past year exam questions "
        "on a specific academic topic or concept."
    ),
    "MULTI_HOP_PREREQ": (
        "Ask about prerequisites, what should I know before taking a course, "
        "or what is required or needed for a subject."
    ),
    "GRAPH_PYQ_MAPPING": (
        "Ask about Bloom's taxonomy level (BTL), course outcomes (CO), "
        "or how past year questions map to learning outcomes."
    ),
    "SIMPLE_CURRICULUM": (
        "Ask about syllabus, topics, units, objectives, evaluation pattern, "
        "textbooks, or the curriculum structure of a course."
    ),
    "SIMPLE_PYQ": (
        "Ask for a specific past year question, exam paper, midterm, "
        "end-semester question, or circuit problem."
    ),
    "GENERAL_QUERY": (
        "General open-ended concept question, explanation request, "
        "or academic query outside specific PYQ/syllabus lookup."
    ),
}


class IntentRouter:
    def __init__(self):
        self.intent_embeddings: dict = {}
        self.initialized: bool = False

    async def initialize(self):
        if self.initialized:
            return
        for intent, description in INTENT_DESCRIPTIONS.items():
            emb = await embed_text(description)
            if emb:
                self.intent_embeddings[intent] = emb
        self.initialized = True

    async def classify_intent(self, question: str) -> str:
        q_lower = question.lower()
        
        # Tier 1: Fast Keyword Routing
        if any(marker in q_lower for marker in [
            "problems on", "problem on", "questions on", "question on",
            "pyqs on", "pyq on", "problems about", "questions about",
            "list of practice problems", "get me questions"
        ]):
            return "PROBLEM_LIST"
        if any(k in q_lower for k in [
            "prerequisite", "before taking", "should i know", "requires", "needed for", "before choosing"
        ]):
            return "MULTI_HOP_PREREQ"
        if any(k in q_lower for k in [
            "btl", "co1", "co2", "co3", "co4", "co5", "bloom", "mapped to", "course outcome questions"
        ]):
            return "GRAPH_PYQ_MAPPING"
        if any(k in q_lower for k in [
            "syllabus", "topics", "units", "course outcomes", "objectives", "evaluation",
            "pattern", "textbook", "reference", "course code", "courses do i have", "electives"
        ]):
            return "SIMPLE_CURRICULUM"
        if any(k in q_lower for k in [
            "question", "questions", "pyq", "past year", "exam", "paper",
            "midterm", "mid-term", "endsem", "end-sem", "problem", "problems", "circuit"
        ]):
            return "SIMPLE_PYQ"

        # Tier 2: Semantic Embedding Fallback
        if not self.initialized:
            await self.initialize()
        q_emb = await embed_text(question)
        if not q_emb:
            return "GENERAL_QUERY"
        best_intent, best_score = "GENERAL_QUERY", -1.0
        for intent, t_emb in self.intent_embeddings.items():
            sim = cosine_similarity(q_emb, t_emb)
            if sim > best_score:
                best_score = sim
                best_intent = intent
        if best_score < 0.45:
            return "GENERAL_QUERY"
        return best_intent


intent_router = IntentRouter()


async def classify_query_intent(question: str) -> str:
    """Public wrapper. Drop-in replacement for the keyword classifier in main.py."""
    return await intent_router.classify_intent(question)
