import pytest
from intent_router import classify_query_intent, intent_router

@pytest.mark.asyncio
async def test_classify_problem_list():
    # Should map to PROBLEM_LIST
    intent = await classify_query_intent("give me a list of practice problems on superposition theorem")
    assert intent == "PROBLEM_LIST"

@pytest.mark.asyncio
async def test_classify_prereq():
    # Should map to MULTI_HOP_PREREQ
    intent = await classify_query_intent("what should I know before taking embedded systems")
    assert intent == "MULTI_HOP_PREREQ"

@pytest.mark.asyncio
async def test_classify_syllabus():
    # Should map to SIMPLE_CURRICULUM
    intent = await classify_query_intent("what are the topics in unit 2 of physics")
    assert intent == "SIMPLE_CURRICULUM"
