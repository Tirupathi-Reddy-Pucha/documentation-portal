import pytest
from database import get_neo4j
from query_neo4j import fetch_problems_by_topic_graph
from intent_router import classify_query_intent

@pytest.mark.asyncio
async def test_neo4j_driver_connection():
    driver = get_neo4j()
    # verify connection by executing a simple query
    with driver.session() as session:
        result = session.run("RETURN 1 as num")
        record = result.single()
        assert record["num"] == 1

@pytest.mark.asyncio
async def test_full_pipeline_to_graph_rag():
    driver = get_neo4j()
    intent = await classify_query_intent("give me a list of practice problems on superposition theorem")
    assert intent == "PROBLEM_LIST"
    
    # We won't assert output length since DB might be empty in CI, 
    # but we ensure the function runs without throwing exceptions.
    problems = await fetch_problems_by_topic_graph(driver, "superposition theorem")
    assert isinstance(problems, list)
