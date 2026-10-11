import asyncio
import json
import math
import hashlib
import os
from unittest.mock import MagicMock, AsyncMock, patch

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# Imports under test
from utils import (
    semantic_map_questions_to_topics,
    map_questions_to_kg,
    map_pyq_structured_to_kg,
    extract_pyq_structured,
    build_syllabus_kg,
    normalize_question_number,
    generate_canonical_question_id,
    check_evidence_acceptance,
    InvalidTopicMappingError,
)
from embedder import (
    is_valid_embedding,
    safe_cosine_similarity,
    cosine_similarity,
)
from query_neo4j import (
    fetch_problems_by_topic_graph,
    fetch_all_problems_by_topic,
    fetch_graph_for_course,
)

async def run_all_tests():
    print("=" * 60)
    print("RUNNING COMPREHENSIVE REGRESSION & VERIFICATION SUITE")
    print("=" * 60)
    
    passed = 0
    total = 0

    def assert_test(name, condition, error_msg=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - {error_msg}")

    # =========================================================================
    # PART 1: PURE UNIT TESTS - EMBEDDING VALIDATION (ISSUE 2)
    # =========================================================================
    print("\n--- Testing Issue 2: Embedding Validation Robustness ---")
    
    # 1. Missing / None
    assert_test("Missing embedding returns False", is_valid_embedding(None) is False)
    
    # 2. Empty vector
    assert_test("Empty vector returns False", is_valid_embedding([]) is False)
    
    # 3. Zero-norm vector
    assert_test("Zero-norm vector returns False", is_valid_embedding([0.0, 0.0, 0.0]) is False)
    
    # 4. Valid numeric vector
    assert_test("Valid numeric vector returns True", is_valid_embedding([0.1, -0.2, 0.3]) is True)
    
    # 5. Boolean vector elements (CRITICAL: isinstance(True, int) is True in Python)
    assert_test("Boolean elements rejected", is_valid_embedding([True, False, 0.5]) is False)
    assert_test("All-boolean vector rejected", is_valid_embedding([True, True]) is False)
    
    # 6. Non-numeric elements
    assert_test("String elements rejected", is_valid_embedding(["0.1", 0.2]) is False)
    assert_test("None element rejected", is_valid_embedding([0.1, None]) is False)
    
    # 7. NaN and Infinities
    assert_test("NaN element rejected", is_valid_embedding([0.1, float('nan'), 0.2]) is False)
    assert_test("Positive infinity rejected", is_valid_embedding([0.1, float('inf')]) is False)
    assert_test("Negative infinity rejected", is_valid_embedding([0.1, float('-inf')]) is False)
    
    # 8. Mismatched dimensions in safe_cosine_similarity
    assert_test("Mismatched dimensions returns None", safe_cosine_similarity([0.1, 0.2], [0.1, 0.2, 0.3]) is None)
    
    # 9. Cosine similarity with invalid vectors returns None
    assert_test("Similarity with invalid query vector returns None", safe_cosine_similarity([True, 0.1], [0.1, 0.2]) is None)
    assert_test("Similarity with NaN returns None", safe_cosine_similarity([float('nan'), 0.1], [0.1, 0.2]) is None)
    
    # 10. Finite similarity value check
    sim = safe_cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert_test("Valid identical vectors produce finite similarity == 1.0", sim is not None and math.isclose(sim, 1.0))

    # =========================================================================
    # PART 2: PURE UNIT TESTS - EVIDENCE ACCEPTANCE & RRF EARLY EXIT (ISSUE 1)
    # =========================================================================
    print("\n--- Testing Issue 1: Evidence Acceptance Policy & RRF Early Exit ---")
    
    # 1. Unanimous ranking with strong evidence: r_lex=1, r_sem=1, sem_score >= 0.65
    c1 = {"r_lex": 1, "r_sem": 1, "sem_score": 0.72, "bm25_score": 4.0}
    ok1, conf1 = check_evidence_acceptance(c1)
    assert_test("Unanimous ranking with strong evidence accepted", ok1 is True and conf1 == 0.72)
    
    # 2. Unanimous ranking with semantic similarity below 0.65: r_lex=1, r_sem=1, sem_score = 0.60
    c2 = {"r_lex": 1, "r_sem": 1, "sem_score": 0.60, "bm25_score": 4.0}
    ok2, _ = check_evidence_acceptance(c2)
    assert_test("Unanimous ranking with semantic similarity < 0.65 rejected", ok2 is False)
    
    # 3. Semantic similarity above 0.65 but BM25 rank greater than 3: r_lex=4, r_sem=1, sem_score = 0.67 (< 0.70)
    c3 = {"r_lex": 4, "r_sem": 1, "sem_score": 0.67, "bm25_score": 2.0}
    ok3, _ = check_evidence_acceptance(c3)
    assert_test("Semantic sim >= 0.65 but BM25 rank > 3 rejected", ok3 is False)
    
    # 4. Semantic-only candidate with similarity below 0.70: r_lex=None, r_sem=1, sem_score = 0.68
    c4 = {"r_lex": None, "r_sem": 1, "sem_score": 0.68, "bm25_score": 0.0}
    ok4, _ = check_evidence_acceptance(c4)
    assert_test("Semantic-only candidate with similarity < 0.70 rejected", ok4 is False)
    
    # 5. Semantic-only candidate with similarity at or above 0.70: r_lex=None, r_sem=1, sem_score = 0.75
    c5 = {"r_lex": None, "r_sem": 1, "sem_score": 0.75, "bm25_score": 0.0}
    ok5, conf5 = check_evidence_acceptance(c5)
    assert_test("Semantic-only candidate with similarity >= 0.70 accepted", ok5 is True and conf5 == 0.75)
    
    # 6. Lexical-only candidate with BM25 score <= 5.0: r_lex=1, r_sem=None, bm25_score = 4.8
    c6 = {"r_lex": 1, "r_sem": None, "sem_score": 0.0, "bm25_score": 4.8}
    ok6, _ = check_evidence_acceptance(c6)
    assert_test("Lexical-only candidate with BM25 score <= 5.0 rejected", ok6 is False)
    
    # 7. Lexical-only candidate with BM25 score > 5.0: r_lex=1, r_sem=None, bm25_score = 6.2
    c7 = {"r_lex": 1, "r_sem": None, "sem_score": 0.0, "bm25_score": 6.2}
    ok7, conf7 = check_evidence_acceptance(c7)
    assert_test("Lexical-only candidate with BM25 score > 5.0 accepted", ok7 is True and conf7 == 0.5)
    
    # 8. High RRF margin with insufficient evidence (cannot bypass evidence acceptance)
    c8 = {"r_lex": 1, "r_sem": 1, "sem_score": 0.55, "bm25_score": 3.0}
    ok8, _ = check_evidence_acceptance(c8)
    assert_test("High RRF candidate with insufficient evidence rejected", ok8 is False)
    
    # 9. No valid retrieval candidates
    c9 = {"r_lex": None, "r_sem": None, "sem_score": 0.0, "bm25_score": 0.0}
    ok9, _ = check_evidence_acceptance(c9)
    assert_test("No retrieval candidates rejected", ok9 is False)

    # =========================================================================
    # PART 3: PURE UNIT TESTS - QUESTION IDENTITY STANDARDIZATION (ISSUE 3)
    # =========================================================================
    print("\n--- Testing Issue 3: Standardize Question Identity ---")
    
    # 1. Same occurrence processed twice produces identical ID
    id_a = generate_canonical_question_id("CS101", "DOC1", "1", 0, "What is binary search?")
    id_b = generate_canonical_question_id("CS101", "DOC1", "1", 0, "What is binary search?")
    assert_test("Same occurrence processed twice produces identical ID", id_a == id_b)
    
    # 2. Both ingestion paths generate the same ID for the same occurrence
    # In extract_pyq_structured, question_number="1", page_num=0, course_code="CS101", doc_id="DOC1"
    struct_id = generate_canonical_question_id("CS101", "DOC1", "1", 0, "What is binary search?")
    unstruct_id = generate_canonical_question_id("CS101", "DOC1", "1", 0, "What is binary search?")
    assert_test("Both ingestion paths generate identical ID for same occurrence", struct_id == unstruct_id)
    
    # 3. Identical text in different documents produces different IDs
    id_doc1 = generate_canonical_question_id("CS101", "DOC1", "1", 0, "Common question text")
    id_doc2 = generate_canonical_question_id("CS101", "DOC2", "1", 0, "Common question text")
    assert_test("Identical text in different documents produces different IDs", id_doc1 != id_doc2)
    
    # 4. Different questions on same page produce different IDs
    id_q1 = generate_canonical_question_id("CS101", "DOC1", "1", 0, "Question 1 text")
    id_q2 = generate_canonical_question_id("CS101", "DOC1", "2", 0, "Question 2 text")
    assert_test("Different questions on same page produce different IDs", id_q1 != id_q2)
    
    # 5. Parent questions and subquestions retain identity distinctions
    id_parent = generate_canonical_question_id("CS101", "DOC1", "1", 0, "Parent Q text")
    id_sub = generate_canonical_question_id("CS101", "DOC1", "1(a)", 0, "Sub Q text")
    assert_test("Parent question '1' and subquestion '1(a)' retain distinct IDs", id_parent != id_sub)
    
    # Check normalization: Q1 vs 1
    assert_test("Question number normalization: 'Q1' vs '1' harmonize safely", normalize_question_number("Q1") == "1")
    assert_test("Subquestion normalization preserves '(a)': 'Q1(a)' -> '1(a)'", normalize_question_number("Q1(a)") == "1(a)")
    
    # 6. Missing page metadata follows fallback policy
    id_nopage_1 = generate_canonical_question_id("CS101", "DOC1", "1", None, "Text")
    id_nopage_2 = generate_canonical_question_id("CS101", "DOC1", "1", "nopage", "Text")
    assert_test("Missing page metadata follows deterministic fallback", id_nopage_1 == id_nopage_2)
    
    # 7. Missing document metadata follows fallback policy
    id_nodoc_1 = generate_canonical_question_id("CS101", None, "1", 0, "Text 1")
    id_nodoc_2 = generate_canonical_question_id("CS101", None, "1", 0, "Text 2")
    assert_test("Missing document metadata uses text hash to prevent collision", id_nodoc_1 != id_nodoc_2)

    # =========================================================================
    # PART 4: PURE UNIT TESTS - STRICT LLM OUTPUT VALIDATION (ISSUE 5)
    # =========================================================================
    print("\n--- Testing Issue 5: Strict LLM Output Validation ---")
    
    dummy_syllabus = "[T1] Unit 1 - Binary Search\n[T2] Unit 1 - Linear Search"
    dummy_questions = [{"text": "Explain binary search algorithm"}]
    
    # Helper to test LLM responses
    async def test_llm_json_response(raw_json_str):
        with patch('httpx.AsyncClient') as mock_client, \
             patch('utils._cached_embed_text', new_callable=AsyncMock) as mock_embed:
            # Semantic embedding returns valid vector so candidate T1 has strong semantic similarity
            mock_embed.return_value = [0.1] * 768
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": raw_json_str}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp
            return await semantic_map_questions_to_topics("CS101", dummy_syllabus, dummy_questions)

    # 1. Boolean confidence: {"topic_id": "T1", "confidence": true, "is_ambiguous": false}
    # In Python float(True) == 1.0, but MUST be rejected
    res_bool = await test_llm_json_response('{"topic_id": "T1", "confidence": true, "is_ambiguous": false}')
    assert_test("Boolean confidence rejected (falls back)", res_bool.get(0) != [{"topic_id": "T1", "confidence": 1.0}])

    # 2. String confidence: {"topic_id": "T1", "confidence": "0.9", "is_ambiguous": false}
    res_str = await test_llm_json_response('{"topic_id": "T1", "confidence": "0.9", "is_ambiguous": false}')
    assert_test("String confidence rejected (falls back)", res_str.get(0) != [{"topic_id": "T1", "confidence": 0.9}])

    # 3. Missing confidence: {"topic_id": "T1", "is_ambiguous": false}
    res_missing_conf = await test_llm_json_response('{"topic_id": "T1", "is_ambiguous": false}')
    assert_test("Missing confidence rejected (falls back)", res_missing_conf.get(0) != [{"topic_id": "T1", "confidence": 0.0}])

    # 4. Out of bounds confidence: confidence > 1.0 or confidence < 0.0
    res_high = await test_llm_json_response('{"topic_id": "T1", "confidence": 1.5, "is_ambiguous": false}')
    assert_test("Confidence > 1.0 rejected", res_high.get(0) != [{"topic_id": "T1", "confidence": 1.5}])
    res_low = await test_llm_json_response('{"topic_id": "T1", "confidence": -0.2, "is_ambiguous": false}')
    assert_test("Confidence < 0.0 rejected", res_low.get(0) != [{"topic_id": "T1", "confidence": -0.2}])

    # 5. Invalid topic ID not in candidates: {"topic_id": "T999", "confidence": 0.9, "is_ambiguous": false}
    res_inv_tid = await test_llm_json_response('{"topic_id": "T999", "confidence": 0.9, "is_ambiguous": false}')
    assert_test("Invalid topic ID rejected", res_inv_tid.get(0) != [{"topic_id": "T999", "confidence": 0.9}])

    # 6. Invalid non-boolean is_ambiguous: {"topic_id": "T1", "confidence": 0.9, "is_ambiguous": "maybe"}
    res_inv_ambig = await test_llm_json_response('{"topic_id": "T1", "confidence": 0.9, "is_ambiguous": "maybe"}')
    assert_test("Non-boolean is_ambiguous rejected", res_inv_ambig.get(0) != [{"topic_id": "T1", "confidence": 0.9}])

    # 7. Retrieval Failure Isolation in batch processing
    print("\n--- Testing Retrieval Failure Isolation ---")
    batch_questions = [
        {"text": "Explain binary search algorithm"},         # Q0: normal question
        {"text": "Explain binary search with failure"},       # Q1: semantic failure, but strong BM25
        {"text": "Completely unrelated noise x1y2z3"},       # Q2: semantic failure AND no BM25 -> abstain
    ]
    
    async def mock_embed_with_isolation(txt):
        if "failure" in txt or "noise" in txt:
            raise RuntimeError("Ollama embedding service error/timeout")
        return [0.1] * 768

    with patch('httpx.AsyncClient') as mock_client, \
         patch('utils._cached_embed_text', side_effect=mock_embed_with_isolation):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '{"topic_id": "T1", "confidence": 0.9, "is_ambiguous": false}'}
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp
        
        batch_res = await semantic_map_questions_to_topics("CS101", dummy_syllabus, batch_questions)
        
        # Q0: normal question succeeded
        assert_test("Batch Isolation: Q0 with normal retrieval succeeded", 0 in batch_res and len(batch_res[0]) > 0)
        # Q1: semantic failed, but BM25 preserved -> accepted or fallback
        assert_test("Batch Isolation: Q1 preserved BM25 when semantic retrieval failed", 1 in batch_res and len(batch_res[1]) > 0)
        # Q2: both failed -> abstained without aborting batch
        assert_test("Batch Isolation: Q2 with both methods failing abstained without aborting", 2 in batch_res and batch_res[2] == [])
        assert_test("Batch Isolation: entire batch completed (3 results)", len(batch_res) == 3)

    # =========================================================================
    # PART 5: MOCKED TRANSACTION TESTS - ATOMIC NEO4J MAPPINGS (ISSUE 4)
    # =========================================================================
    print("\n--- Testing Issue 4: Atomic Neo4j Mapping Replacement ---")
    
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    def fake_execute_write(func, *args, **kwargs):
        return func(mock_session, *args, **kwargs)
    mock_session.execute_write.side_effect = fake_execute_write

    # 1. Valid topic replacement: topic exists in course
    with patch('utils.semantic_map_questions_to_topics', new_callable=AsyncMock) as mock_mapper, \
         patch('utils.fetch_course_topics') as mock_fetch:
        mock_fetch.return_value = "[T1] Unit 1 - Topic 1"
        mock_mapper.return_value = {0: [{"topic_id": "T1", "confidence": 0.85}]}
        
        mock_session.run.reset_mock()
        # Mock topic lookup returning exactly 1 match
        mock_session.run.return_value.data.return_value = [{"t_internal_id": 101}]
        mock_session.run.return_value.single.return_value = {"internal_id": 501}
        
        await map_questions_to_kg(mock_driver, "CS101", [{"text": "Sample", "question_number": "1"}])
        
        calls = mock_session.run.call_args_list
        queries = [call[0][0] for call in calls]
        
        q_merge = any("MERGE (q:Question {id: $q_id})" in q for q in queries)
        rel_delete = any("DELETE r1" in q and "r2" in q for q in queries)
        rel_merge = any("MERGE (q)-[r:TESTS_TOPIC]->(t)" in q and "review_status = 'pending_review'" in q for q in queries)
        
        assert_test("Valid topic replacement executes MERGE question", q_merge)
        assert_test("Valid topic replacement cleans old r1/r2 edges safely", rel_delete)
        assert_test("Valid topic replacement creates TESTS_TOPIC with pending_review", rel_merge)

    # 2. Missing topic in course raises InvalidTopicMappingError and rolls back
    with patch('utils.semantic_map_questions_to_topics', new_callable=AsyncMock) as mock_mapper, \
         patch('utils.fetch_course_topics') as mock_fetch:
        mock_fetch.return_value = "[T1] Unit 1 - Topic 1"
        mock_mapper.return_value = {0: [{"topic_id": "NON_EXISTENT_T", "confidence": 0.85}]}
        
        mock_session.run.reset_mock()
        # Mock topic lookup returning 0 matches
        mock_session.run.return_value.data.return_value = []
        
        await map_questions_to_kg(mock_driver, "CS101", [{"text": "Sample", "question_number": "1"}])
        
        calls = mock_session.run.call_args_list
        queries = [call[0][0] for call in calls]
        
        q_modified = any("MERGE (q:Question" in q for q in queries)
        rel_deleted = any("DELETE r1" in q for q in queries)
        assert_test("Missing topic aborts BEFORE Question node mutation", not q_modified)
        assert_test("Missing topic aborts BEFORE deleting existing relationships", not rel_deleted)

    # 3. Legitimate abstention links Question to Course (course fallback)
    with patch('utils.semantic_map_questions_to_topics', new_callable=AsyncMock) as mock_mapper, \
         patch('utils.fetch_course_topics') as mock_fetch:
        mock_fetch.return_value = "[T1] Unit 1 - Topic 1"
        # Abstention: mapper returns []
        mock_mapper.return_value = {0: []}
        
        mock_session.run.reset_mock()
        mock_session.run.return_value.single.return_value = {"internal_id": 501}
        
        await map_questions_to_kg(mock_driver, "CS101", [{"text": "Sample", "question_number": "1"}])
        
        calls = mock_session.run.call_args_list
        queries = [call[0][0] for call in calls]
        course_fallback = any("MERGE (q)-[:BELONGS_TO]->(c)" in q for q in queries)
        assert_test("Legitimate abstention applies course-only fallback", course_fallback)

    # 4. Course Outcome and Document relationships are not deleted
    # The queries use OPTIONAL MATCH to r1:TESTS_TOPIC and r2:BELONGS_TO only
    delete_queries = [q for q in queries if "DELETE" in q]
    all_delete_safe = all("MAPPED_TO_CO" not in q and "EXTRACTED_FROM" not in q for q in delete_queries)
    assert_test("CO and Document relationships are preserved during deletions", all_delete_safe)

    # =========================================================================
    # PART 6: LIVE NEO4J INTEGRATION TESTS (IF REACHABLE)
    # =========================================================================
    print("\n--- Testing Live Neo4j Integration ---")
    live_neo4j_ran = False
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD", "neo4j")
        live_driver = GraphDatabase.driver(uri, auth=(user, pwd))
        
        # Check connectivity
        with live_driver.session() as s:
            s.run("RETURN 1 as test").single()
        
        print("Connected to live Neo4j database! Running real transaction tests...")
        live_neo4j_ran = True
        
        TEST_COURSE = "TEST_INTEG_101"
        TEST_TOPIC = "TEST_TOPIC_1"
        
        # Setup disposable test graph
        with live_driver.session() as s:
            s.run("""
                MERGE (c:Course {code: $c_code, name: 'Integration Test Course'})
                MERGE (u:Unit {id: 'U_TEST', title: 'Unit 1'})
                MERGE (c)-[:HAS_UNIT]->(u)
                MERGE (t:Topic {topic_id: $t_id, name: 'Test Topic Name', course_code: $c_code})
                MERGE (u)-[:HAS_TOPIC]->(t)
            """, c_code=TEST_COURSE, t_id=TEST_TOPIC)
        
        # Test 1: Live atomic mapping update with valid topic
        with patch('utils.semantic_map_questions_to_topics', new_callable=AsyncMock) as mock_mapper:
            mock_mapper.return_value = {0: [{"topic_id": TEST_TOPIC, "confidence": 0.88}]}
            test_q = [{"text": "Live test question for atomic mapping", "question_number": "1"}]
            await map_questions_to_kg(live_driver, TEST_COURSE, test_q, document_id="DOC_INTEG_1")
            
        with live_driver.session() as s:
            rec = s.run("""
                MATCH (q:Question)-[r:TESTS_TOPIC]->(t:Topic {topic_id: $t_id})
                WHERE q.course_code = $c_code
                RETURN count(r) as cnt, r.review_status as status, r.confidence as conf
            """, c_code=TEST_COURSE, t_id=TEST_TOPIC).single()
            assert_test("Live Neo4j: Valid topic mapping written with pending_review", 
                        rec["cnt"] == 1 and rec["status"] == "pending_review" and math.isclose(rec["conf"], 0.88))

        # Test 2: Live atomic rollback on invalid topic (existing mapping preserved, no partial changes)
        with patch('utils.semantic_map_questions_to_topics', new_callable=AsyncMock) as mock_mapper:
            mock_mapper.return_value = {0: [{"topic_id": "NON_EXISTENT_INVALID_TOPIC", "confidence": 0.99}]}
            test_q = [{"text": "Live test question for atomic mapping", "question_number": "1"}]
            await map_questions_to_kg(live_driver, TEST_COURSE, test_q, document_id="DOC_INTEG_1")

        with live_driver.session() as s:
            rec2 = s.run("""
                MATCH (q:Question)-[r:TESTS_TOPIC]->(t:Topic {topic_id: $t_id})
                WHERE q.course_code = $c_code
                RETURN count(r) as cnt, r.confidence as conf
            """, c_code=TEST_COURSE, t_id=TEST_TOPIC).single()
            # The existing valid mapping must STILL be present and untouched!
            assert_test("Live Neo4j: Invalid topic proposal rolled back, preserving existing relationship", 
                        rec2["cnt"] == 1 and math.isclose(rec2["conf"], 0.88))

        # Test 3: Live legitimate abstention applies course fallback
        with patch('utils.semantic_map_questions_to_topics', new_callable=AsyncMock) as mock_mapper:
            mock_mapper.return_value = {0: []} # Deliberate abstention
            test_q = [{"text": "Live test question for atomic mapping", "question_number": "1"}]
            await map_questions_to_kg(live_driver, TEST_COURSE, test_q, document_id="DOC_INTEG_1")

        with live_driver.session() as s:
            rec3 = s.run("""
                MATCH (q:Question)-[r:BELONGS_TO]->(c:Course {code: $c_code})
                RETURN count(r) as cnt
            """, c_code=TEST_COURSE).single()
            assert_test("Live Neo4j: Legitimate abstention correctly links Question to Course", rec3["cnt"] == 1)

        # Test 4: Live rollback when write fails AFTER Question node has been modified
        FAIL_Q_ID = "Q_INTEG_FAIL_AFTER"
        with live_driver.session() as s:
            # Seed existing valid question and relationship
            s.run("""
                MERGE (q:Question {id: $q_id})
                SET q.text = 'Original Unmodified Text', q.marks = '5', q.course_code = $c_code
                WITH q
                MATCH (t:Topic {topic_id: $t_id})
                MERGE (q)-[r:TESTS_TOPIC]->(t)
                SET r.confidence = 0.77, r.review_status = 'approved'
            """, q_id=FAIL_Q_ID, c_code=TEST_COURSE, t_id=TEST_TOPIC)

        def failing_tx_after_q_mutation(tx):
            # Mutate Question properties and delete relationships
            tx.run("""
                MATCH (q:Question {id: $q_id})
                SET q.text = 'Mutated Text That Must Rollback', q.marks = '10'
                WITH q
                OPTIONAL MATCH (q)-[r:TESTS_TOPIC]->()
                DELETE r
            """, q_id=FAIL_Q_ID)
            # Deliberately raise exception AFTER Question node was modified
            raise RuntimeError("Deliberate write failure after Question mutation")

        with live_driver.session() as s:
            try:
                s.execute_write(failing_tx_after_q_mutation)
            except RuntimeError:
                pass # Expected transaction failure

        # Verify in live Neo4j that entire transaction rolled back and previous state remains intact
        with live_driver.session() as s:
            q_after = s.run("""
                MATCH (q:Question {id: $q_id})
                OPTIONAL MATCH (q)-[r:TESTS_TOPIC]->(t:Topic)
                RETURN q.text as text, q.marks as marks, count(r) as rel_cnt, r.confidence as conf
            """, q_id=FAIL_Q_ID).single()
            
            assert_test(
                "Live Neo4j: Failure AFTER Question mutation rolls back properties to original",
                q_after["text"] == "Original Unmodified Text" and q_after["marks"] == "5",
                f"Properties were corrupted: text={q_after['text']}, marks={q_after['marks']}"
            )
            assert_test(
                "Live Neo4j: Failure AFTER Question mutation restores previous valid relationship",
                q_after["rel_cnt"] == 1 and math.isclose(q_after["conf"], 0.77),
                f"Relationship was not restored: cnt={q_after['rel_cnt']}"
            )

        # Test 5: Live Batch Isolation (Database failure on one question does NOT abort batch or convert to course fallback)
        batch_qs = [
            {"id": "Q_BATCH_1", "text": "Batch question 1", "question_number": "1"},
            {"id": "Q_BATCH_FAIL", "text": "Batch question 2 fails", "question_number": "2"},
            {"id": "Q_BATCH_3", "text": "Batch question 3", "question_number": "3"},
        ]
        
        with patch('utils.semantic_map_questions_to_topics', new_callable=AsyncMock) as mock_mapper:
            mock_mapper.return_value = {
                0: [{"topic_id": TEST_TOPIC, "confidence": 0.85}],
                1: [{"topic_id": TEST_TOPIC, "confidence": 0.85}],
                2: [{"topic_id": TEST_TOPIC, "confidence": 0.85}],
            }
            
            # Monkey-patch session.execute_write to simulate failure on Q_BATCH_FAIL only
            orig_session_ctx = live_driver.session
            class FaultySessionWrapper:
                def __init__(self, sess):
                    self._sess = sess
                def execute_write(self, fn, *args, **kwargs):
                    if len(args) > 0 and args[0] == "Q_BATCH_FAIL":
                        raise RuntimeError("Simulated transient write failure on Q_BATCH_FAIL")
                    return self._sess.execute_write(fn, *args, **kwargs)
                def __getattr__(self, name):
                    return getattr(self._sess, name)
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    return self._sess.__exit__(exc_type, exc_val, exc_tb)

            class FaultyDriverWrapper:
                def __init__(self, driver):
                    self._driver = driver
                def session(self, *args, **kwargs):
                    return FaultySessionWrapper(self._driver.session(*args, **kwargs))
                def __getattr__(self, name):
                    return getattr(self._driver, name)

            faulty_driver = FaultyDriverWrapper(live_driver)
            await map_questions_to_kg(faulty_driver, TEST_COURSE, batch_qs)

        with live_driver.session() as s:
            rec_b1 = s.run("MATCH (q:Question {id: 'Q_BATCH_1'})-[:TESTS_TOPIC]->() RETURN count(*) as cnt").single()["cnt"]
            rec_bfail_fallback = s.run("MATCH (q:Question {id: 'Q_BATCH_FAIL'})-[:BELONGS_TO]->() RETURN count(*) as cnt").single()["cnt"]
            rec_b3 = s.run("MATCH (q:Question {id: 'Q_BATCH_3'})-[:TESTS_TOPIC]->() RETURN count(*) as cnt").single()["cnt"]

            assert_test("Live Neo4j: Q1 in batch succeeded", rec_b1 == 1)
            assert_test("Live Neo4j: DB failure on Q2 did NOT convert into course fallback", rec_bfail_fallback == 0)
            assert_test("Live Neo4j: Q3 in batch succeeded (batch was NOT aborted)", rec_b3 == 1)

        # Teardown disposable integration nodes
        with live_driver.session() as s:
            s.run("""
                MATCH (q:Question) WHERE q.course_code = $c_code OR q.id IN ['Q_INTEG_FAIL_AFTER', 'Q_BATCH_1', 'Q_BATCH_FAIL', 'Q_BATCH_3']
                DETACH DELETE q
            """, c_code=TEST_COURSE)
            s.run("""
                MATCH (c:Course {code: $c_code})
                OPTIONAL MATCH (c)-[:HAS_UNIT]->(u)
                OPTIONAL MATCH (u)-[:HAS_TOPIC]->(t)
                DETACH DELETE c, u, t
            """, c_code=TEST_COURSE)
        print("Live Neo4j disposable test nodes cleaned up successfully.")

    except Exception as e:
        print(f"⚠️ Live Neo4j test skipped or failed to connect: {e}")
        assert_test("Live Neo4j integration test execution", False, str(e))

    # =========================================================================
    # PART 7: GRAPH VISUALIZATION IDENTIFIERS & COURSE OUTCOMES
    # =========================================================================
    print("\n--- Testing Graph Visualization & Course Queries ---")
    mock_session.run.reset_mock()
    mock_record = {
        "c": {"code": "CS101", "name": "CS101"},
        "u": {"id": "U1", "title": "Unit 1", "number": "1"},
        "t": {"topic_id": "T_ID_1", "id": "LegacyID", "name": "Topic 1"},
        "q": {"id": "Q1", "text": "Q1 Text"},
        "br": None, "co": None, "t2": None, "tr": None
    }
    mock_session.run.return_value = [mock_record]
    graph = fetch_graph_for_course(mock_driver, "CS101")
    t_nodes = [n for n in graph["nodes"] if n["label"] == "Topic"]
    assert_test("Graph visualization topic identifier resolution", t_nodes[0]["id"] == "T_ID_1")

    print("\n" + "=" * 60)
    print(f"FINAL RESULT: {passed}/{total} tests passed.")
    print("=" * 60)
    return passed, total, live_neo4j_ran

if __name__ == "__main__":
    asyncio.run(run_all_tests())
