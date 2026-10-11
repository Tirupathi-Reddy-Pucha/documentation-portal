import asyncio
import json
import time
import os
from typing import List, Dict

from utils import semantic_map_questions_to_topics
from embedder import embed_text

async def run_evaluation():
    print("==================================================")
    print("   INDUSTRY IR: HYBRID MAPPING EVALUATION SUITE   ")
    print("==================================================")
    
    # We define a syllabus context designed to test edge cases
    syllabus_context = """[T1] Unit 1 - Object Oriented Programming
[T2] Unit 1 - Object Oriented Design Patterns
[T3] Unit 2 - Data Structures and Algorithms
[T4] Unit 2 - Time and Space Complexity
[T5] Unit 3 - Network Systems
[T6] Unit 4 - Database Management
[T7] Unit 5 - Operating Systems"""
    
    course_code = "CS101"
    
    # 20 Targeted test cases
    mock_questions = [
        # 1. Exact technical phrase match
        {"id": 1, "text": "What is Object Oriented Programming?", "expected": "T1"},
        # 2. Different wording from syllabus topic
        {"id": 2, "text": "Explain how OS manages memory and processes.", "expected": "T7"},
        # 3. Two topics with heavily overlapping names
        {"id": 3, "text": "Discuss Object Oriented Design Patterns.", "expected": "T2"},
        # 4. Technical common words
        {"id": 4, "text": "What is the time complexity of the quicksort algorithm?", "expected": "T4"},
        # 5. Lexical poor, Semantic high
        {"id": 5, "text": "Explain relational table normal forms.", "expected": "T6"}, # Maps to Database Management
        # 6. Semantic poor, Lexical high (trick question mapping to specific keywords)
        {"id": 6, "text": "Compare network systems topologies.", "expected": "T5"},
        # 7. Ambiguous top-ranked (Selective LLM reranking triggers)
        {"id": 7, "text": "Explain object structures.", "expected": "T1"}, # Ambiguous between T1 and T2
        # 8. Insufficient evidence leading to abstention
        {"id": 8, "text": "Discuss the impact of climate change on agriculture.", "expected": None}, # Course Fallback
        # 9. Multiple relevant (LLM should pick best or abstain)
        {"id": 9, "text": "How do database management systems handle network system failures?", "expected": "T6"},
        # 10. Another course fallback
        {"id": 10, "text": "What is the meaning of life?", "expected": None}
    ]
    
    start_time = time.time()
    
    print("\n[Running Pipeline...]")
    results = await semantic_map_questions_to_topics(course_code, syllabus_context, mock_questions)
    
    print("\n--- TEST RESULTS ---")
    correct = 0
    total = len(mock_questions)
    
    for q in mock_questions:
        q_id = q["id"]
        expected = q["expected"]
        
        # Results map contains lists of mapped topic dicts or empty list for abstention
        mapped_tids = results.get(q_id - 1, [])
        actual = mapped_tids[0]["topic_id"] if mapped_tids else None
        
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        if actual == expected:
            correct += 1
            
        print(f"Q{q_id}: '{q['text']}'")
        print(f"  Expected: {expected} | Actual: {actual} -> {status}\n")

    accuracy = (correct / total) * 100
    print("==================================================")
    print(f"EVALUATION ACCURACY: {accuracy:.2f}% ({correct}/{total})")
    print(f"LATENCY: {time.time() - start_time:.2f}s")
    print("==================================================")
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(run_evaluation())
