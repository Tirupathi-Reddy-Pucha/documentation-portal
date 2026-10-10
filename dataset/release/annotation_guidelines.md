# Dataset Annotation Guidelines

The data in `campus_knowledge_dataset_v1.jsonl` was synthetically generated to enforce perfect architectural parity with the Campus Knowledge Engine backend. These guidelines document the exact deterministic rules ("annotations") that guided the generation of each task.

## 1. Intent Classification
- **Constraint:** The output must be a single stringified JSON object: `{"intent": "INTENT_NAME", "topic": "TOPIC_NAME"}`.
- **Allowed Intents:** `PROBLEM_LIST`, `MULTI_HOP_PREREQ`, `GRAPH_PYQ_MAPPING`, `SIMPLE_CURRICULUM`, `SIMPLE_PYQ`, `GENERAL`.
- **Annotation Logic:** Utterances are mapped based on predefined keyword heuristics representing genuine student queries (e.g., "what are the prerequisites" -> `MULTI_HOP_PREREQ`). The topic field is left blank `""` for `GENERAL` intent.

## 2. Topic Mapping
- **Constraint:** The output must be a single stringified JSON object: `{"topics": [{"name": "Exact Topic Name", "confidence": X}]}`.
- **Annotation Logic:** The target topic must exactly match an entity from the candidate list provided in the instruction context. Hallucinated topic names are explicitly forbidden.

## 3. RAG Answer Generation
- **Constraint:** The output must be perfectly formatted Markdown.
- **Annotation Logic (Syllabus):** 
  - Topics must be grouped into `**Unit X:**` bolded blocks.
  - The units must be ordered in STRICT ASCENDING numerical order (e.g., Unit 1, Unit 2, Unit 3), regardless of the order they appear in the provided context.
- **Annotation Logic (PYQs):** 
  - Past year questions must follow the Markdown format: `**Q[Num]** *(CO[X], BTL[Y], [Z] Marks)*\n[Question Text]`
- **Annotation Logic (Prerequisites/Mapping):** 
  - Models must translate `--- PREREQUISITE GRAPH KNOWLEDGE ---` tags from the graph search into a natural language sentence addressing the student's query.

## 4. Query Rewrite
- **Constraint:** The output must be raw text representing the standalone semantic query, free from conversational filler.
- **Annotation Logic:** The rewrite must accurately resolve pronouns (e.g., "that topic") to the exact topic name mentioned in the immediately preceding conversation history.

## 5. Knowledge Graph Extraction
- **Constraint:** The output must be a stringified JSON object: `{"triplets": [{"subject": "A", "predicate": "B", "object": "C"}]}`.
- **Annotation Logic:** Extracted subject and object values must be verbatim concepts from the provided text. No summarization or abstraction is permitted.

## 6. Syllabus Enrichment
- **Constraint:** The output must be a stringified JSON object matching the `curriculum_extractor.py` Pydantic schema, including `topics` (with `bloom_level`, `prerequisite_of`, `part_of`) and `extraction_notes`.
- **Annotation Logic:** The "noise" injected into the `Syllabus text` (such as "Textbook pg. 30") must be deterministically stripped from the resulting JSON array. Bloom's Taxonomy Levels must be valid (L1-L6).
