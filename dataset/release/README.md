# Campus Knowledge Engine - Instruction Fine-Tuning Dataset

## Overview
This repository contains the official instruction fine-tuning dataset for the **Campus Knowledge Engine (CurriculumLens)**. It is designed to train large language models (LLMs) to seamlessly act as the reasoning engine for a Graph RAG system operating on university curricula, syllabi, and past year question (PYQ) papers.

The dataset contains **9,000 synthetic examples** structured to embed strict JSON schemas, curriculum rules (e.g., Bloom's Taxonomy, ascending unit order), and knowledge graph traversal patterns directly into the model weights, thereby dramatically reducing the required inference prompt context size (the "Shrunk Prompts" architecture).

## Dataset Tasks
The dataset encompasses 6 specialized sub-tasks required by the Campus Knowledge Engine pipeline:
1. **Intent Classification** (1,500 rows): Classifies user queries into 6 distinct intents (e.g., `PROBLEM_LIST`, `MULTI_HOP_PREREQ`) and extracts the target academic topic. Output is strict JSON.
2. **Topic Mapping** (1,500 rows): Re-ranks candidate curriculum topics retrieved from Neo4j based on relevance to a question. Output is strict JSON with confidence scores.
3. **RAG Answer Generation** (1,500 rows): Formats Neo4j context (PYQs or Exact Syllabus Matches) into readable Markdown. Enforces strict ascending numerical order for units and incorporates dynamic markdown image links. Includes edge cases for prerequisites and Course Outcome (CO)/Bloom's Taxonomy Level (BTL) mapping queries.
4. **Query Rewrite** (1,500 rows): Rewrites follow-up conversational questions into standalone semantic queries by resolving pronouns from the chat history.
5. **Knowledge Graph Extraction** (1,500 rows): Extracts structural subject-predicate-object triplets from raw text for ingestion into Neo4j. Output is strict JSON.
6. **Syllabus Enrichment** (1,500 rows): Cleans raw syllabus text, maps topics to Bloom's Taxonomy Levels, and infers prerequisites. Output is strict JSON.

## File Structure
- `campus_knowledge_dataset_v1.jsonl`: The primary dataset file in Alpaca JSONL format.
- `annotation_guidelines.md`: Details the procedural rules and deterministic constraints used to synthetically generate and validate the data.
- `metadata.json`: Dataset metadata and statistics.
- `LICENSE`: Creative Commons CC-BY-4.0 License file.

## Data Format
The dataset uses the standard instruction-tuning format (Alpaca style) with three primary fields per record:
- `task`: The specific sub-task (e.g., `rag_answer_generation`).
- `instruction`: The Shrunk Prompt representing the query/input from the Python backend.
- `input`: Left empty for all records, as the context is injected directly into the `instruction`.
- `output`: The exact expected output from the model (e.g., JSON schema, formatted Markdown).

Example Record (Topic Mapping):
```json
{
  "task": "topic_mapping",
  "instruction": "Map to curriculum topic.\nCOURSE:\n23CSE301\nCANDIDATE CURRICULUM TOPICS:\n- Overfitting and Underfitting\n- Neural Networks\n- Linear Regression\nQUESTION:\nExplain Neural Networks.",
  "input": "",
  "output": "{\"topics\": [{\"name\": \"Neural Networks\", \"confidence\": 95}]}"
}
```

## Usage
This dataset is optimized for LoRA/QLoRA fine-tuning on parameter-efficient hardware (e.g., single GPU) using frameworks such as Unsloth, Axolotl, or Hugging Face `SFTTrainer`.

For Gemma-based models (e.g., `gemma4:12b-it-qat`), it is recommended to apply the standard chat template over the `instruction` and `output` keys during data collating.

## Version Details
**Version:** 1.0.0
**Release Date:** October 2026
**Changes:** Initial dataset release containing 9,000 synthetically aligned examples.
