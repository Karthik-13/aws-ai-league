## Code Index

This document provides a high-level index of the codebase for quick navigation.

### Top-level structure
- `agents/`: Agents for generating questions, deduplicating, and producing responses; LangChain and non-LangChain variants; dataset pipeline.
- `data_prcoessing/`: Data processing scripts and docs (note: directory name contains a typo).
- `files/`: Duplicate/alternate copies of data processing scripts and docs.
- `datasets/`: JSONL and parquet datasets.
- `scrap/`: Scraper utility and examples.
- Root: helper scripts and documents.

### agents/
- `pipeline.py`
  - Classes: `DatasetPipeline`
  - Entry points: `main()`
- `langchain_pipeline.py`
  - Classes: `LangChainDatasetPipeline`
  - Entry points: `main()`
- `question_generator.py`
  - Classes: `Question`, `QuestionGenConfig`, `QuestionGenerationAgent`
  - Entry points: `main()`
- `question_deduplicator.py`
  - Classes: `DeduplicationConfig`, `QuestionDeduplicationAgent`
  - Entry points: `main()`
- `response_generator.py`
  - Classes: `ResponseGenConfig`, `ResponseGenerationAgent`
  - Entry points: `main()`
- `langchain_question_generator.py`
  - Classes: `Question (pydantic)`, `QuestionBatch (pydantic)`, `QuestionGenConfig`, `QuestionGenerationAgent`
  - Entry points: `main()`
- `langchain_question_deduplicator.py`
  - Classes: `DeduplicationConfig`, `QuestionDeduplicationAgent`
  - Entry points: `main()`
- `langchain_response_generator.py`
  - Classes: `TrainingExample (pydantic)`, `TrainingBatch (pydantic)`, `ResponseGenConfig`, `ResponseGenerationAgent`
  - Entry points: `main()`
- Docs
  - `ARCHITECTURE.md`, `README_PIPELINE.md`, `README_LANGCHAIN.md`, `QUICKSTART.md`, `QUICK_REFERENCE.txt`, `COMPARISON.md`, `INDEX_COMPLETE.md`
  - Requirements: `requirements_langchain.txt`

### data_prcoessing/
- Scripts
  - `dataset_to_jsonl.py`: `load_data`, `convert_to_jsonl`, `preview_data`, `auto_detect_columns`, `main`
  - `qa_generator.py`: `load_text_file`, `chunk_text`, `generate_qa_pairs`, `generate_from_file`, `main`
  - `demo_qa_output.py`: example generation runner
  - Requirements: `requirements.txt`
- Docs
  - `MASTER_README.md`, `QA_GENERATOR_README.md`, `README.md`, `VISUAL_GUIDE.txt`

### files/ (alternate copies)
- Scripts
  - `dataset_to_jsonl.py`: `load_data`, `convert_to_jsonl`, `preview_data`, `auto_detect_columns`, `main`
  - `qa_generator.py`: `load_text_file`, `chunk_text`, `generate_qa_pairs`, `generate_from_file`, `main`
  - `demo_qa_output.py`
- Docs
  - `MASTER_README.md`, `QA_GENERATOR_README.md`, `README.md`, `VISUAL_GUIDE.txt`

### scrap/
- `usagov_scraper.py`
  - Classes: `USAGovScraper`
  - Entry points: `main()`
- `test_scraper.py`
  - Functions: `test_scraper()`
- Docs: `README.md`, `QUICKSTART.md`, `CHANGELOG_V2.md`

### Root scripts
- `keep-jsonl-keys.py`: `keep_only_keys`
- `rename_keys.py`: `rename_keys`
- `parquet_to_jsonl.py`: `parquet_to_jsonl`

### Notes
- Multiple entry-point scripts expose `main()` for CLI usage across agents and data utilities.
- Consider consolidating duplicated scripts between `data_prcoessing/` and `files/` or fixing the directory typo.


