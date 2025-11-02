# System Architecture

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Fine-tuning Dataset Pipeline              │
│                      (LangChain Implementation)                 │
└─────────────────────────────────────────────────────────────────┘

                              INPUT
                                ↓
                          [Your Topic]
                                ↓
        ┌───────────────────────────────────────────────────┐
        │     AGENT 1: Question Generator (LangChain)      │
        │  ┌─────────────────────────────────────────────┐  │
        │  │  • Generates diverse questions                  │  │
        │  │  • Tracks previously generated questions        │  │
        │  │  • Real-time duplicate avoidance               │  │
        │  │  • Category-based generation                    │  │
        │  │  • Custom system prompts                       │  │
        │  │  • Resumable                                   │  │
        │  │  • Type-safe with Pydantic                     │  │
        │  └─────────────────────────────────────────────┘  │
        │  Uses: ChatBedrock, ChatPromptTemplate,           │
        │        JsonOutputParser with Pydantic             │
        │             Claude via AWS Bedrock                 │
        └───────────────────────────────────────────────────┘
                                ↓
                    questions_topic.jsonl
                    (e.g., 1000 questions)
                                ↓
        ┌───────────────────────────────────────────────────┐
        │     AGENT 2: Question Deduplicator (LangChain)  │
        │  ┌─────────────────────────────────────────────┐  │
        │  │  • Semantic similarity detection             │  │
        │  │  • Embedding-based comparison                │  │
        │  │  • Two methods:                             │  │
        │  │    - Threshold (cosine similarity)           │  │
        │  │    - Clustering (DBSCAN)                     │  │
        │  │  • Configurable thresholds                  │  │
        │  │  • Detailed statistics                       │  │
        │  │  • Automatic batching                        │  │
        │  └─────────────────────────────────────────────┘  │
        │  Uses: BedrockEmbeddings                         │
        │     Amazon Titan Embeddings V2                    │
        └───────────────────────────────────────────────────┘
                                ↓
                questions_deduplicated.jsonl
                    (e.g., 850 unique)
                                ↓
        ┌───────────────────────────────────────────────────┐
        │     AGENT 3: Response Generator (LangChain)      │
        │  ┌─────────────────────────────────────────────┐  │
        │  │  • Generates context & responses             │  │
        │  │  • Batch/chunk processing                    │  │
        │  │  • Custom system prompts                     │  │
        │  │  • Error recovery                            │  │
        │  │  • Progress tracking                         │  │
        │  │  • Resumable                                 │  │
        │  │  • Parallel processing support               │  │
        │  │  • Incremental writing                       │  │
        │  │  • Type-safe with Pydantic                   │  │
        │  └─────────────────────────────────────────────┘  │
        │  Uses: ChatBedrock, ChatPromptTemplate,           │
        │        JsonOutputParser with Pydantic             │
        │             Claude via AWS Bedrock                 │
        └───────────────────────────────────────────────────┘
                                ↓
                        OUTPUT
                training_data.jsonl
    (Ready for LLM instruction fine-tuning!)
                                ↓
        ┌───────────────────────────────────────────────────┐
        │     AGENT 4: Response Improver (LangChain)        │
        │              [OPTIONAL]                            │
        │  ┌─────────────────────────────────────────────┐  │
        │  │  • Evaluates response quality                │  │
        │  │  • Fixes issues (grammar, clarity, etc.)     │  │
        │  │  • Enhances good responses                    │  │
        │  │  • Uses different/larger model               │  │
        │  │  • Parallel processing support               │  │
        │  │  • Incremental writing                       │  │
        │  │  • Type-safe with Pydantic                   │  │
        │  └─────────────────────────────────────────────┘  │
        │  Uses: ChatBedrock, ChatPromptTemplate,           │
        │        JsonOutputParser with Pydantic             │
        │         Claude/Llama via AWS Bedrock              │
        └───────────────────────────────────────────────────┘
                                ↓
                    training_data_improved.jsonl
                    (Enhanced training data)
```

## Data Flow

### Input → Agent 1 → Questions
```
Topic: "customer service"
     ↓
[Question Generator (LangChain)]
  • ChatBedrock LLM
  • ChatPromptTemplate
  • JsonOutputParser
     ↓
{
  "instruction": "How do I return a product?",
  "context": "",
  "response": ""
}
```

### Questions → Agent 2 → Unique Questions
```
1000 questions
     ↓
[Deduplicator (LangChain)]
  • BedrockEmbeddings
  • Embedding generation
  • Similarity check
  • Remove duplicates
     ↓
850 unique questions (15% removed)
```

### Unique Questions → Agent 3 → Training Data
```
{
  "instruction": "How do I return a product?",
  "context": "",
  "response": ""
}
     ↓
[Response Generator (LangChain)]
  • ChatBedrock LLM
  • ChatPromptTemplate
  • JsonOutputParser
  • Add context
  • Generate response
     ↓
{
  "instruction": "How do I return a product?",
  "context": "Returns are accepted within 30 days...",
  "response": "To return a product: 1) Log into your account..."
}
```

### Training Data → Agent 4 → Improved Data (Optional)
```
{
  "instruction": "...",
  "context": "...",
  "response": "..."
}
     ↓
[Response Improver (LangChain)]
  • Evaluation model (larger/different)
  • Quality assessment
  • Fix issues
  • Enhance responses
     ↓
{
  "instruction": "...",
  "context": "...",
  "response": "[Improved response]"
}
```

## File Structure

```
project/
├── langchain_question_generator.py     # Agent 1
├── langchain_question_deduplicator.py  # Agent 2
├── langchain_response_generator.py     # Agent 3
├── langchain_response_improver.py      # Agent 4 (optional)
├── langchain_pipeline.py               # Orchestrator
├── config.yaml                         # Unified configuration
├── requirements_langchain.txt          # Dependencies
├── README.md                          # Main documentation
├── QUICKSTART.md                      # Quick start guide
├── ARCHITECTURE.md                    # This file
│
└── output/                            # Generated files
    ├── questions_topic_timestamp.jsonl
    ├── questions_topic_timestamp_stats.json
    ├── questions_deduplicated_topic_timestamp.jsonl
    ├── questions_deduplicated_..._dedup_stats.json
    ├── training_data_topic_timestamp.jsonl  ✨
    ├── training_data_..._response_stats.json
    ├── training_data_..._improved.jsonl      (optional)
    └── training_data_..._improved_stats.json (optional)
```

## Agent Details

### Agent 1: Question Generator (`langchain_question_generator.py`)
**Input:** Topic string (or from config.yaml)
**Output:** JSONL file with questions
**AWS Services:** Bedrock (Claude)
**LangChain Components:**
- `ChatBedrock` - LLM integration
- `ChatPromptTemplate` - Prompt management
- `JsonOutputParser` - Structured output parsing
- Pydantic models for validation

**Key Features:**
- Batch generation (configurable batch size)
- Real-time duplicate checking
- Category guidance
- Temperature control for diversity
- Resume capability
- Statistics tracking
- Type-safe with Pydantic

**Configuration:**
```python
{
    "topic": "customer service",
    "num_questions": 1000,
    "batch_size": 20,
    "temperature": 0.9,
    "system_prompt": "Optional custom prompt",
    "categories": ["orders", "returns", "shipping"]
}
```

### Agent 2: Question Deduplicator (`langchain_question_deduplicator.py`)
**Input:** JSONL file with questions
**Output:** JSONL file with unique questions
**AWS Services:** Bedrock (Titan Embeddings)
**LangChain Components:**
- `BedrockEmbeddings` - Native embedding support
- Automatic batching

**Key Features:**
- Semantic similarity detection
- Two deduplication methods
- Configurable threshold
- Detailed statistics
- Sample duplicate groups
- Automatic batch embedding

**Methods:**
1. **Threshold:** Cosine similarity >= threshold → duplicate
2. **Clustering:** DBSCAN to group similar questions

**Configuration:**
```python
{
    "similarity_threshold": 0.85,
    "method": "threshold",  # or "clustering"
    "eps": 0.15  # for clustering
}
```

### Agent 3: Response Generator (`langchain_response_generator.py`)
**Input:** JSONL file with questions
**Output:** Complete training data JSONL
**AWS Services:** Bedrock (Claude)
**LangChain Components:**
- `ChatBedrock` - LLM integration
- `ChatPromptTemplate` - Multi-message templates
- `JsonOutputParser` - Structured output parsing
- Pydantic models for validation

**Key Features:**
- Chunk processing (multiple questions per API call)
- Context generation (optional)
- Error recovery
- Resume capability
- Progress tracking
- Parallel processing support
- Incremental writing (saves after each chunk)
- Configurable timeout

**Configuration:**
```python
{
    "chunk_size": 5,
    "temperature": 0.7,
    "max_tokens": 2048,
    "timeout": 300,
    "max_workers": 1,  # 1 = sequential, 2+ = parallel
    "add_context": True,
    "system_prompt": "Optional custom prompt"
}
```

### Agent 4: Response Improver (`langchain_response_improver.py`) (Optional)
**Input:** JSONL file with training data
**Output:** Improved training data JSONL
**AWS Services:** Bedrock (Claude/Llama - different/larger model)
**LangChain Components:**
- `ChatBedrock` - LLM integration
- `ChatPromptTemplate` - Evaluation templates
- `JsonOutputParser` - Structured output parsing
- Pydantic models for validation

**Key Features:**
- Evaluates response quality
- Fixes issues (grammar, clarity, missing sections)
- Enhances good responses
- Uses different/larger model for evaluation
- Parallel processing support
- Incremental writing
- Custom evaluation criteria

**Configuration:**
```python
{
    "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Evaluation model
    "chunk_size": 5,
    "temperature": 0.3,
    "max_tokens": 4096,
    "timeout": 600,
    "max_workers": 1,
    "fix_only": False,  # If True, only fix issues
    "evaluation_criteria": "Optional custom criteria"
}
```

## Output Format

Final training data format (per line):
```json
{
  "instruction": "User question here",
  "context": "Optional background information",
  "response": "Detailed, helpful response"
}
```

This matches the standard instruction fine-tuning format used by most LLM training frameworks.

## API Calls Overview

For 1000 questions with default settings:

**Question Generation:**
- API calls: ~50 (20 questions per batch)
- Model: Claude 3.5 Sonnet
- Cost: ~$2-3

**Deduplication:**
- API calls: ~40 (25 questions per batch)
- Model: Titan Embeddings V2
- Cost: ~$0.20

**Response Generation:**
- API calls: ~200 (5 questions per chunk after dedup)
- Model: Claude 3.5 Sonnet
- Cost: ~$2-3

**Response Improvement (Optional):**
- API calls: ~170 (5 examples per chunk)
- Model: Claude 3.5 Sonnet (or larger)
- Cost: ~$2-3

**Total: ~$5 for 1000 question dataset (without improvement)**
**Total: ~$7 for 1000 question dataset (with improvement)**

## Error Handling

Each agent includes:
- ✅ Try-catch blocks for API errors
- ✅ Batch-level error isolation
- ✅ Progress tracking
- ✅ Resume capability
- ✅ Detailed error logging
- ✅ Statistics on success/failure
- ✅ Retry logic with exponential backoff
- ✅ Timeout handling

## Performance Optimization

1. **Batching:** Process multiple items per API call
2. **Caching:** Track generated questions to avoid duplicates
3. **Parallel Processing:** Process chunks concurrently with ThreadPoolExecutor
4. **Rate Limiting:** Built-in delays to avoid throttling
5. **Resume:** Don't restart from scratch on failures
6. **Incremental Writing:** Save results after each chunk
7. **Connection Pooling:** Reuse Bedrock clients efficiently

## Extensibility

Easy to extend with LangChain:
- Add new deduplication algorithms
- Support additional embedding models
- Add quality scoring
- Implement active learning
- Add multi-language support
- Custom post-processing
- RAG integration
- Vector stores for semantic search
- Agents for multi-step reasoning
- Tools for external API calls
- Callbacks for logging and monitoring
- Streaming for real-time output

## Best Practices

1. **Start Small:** Test with 50-100 questions
2. **Review Samples:** Check quality before scaling
3. **Use Config:** Centralize settings in config.yaml
4. **Tune Parameters:** Adjust based on your needs
5. **Monitor Stats:** Use statistics files to track quality
6. **Iterate:** Generate → Review → Adjust → Regenerate
7. **Use Response Improver:** For critical datasets, use a stronger model
8. **Parallel Processing:** Use max_workers for faster generation (2-4 recommended)
9. **Chunk Size:** Use 1 for detailed responses, 5 for balance
10. **Timeout:** Increase timeout (300-600s) for longer responses
