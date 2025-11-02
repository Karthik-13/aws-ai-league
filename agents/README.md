# LLM Fine-tuning Dataset Generation Pipeline

A complete three-agent system for generating high-quality instruction fine-tuning datasets using AWS Bedrock and LangChain. Creates datasets in the exact format needed for LLM instruction tuning.

## 🎯 Output Format

Each row in the final dataset follows this structure:
```jsonl
{"instruction": "What permits do I need to start a home-based daycare business?", "context": "", "response": "You need: 1) State child care license, 2) Home occupation permit, 3) Business license, 4) Food handler permit (if serving meals), 5) CPR/first aid certification, 6) Home safety inspection, 7) Background checks."}
```

## 🏗️ Architecture

The pipeline consists of three specialized agents:

### 1. **Question Generator Agent** (`langchain_question_generator.py`)
- Generates diverse, unique questions about your topic
- Tracks previously generated questions to avoid duplicates
- Supports custom system prompts
- Creates questions in batches for efficiency
- Can resume from existing files to add more questions

**Key Features:**
- Real-time duplicate checking during generation
- Category-based question generation
- Adjustable diversity via temperature
- Progress tracking and statistics
- Type-safe with Pydantic models

### 2. **Question Deduplicator Agent** (`langchain_question_deduplicator.py`)
- Removes semantically similar questions using embeddings
- Uses Amazon Titan Embeddings for semantic similarity
- Two methods: threshold-based or clustering-based
- Provides detailed statistics on removed duplicates

**Key Features:**
- Cosine similarity for semantic matching
- DBSCAN clustering for grouping similar questions
- Configurable similarity thresholds
- Shows sample duplicate groups for review
- Automatic batching with LangChain

### 3. **Response Generator Agent** (`langchain_response_generator.py`)
- Generates context and responses for questions
- Processes questions in chunks for efficiency
- Supports custom system prompts
- Can resume from partially completed datasets
- Parallel processing support

**Key Features:**
- Batch processing for cost efficiency
- Automatic retry on failures
- Progress tracking
- Generates context when needed
- Incremental writing (saves after each chunk)
- Parallel processing with configurable workers

### 4. **Response Improver Agent** (`langchain_response_improver.py`) (Optional)
- Evaluates and improves generated responses
- Uses a different (often larger) model for evaluation
- Fixes issues: grammar, clarity, missing sections
- Enhances good responses

**Key Features:**
- Custom evaluation criteria
- Batch processing with parallel support
- Incremental writing
- Detailed improvement statistics

## 📦 Installation

```bash
# Clone or download the repository
cd aileague

# Install dependencies
pip install -r requirements_langchain.txt

# Configure AWS credentials
aws configure
```

## 🚀 Quick Start

### Option 1: Use the Complete Pipeline (Recommended)

Run all agents in sequence using YAML configuration:

```bash
python agents/langchain_pipeline.py --config agents/config.yaml
```

Or with command-line arguments:
```bash
python agents/langchain_pipeline.py "customer service scenarios" --num-questions 500
```

This will:
1. Generate 500 diverse questions
2. Remove semantically similar duplicates
3. Generate context and responses for each question
4. Save the final training data in JSONL format

### Option 2: Run Agents Individually

For more control, run each agent separately using the YAML config:

```bash
# Step 1: Generate questions
python agents/langchain_question_generator.py --config agents/config.yaml

# Step 2: Deduplicate
python agents/langchain_question_deduplicator.py --config agents/config.yaml

# Step 3: Generate responses
python agents/langchain_response_generator.py --config agents/config.yaml

# Step 4: Improve responses (optional)
python agents/langchain_response_improver.py --config agents/config.yaml
```

## ⚙️ Configuration

All agents support configuration via `config.yaml`. See `agents/config.yaml` for detailed configuration options.

### Key Configuration Sections:

- **`question`**: Question generator settings (model, system prompt, categories, batch size, etc.)
- **`dedup`**: Deduplicator settings (threshold, method, etc.)
- **`response`**: Response generator settings (model, system prompt, chunk size, temperature, max_tokens, etc.)
- **`improver`**: Response improver settings (model, evaluation criteria, etc.)
- **`pipeline`**: Pipeline-specific overrides

All agents can use the config file automatically - they will search for `config.yaml` in the current directory or `agents/config.yaml`.

## 📖 Detailed Usage

### Question Generator

```bash
# Using config.yaml (recommended)
python agents/langchain_question_generator.py --config agents/config.yaml

# With command-line arguments
python agents/langchain_question_generator.py "legal advice" --num-questions 500

# With categories for better coverage
python agents/langchain_question_generator.py "programming help" \
  --num-questions 1000 \
  --categories "debugging,syntax,best-practices,architecture,testing"

# With custom system prompt
python agents/langchain_question_generator.py "medical information" \
  --system-prompt "Generate questions that patients typically ask their doctors" \
  --num-questions 300

# Resume from existing file to add more
python agents/langchain_question_generator.py "customer service" \
  --num-questions 2000 \
  --resume-from output/questions_customer_service.jsonl

# Adjust diversity
python agents/langchain_question_generator.py "cooking recipes" \
  --num-questions 500 \
  --temperature 0.95 \
  --batch-size 15
```

### Question Deduplicator

```bash
# Using config.yaml
python agents/langchain_question_deduplicator.py --config agents/config.yaml

# With command-line arguments
python agents/langchain_question_deduplicator.py questions.jsonl

# Stricter similarity threshold (fewer duplicates removed)
python agents/langchain_question_deduplicator.py questions.jsonl --threshold 0.90

# Use clustering method
python agents/langchain_question_deduplicator.py questions.jsonl \
  --method clustering \
  --eps 0.15

# Specify output file
python agents/langchain_question_deduplicator.py questions.jsonl \
  --output unique_questions.jsonl \
  --threshold 0.85
```

### Response Generator

```bash
# Using config.yaml
python agents/langchain_response_generator.py --config agents/config.yaml

# With command-line arguments
python agents/langchain_response_generator.py questions.jsonl

# Process more questions per API call
python agents/langchain_response_generator.py questions.jsonl --chunk-size 10

# With custom system prompt for responses
python agents/langchain_response_generator.py questions.jsonl \
  --system-prompt "Provide detailed, accurate answers suitable for training an AI assistant"

# Skip context generation (faster)
python agents/langchain_response_generator.py questions.jsonl --no-context

# Resume from partially completed file
python agents/langchain_response_generator.py questions.jsonl \
  --resume-from training_data_partial.jsonl \
  --output training_data_complete.jsonl

# Adjust creativity and parallel processing
python agents/langchain_response_generator.py questions.jsonl \
  --temperature 0.8 \
  --max-tokens 3000 \
  --max-workers 3

# Increase timeout for long responses
python agents/langchain_response_generator.py questions.jsonl \
  --timeout 600 \
  --chunk-size 1
```

### Response Improver (Optional)

```bash
# Using config.yaml
python agents/langchain_response_improver.py --config agents/config.yaml

# With command-line arguments
python agents/langchain_response_improver.py fine-tuning-dataset.jsonl

# Use a different/larger model for evaluation
python agents/langchain_response_improver.py fine-tuning-dataset.jsonl \
  --model-id "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Process in parallel
python agents/langchain_response_improver.py fine-tuning-dataset.jsonl \
  --max-workers 4

# Only fix issues (don't improve already good responses)
python agents/langchain_response_improver.py fine-tuning-dataset.jsonl \
  --fix-only
```

### Complete Pipeline

```bash
# Using config.yaml (recommended)
python agents/langchain_pipeline.py --config agents/config.yaml

# With command-line arguments
python agents/langchain_pipeline.py "tech support" --num-questions 1000

# With all customizations
python agents/langchain_pipeline.py "financial advice" \
  --num-questions 2000 \
  --question-batch-size 25 \
  --question-system-prompt "Generate questions about personal finance and investing" \
  --categories "budgeting,investing,retirement,taxes,debt" \
  --dedup-threshold 0.88 \
  --response-chunk-size 8 \
  --response-system-prompt "Provide clear, practical financial advice" \
  --output-dir ./datasets/finance

# Quick dataset without context
python agents/langchain_pipeline.py "movie recommendations" \
  --num-questions 500 \
  --no-context \
  --output-dir ./datasets/movies
```

## 🎛️ Configuration Options

### Question Generator Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-questions` | 100 | Total questions to generate |
| `--batch-size` | 20 | Questions per batch |
| `--temperature` | 0.9 | Diversity (0.0-1.0) |
| `--system-prompt` | None | Custom system prompt |
| `--categories` | None | Comma-separated categories |
| `--resume-from` | None | Resume from existing file |
| `--region` | us-east-1 | AWS region |
| `--config` | Auto-discovered | Path to config.yaml |

### Deduplicator Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 0.85 | Similarity threshold (0.0-1.0) |
| `--method` | threshold | `threshold` or `clustering` |
| `--eps` | 0.15 | DBSCAN epsilon for clustering |
| `--region` | us-east-1 | AWS region |
| `--config` | Auto-discovered | Path to config.yaml |

### Response Generator Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chunk-size` | 5 | Questions per API call |
| `--temperature` | 0.7 | Response creativity |
| `--max-tokens` | 2048 | Max tokens per response |
| `--timeout` | 300 | Request timeout (seconds) |
| `--max-workers` | 1 | Parallel workers (1 = sequential) |
| `--no-context` | False | Skip context generation |
| `--system-prompt` | None | Custom system prompt |
| `--resume-from` | None | Resume from partial file |
| `--region` | us-east-1 | AWS region |
| `--config` | Auto-discovered | Path to config.yaml |

### Response Improver Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chunk-size` | 5 | Examples per evaluation call |
| `--temperature` | 0.3 | Lower for consistent improvements |
| `--max-tokens` | 4096 | Max tokens per evaluation |
| `--timeout` | 600 | Request timeout (seconds) |
| `--max-workers` | 1 | Parallel workers |
| `--fix-only` | False | Only fix issues, don't improve good responses |
| `--config` | Auto-discovered | Path to config.yaml |

## 💡 Use Cases

### 1. Customer Support Training
```bash
python agents/langchain_pipeline.py "customer support for e-commerce platform" \
  --num-questions 2000 \
  --categories "order-issues,returns,payments,shipping,account" \
  --response-system-prompt "Provide empathetic, solution-focused customer service responses"
```

### 2. Technical Documentation Q&A
```bash
python agents/langchain_pipeline.py "Python programming help" \
  --num-questions 1500 \
  --categories "syntax,debugging,libraries,best-practices,performance" \
  --question-system-prompt "Generate technical questions developers ask" \
  --response-system-prompt "Provide code examples and clear technical explanations"
```

### 3. Educational Content
```bash
python agents/langchain_pipeline.py "high school mathematics" \
  --num-questions 1000 \
  --categories "algebra,geometry,trigonometry,calculus,statistics" \
  --response-system-prompt "Explain concepts clearly with step-by-step solutions"
```

### 4. Domain-Specific Assistant
```bash
python agents/langchain_pipeline.py "medical information for primary care" \
  --num-questions 800 \
  --categories "symptoms,diagnosis,treatment,prevention,medications" \
  --question-system-prompt "Generate questions patients ask doctors" \
  --response-system-prompt "Provide accurate medical information in accessible language"
```

### 5. Product Knowledge Base
```bash
python agents/langchain_pipeline.py "SaaS product features and troubleshooting" \
  --num-questions 600 \
  --categories "features,setup,troubleshooting,integrations,billing" \
  --no-context
```

## 📊 Output Structure

The pipeline creates several files:

```
output/
├── questions_topic_timestamp.jsonl              # Raw generated questions
├── questions_topic_timestamp_stats.json         # Question generation stats
├── questions_deduplicated_topic_timestamp.jsonl # After deduplication
├── questions_deduplicated_..._dedup_stats.json  # Deduplication stats
├── training_data_topic_timestamp.jsonl          # Final training data ✨
├── training_data_..._response_stats.json        # Response generation stats
├── training_data_..._improved.jsonl             # Improved responses (optional)
└── training_data_..._improved_stats.json        # Improvement stats (optional)
```

### Example Training Data Entry
```json
{
  "instruction": "How do I reset my password if I forgot my email?",
  "context": "Users can reset passwords through email verification or by contacting support if they don't have access to their registered email.",
  "response": "If you don't have access to your registered email, you have two options: 1) Contact our support team at support@company.com or call 1-800-SUPPORT with proof of identity, or 2) If you have a backup email or phone number on file, use the 'Alternative verification' option on the password reset page. Our team will verify your identity and help you regain access within 24 hours."
}
```

## 🔧 Advanced Features

### Resuming Interrupted Pipelines

All agents support resuming:

```bash
# If question generation was interrupted
python agents/langchain_question_generator.py "topic" \
  --num-questions 5000 \
  --resume-from output/questions_topic.jsonl

# If response generation was interrupted
python agents/langchain_response_generator.py questions.jsonl \
  --resume-from output/training_data_partial.jsonl
```

### Parallel Processing

Response generator and improver support parallel processing:

```bash
# Process multiple chunks simultaneously
python agents/langchain_response_generator.py questions.jsonl \
  --chunk-size 5 \
  --max-workers 3  # Process 3 chunks in parallel
```

### Quality Control

Review statistics files to ensure quality:

```bash
# Check question diversity
cat output/questions_*_stats.json | jq '.categories'

# Check deduplication effectiveness
cat output/*_dedup_stats.json | jq '.removal_rate'

# Check response success rate
cat output/*_response_stats.json | jq '.successful, .failed'
```

### Iterative Improvement

Generate questions in stages:

```bash
# Stage 1: Initial dataset
python agents/langchain_pipeline.py "topic" --num-questions 1000

# Review and refine

# Stage 2: Add more questions
python agents/langchain_question_generator.py "topic" \
  --num-questions 2000 \
  --resume-from output/questions_topic.jsonl

# Deduplicate and add responses
python agents/langchain_question_deduplicator.py output/questions_topic.jsonl
python agents/langchain_response_generator.py output/questions_topic_deduplicated.jsonl

# Optionally improve responses
python agents/langchain_response_improver.py output/training_data.jsonl
```

## 🎯 Best Practices

1. **Start Small**: Test with 50-100 questions first to verify quality
2. **Use Categories**: Specify categories for better topic coverage
3. **Use YAML Config**: Centralize all settings in `config.yaml`
4. **Tune Thresholds**: Adjust deduplication threshold based on your needs
   - 0.80-0.85: Aggressive (more duplicates removed)
   - 0.85-0.90: Balanced (recommended)
   - 0.90-0.95: Conservative (fewer duplicates removed)
5. **System Prompts**: Use them to control tone and style
6. **Chunk Size**: Balance cost and speed
   - Smaller (1-3): Better quality, more API calls, lower risk on errors
   - Larger (8-10): Fewer API calls but risk losing more on errors
   - Recommended: 1 for detailed responses, 5 for balance
7. **Review Samples**: Always check statistics and sample outputs
8. **Use Response Improver**: For critical datasets, run the improver with a stronger model

## 💰 Cost Estimation

Approximate costs with Claude 3.5 Sonnet on AWS Bedrock:

| Questions | Est. Cost | Notes |
|-----------|-----------|-------|
| 100 | $0.50 | Good for testing |
| 500 | $2.50 | Small dataset |
| 1,000 | $5.00 | Medium dataset |
| 5,000 | $25.00 | Large dataset |

*Costs include all agents. Actual costs vary based on question complexity and response length.*

## 🔍 Troubleshooting

### AWS Credentials Not Found
```bash
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Model Access Denied
- Enable model access in AWS Bedrock console
- Ensure IAM permissions include `bedrock:InvokeModel`
- Check if using inference profile ARN (requires special setup)

### Config File Not Found
Agents automatically search for `config.yaml` in:
1. Current directory
2. `agents/config.yaml`
3. Path specified by `--config` argument
4. Environment variable `AILEAGUE_CONFIG`

### High Duplicate Rate
- Increase question diversity with higher temperature
- Use categories to guide generation
- Ensure batch size isn't too small
- Check system prompt encourages diversity

### Low Response Quality
- Adjust system prompt for better guidance
- Lower temperature for more focused responses
- Increase max_tokens for longer responses
- Reduce chunk_size to 1 for detailed responses
- Use response improver with a stronger model

### Rate Limiting
- Reduce chunk size or max_workers
- Add delays between batches
- Check AWS service quotas

### Timeout Errors
- Increase `--timeout` parameter (default 300s)
- Reduce `--chunk-size` to 1
- Reduce `--max-tokens` if responses are too long

## 🛠️ Technology Stack

- **LangChain**: Framework for LLM applications
- **AWS Bedrock**: LLM inference service
- **Claude 3.5 Sonnet**: Primary LLM for generation
- **Amazon Titan Embeddings**: For semantic similarity
- **Pydantic**: Type-safe data validation
- **Python**: Core language

## 📝 License

MIT License - feel free to use and modify

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional deduplication algorithms
- Multi-language support
- Custom embedding models
- Quality scoring
- Active learning for iterative improvement

## 📚 Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [LLM Fine-tuning Best Practices](https://platform.openai.com/docs/guides/fine-tuning)

