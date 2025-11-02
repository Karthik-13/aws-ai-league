# Complete Package Index

## 📦 LangChain Implementation

This package provides a complete pipeline for generating high-quality fine-tuning datasets using **LangChain** and AWS Bedrock.

**Key Features:**
- Type-safe with Pydantic models
- Modular LangChain architecture
- Easy to extend and customize
- Parallel processing support
- Incremental writing for reliability

---

## 🗂️ File Organization

### Core Agent Files
- **[langchain_question_generator.py](langchain_question_generator.py)** - Question generation agent
- **[langchain_question_deduplicator.py](langchain_question_deduplicator.py)** - Deduplication agent
- **[langchain_response_generator.py](langchain_response_generator.py)** - Response generation agent
- **[langchain_response_improver.py](langchain_response_improver.py)** - Response improvement agent (optional)
- **[langchain_pipeline.py](langchain_pipeline.py)** - Complete pipeline orchestrator

### Configuration
- **[config.yaml](config.yaml)** - Unified configuration for all agents
- **[requirements_langchain.txt](requirements_langchain.txt)** - Python dependencies

### Documentation
- **[README.md](README.md)** - Main documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[README_LANGCHAIN.md](README_LANGCHAIN.md)** - Detailed LangChain documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
- **[COMPARISON.md](COMPARISON.md)** - Implementation details
- **[QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)** - Quick reference card
- **[INDEX_COMPLETE.md](INDEX_COMPLETE.md)** - This file

---

## 🚀 Quick Start Guide

### Installation

```bash
# Install dependencies
pip install -r requirements_langchain.txt

# Configure AWS
aws configure
```

### Run Pipeline

**Using config.yaml (recommended):**
```bash
cd agents
python langchain_pipeline.py --config config.yaml
```

**With command-line arguments:**
```bash
python agents/langchain_pipeline.py "customer service" --num-questions 100
```

### Individual Agents

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

---

## 🎯 Agent Overview

### Agent 1: Question Generator
**File:** `langchain_question_generator.py`

**Purpose:** Generate diverse, unique questions about your topic

**Key Features:**
- Real-time duplicate checking
- Category-based generation
- Custom system prompts
- Resume capability
- Pydantic type safety

**Usage:**
```bash
python agents/langchain_question_generator.py --config agents/config.yaml
```

### Agent 2: Question Deduplicator
**File:** `langchain_question_deduplicator.py`

**Purpose:** Remove semantically similar questions

**Key Features:**
- Semantic similarity detection using embeddings
- Two methods: threshold or clustering
- Automatic batch embedding
- Detailed statistics

**Usage:**
```bash
python agents/langchain_question_deduplicator.py --config agents/config.yaml
```

### Agent 3: Response Generator
**File:** `langchain_response_generator.py`

**Purpose:** Generate context and responses for questions

**Key Features:**
- Chunk processing for efficiency
- Parallel processing support
- Incremental writing (saves after each chunk)
- Configurable timeout
- Custom system prompts

**Usage:**
```bash
python agents/langchain_response_generator.py --config agents/config.yaml
```

### Agent 4: Response Improver (Optional)
**File:** `langchain_response_improver.py`

**Purpose:** Evaluate and improve generated responses

**Key Features:**
- Uses different/larger model for evaluation
- Fixes issues: grammar, clarity, missing sections
- Enhances good responses
- Parallel processing support
- Custom evaluation criteria

**Usage:**
```bash
python agents/langchain_response_improver.py --config agents/config.yaml
```

### Pipeline Orchestrator
**File:** `langchain_pipeline.py`

**Purpose:** Run all agents in sequence

**Key Features:**
- Runs all agents automatically
- Supports overrides via config.yaml
- Single command for complete pipeline

**Usage:**
```bash
python agents/langchain_pipeline.py --config agents/config.yaml
```

---

## 📖 Documentation Roadmap

### New to the Project?
1. Start: **[QUICKSTART.md](QUICKSTART.md)** - Get running in 30 seconds
2. Main docs: **[README.md](README.md)** - Complete documentation
3. Deep dive: **[README_LANGCHAIN.md](README_LANGCHAIN.md)** - LangChain details

### Understanding the System?
1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and flow
2. **[COMPARISON.md](COMPARISON.md)** - Implementation details

### Quick Reference
- **[QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)** - Command cheat sheet

---

## 🎨 Key Features

### ✅ All Agents Support
- YAML configuration via `config.yaml`
- Automatic config file discovery
- Command-line argument overrides
- Resume capability
- Statistics tracking
- Error handling and retries

### ✅ Technical Features
- **Type Safety:** Pydantic models ensure correct outputs
- **Modularity:** Easy to swap models and components
- **Chain Composition:** Clean LangChain pipelines
- **Parallel Processing:** ThreadPoolExecutor support
- **Incremental Writing:** Save results after each chunk
- **Timeout Handling:** Configurable request timeouts
- **Error Recovery:** Automatic retries with exponential backoff

---

## 🔧 Installation & Setup

### Dependencies

```bash
pip install -r requirements_langchain.txt
```

**Required packages:**
- langchain, langchain-aws, langchain-core
- boto3, botocore
- numpy, scikit-learn
- pydantic>=2.0.0
- pyyaml>=6.0

### AWS Configuration

```bash
aws configure
```

Ensure:
- AWS credentials are configured
- Bedrock models are enabled in console
- IAM permissions include `bedrock:InvokeModel`

---

## 💡 Example Commands

### Basic Pipeline

```bash
# Using config.yaml
cd agents
python langchain_pipeline.py --config config.yaml

# With CLI arguments
python agents/langchain_pipeline.py "customer service" --num-questions 500
```

### With Categories

```bash
python agents/langchain_pipeline.py "programming help" \
  --num-questions 1000 \
  --categories "debugging,syntax,testing"
```

### Custom System Prompts

```bash
python agents/langchain_pipeline.py "medical advice" \
  --num-questions 300 \
  --question-system-prompt "Generate patient questions" \
  --response-system-prompt "Provide clear medical info"
```

### Individual Agents

```bash
# Generate questions
python agents/langchain_question_generator.py "topic" --num-questions 500

# Deduplicate
python agents/langchain_question_deduplicator.py questions.jsonl

# Generate responses
python agents/langchain_response_generator.py questions_deduplicated.jsonl

# Improve responses
python agents/langchain_response_improver.py training_data.jsonl
```

### Parallel Processing

```bash
# Process multiple chunks simultaneously
python agents/langchain_response_generator.py questions.jsonl \
  --max-workers 3

# Improve responses in parallel
python agents/langchain_response_improver.py training_data.jsonl \
  --max-workers 4
```

### Resume from Interruption

```bash
# Resume question generation
python agents/langchain_question_generator.py "topic" \
  --num-questions 1000 \
  --resume-from output/questions_topic.jsonl

# Resume response generation
python agents/langchain_response_generator.py questions.jsonl \
  --resume-from output/training_data_partial.jsonl
```

---

## 📊 Output Format

Standard fine-tuning format (JSONL):

```json
{
  "instruction": "How do I reset my password?",
  "context": "Password reset requires email verification",
  "response": "To reset your password: 1) Click 'Forgot Password', 2) Enter your email..."
}
```

Ready for LLM instruction fine-tuning!

---

## 💰 Cost Estimates

Using Claude 3.5 Sonnet on AWS Bedrock:

| Questions | Est. Cost | Notes |
|-----------|-----------|-------|
| 100 | $0.50 | Good for testing |
| 500 | $2.50 | Small dataset |
| 1,000 | $5.00 | Medium dataset |
| 5,000 | $25.00 | Large dataset |

*With response improvement: add ~$2-3 per 1000 questions*

---

## 🎓 Learning Resources

### Documentation
- **[README.md](README.md)** - Main documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start
- **[README_LANGCHAIN.md](README_LANGCHAIN.md)** - LangChain details
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design

### External Resources
- [LangChain Documentation](https://python.langchain.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)

---

## 🔄 Configuration

All agents support configuration via `config.yaml`:

```yaml
question:
  model_id: "anthropic.claude-3-5-sonnet-20240620-v1:0"
  system_prompt: "..."
  num_questions: 400
  batch_size: 20
  temperature: 0.5
  categories: [...]

dedup:
  model_id: "amazon.titan-embed-text-v2:0"
  threshold: 0.88
  method: "threshold"

response:
  model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0"
  system_prompt: "..."
  chunk_size: 1
  temperature: 0.3
  max_tokens: 4096
  timeout: 600
  max_workers: 3

improver:
  model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0"
  chunk_size: 5
  max_workers: 4
  fix_only: false
```

See `config.yaml` for all available options.

---

## 🆘 Getting Help

1. **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
2. **Main Docs:** [README.md](README.md)
3. **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Reference:** [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)

### Common Issues

**"No credentials found"**
→ Run `aws configure`

**"Access denied"**
→ Enable Claude models in AWS Bedrock console
→ Check IAM permissions

**"Read timeout"**
→ Increase `--timeout` to 600
→ Reduce `--chunk-size` to 1

**"Too many duplicates"**
→ Lower `--dedup-threshold` to 0.80
→ Increase `--question-temperature` to 0.95

---

## 🎉 Summary

You have a **complete, production-ready pipeline** for generating fine-tuning datasets:

- ✅ **4 specialized agents** (Question Generator, Deduplicator, Response Generator, Response Improver)
- ✅ **Type-safe** with Pydantic models
- ✅ **Modular** LangChain architecture
- ✅ **Parallel processing** support
- ✅ **Incremental writing** for reliability
- ✅ **Complete documentation**
- ✅ **Unified configuration** via YAML

**Start generating high-quality fine-tuning datasets today!**

---

## 📝 File Count

- **Agent files:** 5 (4 agents + 1 pipeline)
- **Configuration:** 1 (config.yaml)
- **Documentation:** 7 files
- **Dependencies:** 1 (requirements_langchain.txt)

**Total: 14 core files** - Everything you need for professional LLM dataset generation!

Happy fine-tuning! 🚀
