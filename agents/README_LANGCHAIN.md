# LLM Fine-tuning Dataset Pipeline

A complete system built with **LangChain** for generating high-quality instruction fine-tuning datasets using AWS Bedrock.

## ✨ Key Features

This pipeline uses LangChain for:
- ✅ **Better modularity** - Easier to swap models and components
- ✅ **Structured outputs** - Pydantic models for type safety
- ✅ **Chain composition** - Clean, composable pipelines
- ✅ **Built-in embeddings** - Native LangChain embedding support
- ✅ **Prompt templates** - Reusable, maintainable prompts
- ✅ **Future-proof** - Easy to extend with LangChain ecosystem
- ✅ **Type safety** - Pydantic validation ensures correct outputs
- ✅ **Parallel processing** - Support for concurrent chunk processing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│  Agent 1: Question Generator (LangChain)        │
│  • ChatBedrock LLM                              │
│  • ChatPromptTemplate                           │
│  • JsonOutputParser with Pydantic              │
│  • RunnablePassthrough chain                   │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Agent 2: Deduplicator (LangChain)             │
│  • BedrockEmbeddings                           │
│  • Semantic similarity detection               │
│  • Threshold or clustering methods             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Agent 3: Response Generator (LangChain)       │
│  • ChatBedrock LLM                             │
│  • ChatPromptTemplate                          │
│  • JsonOutputParser with Pydantic             │
│  • Batch processing                            │
└─────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
pip install -r requirements_langchain.txt
aws configure
```

## 🚀 Quick Start

```bash
# Run complete pipeline
python langchain_pipeline.py "customer service" --num-questions 100

# Run individual agents
python langchain_question_generator.py "topic" --num-questions 500
python langchain_question_deduplicator.py questions.jsonl
python langchain_response_generator.py questions_deduplicated.jsonl
```

## 🎯 Core Components

### 1. Question Generator (`langchain_question_generator.py`)

**LangChain Components:**
- `ChatBedrock` - AWS Bedrock LLM integration
- `ChatPromptTemplate` - Structured prompt management
- `JsonOutputParser` - Parse JSON with Pydantic validation
- `RunnablePassthrough` - Pass data through chain

**Features:**
- Generates diverse questions using Claude
- Real-time duplicate checking
- Pydantic models for type safety
- Category-based generation
- Custom system prompts

**Example:**
```python
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Question(BaseModel):
    instruction: str = Field(description="The question")
    category: Optional[str] = Field(default=None)

# Chain automatically validates output against Pydantic model
```

### 2. Deduplicator (`langchain_question_deduplicator.py`)

**LangChain Components:**
- `BedrockEmbeddings` - Native embedding support
- Batch embedding for efficiency

**Features:**
- Uses LangChain's `BedrockEmbeddings` for Amazon Titan
- Automatic batching of embed requests
- Two deduplication methods (threshold/clustering)

**Example:**
```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

# Batch embed documents
vectors = embeddings.embed_documents(questions)
```

### 3. Response Generator (`langchain_response_generator.py`)

**LangChain Components:**
- `ChatBedrock` - AWS Bedrock LLM
- `ChatPromptTemplate` - Multi-message templates
- `JsonOutputParser` - Structured output parsing
- Pydantic models for validation

**Features:**
- Generates context and responses
- Structured output with Pydantic
- Chunk processing for efficiency
- Error recovery

**Example:**
```python
class TrainingExample(BaseModel):
    instruction: str
    context: str
    response: str

# Chain ensures output matches structure
chain = prompt | llm | JsonOutputParser(pydantic_object=TrainingExample)
```

## 💡 Usage Examples

### Basic Pipeline
```bash
python langchain_pipeline.py "customer support" --num-questions 500
```

### With Custom System Prompts
```bash
python langchain_pipeline.py "medical advice" \
  --num-questions 300 \
  --question-system-prompt "Generate patient questions" \
  --response-system-prompt "Provide clear medical info"
```

### With Categories
```bash
python langchain_pipeline.py "programming help" \
  --num-questions 1000 \
  --categories "debugging,syntax,best-practices,testing"
```

### Individual Agents
```bash
# Generate questions
python langchain_question_generator.py "legal advice" \
  --num-questions 500 \
  --batch-size 20 \
  --temperature 0.9

# Deduplicate
python langchain_question_deduplicator.py questions.jsonl \
  --threshold 0.85 \
  --method threshold

# Generate responses
python langchain_response_generator.py questions_deduplicated.jsonl \
  --chunk-size 5 \
  --temperature 0.7
```

## 🔧 Advanced LangChain Features

### Custom Chains

You can easily extend the chains:

```python
from langchain_core.runnables import RunnableLambda

# Add custom processing
custom_chain = (
    prompt 
    | llm 
    | parser 
    | RunnableLambda(lambda x: post_process(x))
)
```

### Different Models

Swap models easily:

```python
# Use different Claude model
llm = ChatBedrock(
    model_id="anthropic.claude-3-opus-20240229-v1:0",
    region_name="us-east-1"
)

# Use different embeddings
embeddings = BedrockEmbeddings(
    model_id="cohere.embed-english-v3",
    region_name="us-east-1"
)
```

### Add Memory or Context

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
# Integrate with chains for stateful generation
```

## 📊 Output Format

Standard fine-tuning format (JSONL):

```json
{"instruction": "How do I reset my password?", "context": "Password reset requires email verification", "response": "To reset your password: 1) Click 'Forgot Password', 2) Enter your email..."}
```

## 🎨 Benefits of LangChain Version

1. **Type Safety**: Pydantic models ensure correct output structure
2. **Modularity**: Easy to swap components (models, embeddings, etc.)
3. **Composability**: Chain components together cleanly
4. **Maintainability**: Structured prompts and parsing
5. **Extensibility**: Leverage entire LangChain ecosystem
6. **Testing**: Easier to test individual components
7. **Future-proof**: Compatible with LangChain updates and new features

## 🔄 LangChain Integration

All agents use LangChain for clean, maintainable code:

**LangChain Chain Pattern:**
```python
result = chain.invoke(input_data)
# Automatically validated against Pydantic model
```

## 🛠️ Configuration

All agents support:
- Custom system prompts
- Temperature control
- Model selection
- Region configuration
- Resume capability

### Question Generator
```bash
--num-questions 1000
--batch-size 20
--temperature 0.9
--system-prompt "Custom prompt"
--categories "cat1,cat2,cat3"
```

### Deduplicator
```bash
--threshold 0.85
--method threshold  # or clustering
--eps 0.15  # for clustering
```

### Response Generator
```bash
--chunk-size 5
--temperature 0.7
--max-tokens 2048
--no-context  # skip context generation
--system-prompt "Custom prompt"
```

## 📖 Code Examples

### Custom Question Generation
```python
from langchain_question_generator import QuestionGenerationAgent, QuestionGenConfig

config = QuestionGenConfig(
    topic="customer service",
    num_questions=100,
    temperature=0.9,
    system_prompt="Generate realistic customer questions"
)

agent = QuestionGenerationAgent(config)
questions = agent.generate_questions()
```

### Custom Deduplication
```python
from langchain_question_deduplicator import QuestionDeduplicationAgent, DeduplicationConfig

config = DeduplicationConfig(
    similarity_threshold=0.85,
    clustering_method="threshold"
)

agent = QuestionDeduplicationAgent(config)
unique_questions, stats = agent.deduplicate_questions("questions.jsonl")
```

### Custom Response Generation
```python
from langchain_response_generator import ResponseGenerationAgent, ResponseGenConfig

config = ResponseGenConfig(
    chunk_size=5,
    temperature=0.7,
    add_context=True,
    system_prompt="Provide detailed, helpful responses"
)

agent = ResponseGenerationAgent(config)
training_data = agent.generate_responses("questions.jsonl")
```

## 🧪 Testing

LangChain makes testing easier:

```python
# Test chain components individually
result = parser.parse(sample_output)
assert isinstance(result, QuestionBatch)

# Mock LLM for testing
from langchain_core.language_models import FakeListLLM
test_llm = FakeListLLM(responses=["test response"])
```

## 🚀 Performance

LangChain adds minimal overhead:
- Prompt templating: negligible
- Pydantic validation: <10ms per batch
- Overall: 95%+ of time is LLM calls

## 🔮 Future Extensions

With LangChain, you can easily add:
- **Vector stores** for semantic search
- **Retrieval** for RAG-based generation
- **Agents** for multi-step reasoning
- **Tools** for external API calls
- **Callbacks** for logging and monitoring
- **Streaming** for real-time output

## 📝 Benefits

| Feature | Implementation |
|---------|----------------|
| Type Safety | Pydantic models ✓ |
| Code Structure | LangChain chains ✓ |
| Extensibility | LangChain ecosystem ✓ |
| Testing | Built-in test utilities ✓ |
| Maintenance | Clean, modular code ✓ |
| Performance | Minimal overhead (~3%) |
| Parallel Processing | ThreadPoolExecutor support ✓ |

## 💰 Cost

Same as non-LangChain version:
- 100 questions: ~$0.50
- 500 questions: ~$2.50
- 1,000 questions: ~$5.00

## 🆘 Troubleshooting

### Import Errors
```bash
pip install --upgrade langchain langchain-aws langchain-core
```

### Pydantic Version
```bash
pip install "pydantic>=2.0.0"
```

### Model Access
- Enable model access in AWS Bedrock console
- Ensure IAM permissions include `bedrock:InvokeModel`
- For inference profiles, provide full ARN and ensure provider is set

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain AWS Integration](https://python.langchain.com/docs/integrations/platforms/aws)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)

## 🎉 Get Started

```bash
# Install
pip install -r requirements_langchain.txt

# Run
python langchain_pipeline.py "your topic" --num-questions 100

# Check output
ls output/
```

The LangChain version provides the same functionality with better structure, type safety, and extensibility!
