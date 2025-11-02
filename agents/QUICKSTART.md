# Quick Start Guide

Get started generating your LLM fine-tuning dataset in 3 steps!

## Prerequisites

```bash
pip install -r requirements_langchain.txt
aws configure  # Set up AWS credentials
```

## Step 1: Generate Your First Dataset (30 seconds)

Using YAML configuration (recommended):
```bash
cd agents
python langchain_pipeline.py --config config.yaml
```

Or with command-line arguments:
```bash
python agents/langchain_pipeline.py "your topic here" --num-questions 100
```

Example:
```bash
python agents/langchain_pipeline.py "customer service for tech support" --num-questions 100
```

This creates a complete training dataset in JSONL format, ready for fine-tuning!

## Step 2: Check Your Results

Look in the `output/` directory:
- `training_data_*.jsonl` - Your final dataset ✨
- `*_stats.json` - Statistics and quality metrics

## Step 3: Review Sample Data

```bash
head -n 1 output/training_data_*.jsonl | jq .
```

You'll see:
```json
{
  "instruction": "How do I troubleshoot connection issues?",
  "context": "",
  "response": "To troubleshoot connection issues: 1) Check your internet connection..."
}
```

## Common Commands

### Quick test (50 questions)
```bash
python agents/langchain_pipeline.py "your topic" --num-questions 50
```

### Larger dataset (1000 questions)
```bash
python agents/langchain_pipeline.py "your topic" --num-questions 1000
```

### With categories for better coverage
```bash
python agents/langchain_pipeline.py "programming help" \
  --num-questions 500 \
  --categories "debugging,syntax,best-practices,testing"
```

### Custom tone/style
```bash
python agents/langchain_pipeline.py "medical advice" \
  --num-questions 300 \
  --question-system-prompt "Generate patient questions" \
  --response-system-prompt "Provide clear medical information"
```

### Using YAML config (recommended)
```bash
cd agents
# Edit config.yaml with your settings
python langchain_pipeline.py --config config.yaml
```

## Example Topics

Try these topics:
- "customer service for e-commerce"
- "Python programming help"
- "cooking recipes and techniques"
- "financial planning advice"
- "legal information for small business"
- "medical symptoms and treatment"
- "product troubleshooting and support"
- "educational math tutoring"

## Next Steps

1. Start with 50-100 questions to test quality
2. Review the output and statistics
3. Adjust parameters in `config.yaml` or via CLI:
   - `--dedup-threshold` - Control duplicate removal (0.85 default)
   - `--question-temperature` - Control diversity (0.9 default)
   - `--response-chunk-size` - Questions per API call (5 default, 1 recommended for quality)
   - `--max-workers` - Parallel processing (1 default, 2-4 for speed)
4. Generate your full dataset!

## Individual Agents

Run agents separately for more control:

```bash
# Step 1: Generate questions
python agents/langchain_question_generator.py "your topic" --num-questions 1000

# Step 2: Deduplicate
python agents/langchain_question_deduplicator.py output/questions_*.jsonl

# Step 3: Generate responses
python agents/langchain_response_generator.py output/questions_*_deduplicated.jsonl

# Step 4: Improve responses (optional)
python agents/langchain_response_improver.py output/training_data_*.jsonl
```

## Need Help?

- Check `README.md` for full documentation
- See `config.yaml` for all configuration options
- Review the `*_stats.json` files to understand what happened

## Cost Estimate

Approximate costs per dataset:
- 100 questions: ~$0.50
- 500 questions: ~$2.50
- 1000 questions: ~$5.00

## Troubleshooting

**"No credentials found"**
```bash
aws configure
```

**"Access denied"**
- Enable Claude models in AWS Bedrock console
- Check IAM permissions
- Verify model ID is correct (or use inference profile ARN)

**"Too many duplicates"**
- Lower `--dedup-threshold` to 0.80
- Increase `--question-temperature` to 0.95
- Use categories to guide generation

**"Read timeout"**
- Increase `--timeout` to 600
- Reduce `--chunk-size` to 1
- Reduce `--max-tokens`

**Want to add more questions?**
```bash
python agents/langchain_question_generator.py "your topic" \
  --num-questions 1000 \
  --resume-from output/questions_*.jsonl
```

## That's It!

You're ready to generate high-quality fine-tuning datasets. Start small, review quality, then scale up!
