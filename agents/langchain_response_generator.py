#!/usr/bin/env python3
"""
Response Generation Agent using LangChain
Generates context and responses for questions in chunks.
"""

import json
import argparse
from typing import Any, Dict
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field


# Pydantic models for structured output
class TrainingExample(BaseModel):
    """A complete training example"""
    instruction: str = Field(description="The question")
    context: str = Field(description="Relevant context or background information")
    response: str = Field(description="Detailed, helpful response")


class TrainingBatch(BaseModel):
    """Batch of training examples"""
    examples: List[TrainingExample] = Field(description="List of training examples")


@dataclass
class ResponseGenConfig:
    """Configuration for response generation"""
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    region: str = "us-east-1"
    system_prompt: Optional[str] = None
    topic: Optional[str] = None  # Topic/context for response generation
    requirements: Optional[str] = None  # Custom requirements section for prompt
    chunk_size: int = 5
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 300  # Request timeout in seconds (default: 5 minutes)
    max_workers: int = 1  # Number of parallel workers (1 = sequential, 2+ = parallel)
    add_context: bool = True


class ResponseGenerationAgent:
    """LangChain-based agent for response generation"""
    
    def __init__(self, config: ResponseGenConfig):
        """
        Initialize the response generation agent
        
        Args:
            config: Configuration for response generation
        """
        self.config = config
        
        # Configure boto3 client with timeout for ChatBedrock
        boto_config = Config(
            read_timeout=config.timeout,
            retries={'max_attempts': 3, 'mode': 'standard'}
        )
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=config.region,
            config=boto_config
        )
        
        # Initialize LangChain components
        # Handle ARN vs model ID - if ARN, need provider
        llm_kwargs = {
            "model_id": config.model_id,
            "region_name": config.region,
            "client": bedrock_client,
            "model_kwargs": {
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
        }
        # If using an ARN (inference profile), extract or set provider
        if config.model_id.startswith("arn:aws:bedrock"):
            # Try to detect provider from model ID if it contains the model name
            if "anthropic" in config.model_id.lower() or "claude" in config.model_id.lower():
                llm_kwargs["provider"] = "anthropic"
            elif "amazon" in config.model_id.lower() or "titan" in config.model_id.lower():
                llm_kwargs["provider"] = "amazon"
            elif "meta" in config.model_id.lower() or "llama" in config.model_id.lower():
                llm_kwargs["provider"] = "meta"
            else:
                # Default to anthropic for inference profiles (most common)
                llm_kwargs["provider"] = "anthropic"
        
        self.llm = ChatBedrock(**llm_kwargs)
        
        # Setup output parser
        self.parser = JsonOutputParser(pydantic_object=TrainingBatch)
        
        # Create prompt template
        self.prompt = self._create_prompt_template()
        
        # Create the chain
        self.chain = self._build_chain()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for response generation"""
        
        # Build system prompt: combine base system_prompt with requirements if provided
        system_prompt_parts = []
        base_system = self.config.system_prompt or "You are an expert at providing detailed, accurate, and helpful responses to questions."
        system_prompt_parts.append(base_system)
        
        # Add requirements to system prompt if provided (better than repeating in user message)
        if self.config.requirements:
            system_prompt_parts.append("\n\n" + self.config.requirements)
        
        system_message = "\n".join(system_prompt_parts)
        
        context_instruction = """- "context": Relevant background information or context needed to understand the response (can be empty string "" if no context needed)""" if self.config.add_context else """- "context": Leave as empty string ""."""
        
        # Include topic context if available
        if self.config.topic:
            topic_context = "\n\nCONTEXT: {topic}\n"
        else:
            topic_context = ""
        
        user_template = """Generate detailed, accurate, and helpful responses for the following questions.""" + topic_context + """

QUESTIONS:
{questions_list}

For each question, provide:
- A comprehensive, well-structured response
{context_instruction}

IMPORTANT: Return exactly {num_questions} objects in the array, matching the order of questions above.

{format_instructions}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_template)
        ])
    
    def _build_chain(self):
        """Build the LangChain chain"""
        chain_input = {
            "questions_list": RunnablePassthrough(),
            "num_questions": RunnablePassthrough(),
            "context_instruction": lambda _: 
                """- "context": Relevant background information or context needed to understand the response (can be empty string "" if no context needed)"""
                if self.config.add_context else 
                """- "context": Leave as empty string "".""",
            "format_instructions": lambda _: self.parser.get_format_instructions()
        }
        # Add topic if available
        if self.config.topic:
            chain_input["topic"] = lambda _: self.config.topic
        
        return (
            chain_input
            | self.prompt
            | self.llm
            | self.parser
        )
    
    def _format_questions_list(self, questions: List[Dict]) -> str:
        """Format questions for the prompt"""
        return "\n".join([
            f"{i+1}. {q['instruction']}"
            for i, q in enumerate(questions)
        ])
    
    def generate_responses_for_chunk(self, questions: List[Dict]) -> List[Dict]:
        """
        Generate responses for a chunk of questions
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            List of complete training examples
        """
        try:
            # Prepare input
            chain_input = {
                "questions_list": self._format_questions_list(questions),
                "num_questions": len(questions)
            }
            # Add topic if available
            if self.config.topic:
                chain_input["topic"] = self.config.topic
            
            # Invoke chain
            result = self.chain.invoke(chain_input)
            
            # Parse results
            if isinstance(result, dict) and 'examples' in result:
                examples = result['examples']
            elif isinstance(result, list):
                examples = result
            else:
                examples = []
            
            # Check if we got all responses
            if len(examples) < len(questions):
                avg_tokens = self.config.max_tokens // len(questions)
                print(f"  ⚠ Warning: Got {len(examples)} responses for {len(questions)} questions")
                print(f"  💡 Token budget: ~{avg_tokens} tokens per response (max_tokens: {self.config.max_tokens})")
                if avg_tokens < 200:
                    print(f"  💡 Consider reducing chunk_size to {max(1, len(questions) // 2)} or increasing max_tokens")
            
            # Build training examples
            training_examples = []
            for i, q in enumerate(questions):
                if i < len(examples):
                    if isinstance(examples[i], dict):
                        example = TrainingExample(**examples[i])
                    else:
                        example = examples[i]
                    
                    training_example = {
                        'instruction': q['instruction'],
                        'context': example.context,
                        'response': example.response
                    }
                    # Check if response is suspiciously short (might be truncated)
                    response_text = example.response or ''
                    if response_text and len(response_text) < 20 and not response_text.endswith('.'):
                        print(f"  ⚠ Warning: Response {i+1} is very short ({len(response_text)} chars) - may be truncated")
                else:
                    training_example = {
                        'instruction': q['instruction'],
                        'context': '',
                        'response': '[Response generation failed]'
                    }
                
                # Preserve additional fields
                for key in q:
                    if key not in training_example:
                        training_example[key] = q[key]
                
                training_examples.append(training_example)
            
            return training_examples
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            # Return failed examples
            return [
                {
                    'instruction': q['instruction'],
                    'context': '',
                    'response': '[Response generation failed]',
                    **{k: v for k, v in q.items() if k != 'instruction'}
                }
                for q in questions
            ]
    
    def generate_responses(self, input_file: str, output_file: Optional[str] = None, resume_from: Optional[str] = None) -> List[Dict]:
        """
        Generate responses for all questions
        
        Args:
            input_file: Path to input JSONL file
            output_file: Optional path to output file (for incremental writing)
            resume_from: Optional path to partially completed file (defaults to output_file if not provided)
            
        Returns:
            List of complete training examples
        """
        # Determine resume file (use output_file if resume_from not provided)
        resume_file = resume_from or output_file
        
        # Load questions
        print(f"Loading questions from {input_file}...")
        questions = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        
        print(f"✓ Loaded {len(questions)} questions")
        
        # Check if resuming from existing file
        completed = []
        completed_instructions = set()
        file_mode = 'a' if (resume_file and os.path.exists(resume_file)) else 'w'
        
        if resume_file and os.path.exists(resume_file):
            print(f"Resuming from {resume_file}...")
            with open(resume_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        if example.get('response') and example['response'] != '[Response generation failed]':
                            completed.append(example)
                            completed_instructions.add(example['instruction'])
            print(f"✓ Loaded {len(completed)} completed examples")
        
        # Filter remaining
        remaining = [q for q in questions if q['instruction'] not in completed_instructions]
        
        if not remaining:
            print("✓ All questions already have responses!")
            return completed
        
        print(f"Generating responses for {len(remaining)} questions...")
        
        # Process in chunks
        num_chunks = (len(remaining) + self.config.chunk_size - 1) // self.config.chunk_size
        avg_tokens_per_response = self.config.max_tokens // self.config.chunk_size
        print(f"Processing in {num_chunks} chunks of {self.config.chunk_size}")
        print(f"Token budget: {self.config.max_tokens} max tokens per chunk (~{avg_tokens_per_response} tokens per response)")
        if avg_tokens_per_response < 200:
            print(f"  ⚠ Warning: Low token budget per response. Consider reducing chunk_size or increasing max_tokens")
        if output_file:
            print(f"Incremental writing: Results will be saved to {output_file} after each chunk")
        if self.config.max_workers > 1:
            print(f"Parallel processing: {self.config.max_workers} workers")
        print("=" * 60)
        
        all_examples = completed.copy()
        
        # Thread-safe file writing lock
        file_lock = Lock() if output_file else None
        
        # Open file for incremental writing if output_file provided
        file_handle = None
        if output_file:
            file_handle = open(output_file, file_mode, encoding='utf-8')
            if file_mode == 'w':
                print(f"✓ Opened {output_file} for writing")
            else:
                print(f"✓ Opened {output_file} for appending (resume mode)")
        
        # Worker function for processing a single chunk
        def process_chunk(chunk_data):
            """Process a single chunk and return results"""
            chunk_idx, chunk = chunk_data
            chunk_num = (chunk_idx // self.config.chunk_size) + 1
            
            try:
                examples = self.generate_responses_for_chunk(chunk)
                
                # Write incrementally to file if output_file provided (thread-safe)
                if file_handle and file_lock:
                    with file_lock:
                        for example in examples:
                            file_handle.write(json.dumps(example, ensure_ascii=False) + '\n')
                            file_handle.flush()  # Ensure immediate write to disk
                
                valid = sum(1 for e in examples if e['response'] and e['response'] != '[Response generation failed]')
                return {
                    'chunk_num': chunk_num,
                    'examples': examples,
                    'valid': valid,
                    'success': True
                }
                
            except Exception as e:
                print(f"✗ Error in chunk {chunk_num}: {str(e)}")
                # Create failed examples
                failed_examples = []
                for q in chunk:
                    failed_example = {
                        'instruction': q['instruction'],
                        'context': '',
                        'response': '[Response generation failed]'
                    }
                    failed_examples.append(failed_example)
                
                # Write failed examples to file too (thread-safe)
                if file_handle and file_lock:
                    with file_lock:
                        for example in failed_examples:
                            file_handle.write(json.dumps(example, ensure_ascii=False) + '\n')
                            file_handle.flush()
                
                return {
                    'chunk_num': chunk_num,
                    'examples': failed_examples,
                    'valid': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Prepare chunk data
        chunk_data_list = [
            (chunk_idx, remaining[chunk_idx:chunk_idx + self.config.chunk_size])
            for chunk_idx in range(0, len(remaining), self.config.chunk_size)
        ]
        
        # Process chunks (sequential or parallel based on max_workers)
        if self.config.max_workers == 1:
            # Sequential processing (original behavior)
            for chunk_data in chunk_data_list:
                chunk_idx, chunk = chunk_data
                chunk_num = (chunk_idx // self.config.chunk_size) + 1
                print(f"\nChunk {chunk_num}/{num_chunks} ({len(chunk)} questions)")
                print(f"Progress: {len(all_examples)}/{len(questions)} total")
                
                result = process_chunk(chunk_data)
                all_examples.extend(result['examples'])
                
                if result['success']:
                    if file_handle:
                        print(f"✓ Wrote {len(result['examples'])} examples to file")
                    print(f"✓ Generated {result['valid']}/{len(result['examples'])} valid responses")
                else:
                    print(f"✗ Chunk {chunk_num} failed: {result.get('error', 'Unknown error')}")
                
                # Small delay to avoid rate limits
                if chunk_num < num_chunks:
                    time.sleep(0.5)
        else:
            # Parallel processing
            print(f"Using {self.config.max_workers} parallel workers")
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all chunks
                future_to_chunk = {
                    executor.submit(process_chunk, chunk_data): chunk_data
                    for chunk_data in chunk_data_list
                }
                
                # Process results as they complete
                for future in as_completed(future_to_chunk):
                    result = future.result()
                    all_examples.extend(result['examples'])
                    completed_count += 1
                    
                    if result['success']:
                        print(f"✓ Chunk {result['chunk_num']}/{num_chunks}: {result['valid']}/{len(result['examples'])} valid responses")
                    else:
                        print(f"✗ Chunk {result['chunk_num']}/{num_chunks} failed: {result.get('error', 'Unknown error')}")
                    
                    print(f"Progress: {len(all_examples)}/{len(questions)} total ({completed_count}/{num_chunks} chunks)")
        
        # Close file handle
        if file_handle:
            file_handle.close()
            print(f"\n✓ Closed output file: {output_file}")
        
        print(f"\n{'='*60}")
        print(f"✓ Total examples: {len(all_examples)}")
        
        successful = sum(1 for e in all_examples if e['response'] and e['response'] != '[Response generation failed]')
        print(f"✓ Successful responses: {successful}/{len(all_examples)}")
        
        return all_examples
    
    def save_results(self, examples: List[Dict], output_file: str, stats_only: bool = False):
        """
        Save training examples to JSONL file (or just statistics if stats_only=True)
        
        Args:
            examples: List of training examples
            output_file: Path to output file
            stats_only: If True, only save statistics (don't overwrite file with examples)
        """
        if not stats_only:
            # Write all examples to file (legacy mode)
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            print(f"\n✓ Training data saved to: {output_file}")
        else:
            # File already written incrementally, just confirm
            if os.path.exists(output_file):
                print(f"\n✓ Training data already saved incrementally to: {output_file}")
        
        # Statistics
        stats = {
            'total_examples': len(examples),
            'successful': sum(1 for e in examples if e.get('response') and e['response'] != '[Response generation failed]'),
            'failed': sum(1 for e in examples if not e.get('response') or e['response'] == '[Response generation failed]'),
            'with_context': sum(1 for e in examples if e.get('context') and e['context'].strip()),
            'generated_at': datetime.now().isoformat(),
            'config': {
                'model_id': self.config.model_id,
                'chunk_size': self.config.chunk_size,
                'temperature': self.config.temperature,
                'add_context': self.config.add_context
            }
        }
        
        stats_file = output_file.replace('.jsonl', '_response_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Statistics saved to: {stats_file}")
        
        print(f"\n{'='*60}")
        print(f"Response Generation Summary:")
        print(f"  Total: {stats['total_examples']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Success rate: {stats['successful']/stats['total_examples']*100:.1f}%")
        print(f"{'='*60}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Generate responses using LangChain and AWS Bedrock"
    )
    
    parser.add_argument("input_file", nargs='?', default=None, type=str, help="Input JSONL file (optional if provided in YAML)")
    parser.add_argument("--chunk-size", "-c", type=int, default=5)
    parser.add_argument("--system-prompt", "-s", type=str, help="Custom system prompt")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--output", "-o", type=str, help="Output filename")
    parser.add_argument("--resume-from", "-r", type=str, help="Resume from partial file")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--model-id", type=str, default="anthropic.claude-3-5-sonnet-20241022-v2:0")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds (default: 300 / 5 minutes)")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of parallel workers (default: 1 = sequential, 2+ = parallel)")
    parser.add_argument("--config", type=str, help="Path to YAML config. Values override CLI where provided")
    
    args = parser.parse_args()

    # Load YAML config (explicit, env, fallback paths)
    yaml_cfg: Dict[str, Any] = {}
    def load_yaml_at(path: str) -> Dict[str, Any]:
        import yaml  # type: ignore
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Top-level YAML must be a mapping")
            return data
    try:
        if args.config:
            yaml_cfg = load_yaml_at(args.config)
        else:
            import os as _os
            env_path = _os.environ.get('AILEAGUE_CONFIG')
            candidates = [
                env_path,
                _os.path.join(_os.getcwd(), 'config.yaml'),
                _os.path.join(_os.getcwd(), 'agents', 'config.yaml'),
            ]
            for p in candidates:
                if p and _os.path.exists(p):
                    yaml_cfg = load_yaml_at(p)
                    break
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1

    # Load response-specific section from YAML
    r_cfg = yaml_cfg.get('response', {}) if isinstance(yaml_cfg.get('response'), dict) else {}
    
    input_file = r_cfg.get('input_file', args.input_file)
    if not input_file or not os.path.exists(input_file):
        print(f"❌ Error: Input file not found or unspecified. Set 'input_file' in YAML response section or pass as CLI.")
        return 1
    
    model_id = r_cfg.get('model_id', args.model_id)
    region = r_cfg.get('region', args.region)
    system_prompt = r_cfg.get('system_prompt', args.system_prompt)
    topic = r_cfg.get('topic', None)  # Optional topic for context
    requirements = r_cfg.get('requirements', None)  # Optional custom requirements
    chunk_size = int(r_cfg.get('chunk_size', args.chunk_size))
    temperature = float(r_cfg.get('temperature', args.temperature))
    max_tokens = int(r_cfg.get('max_tokens', args.max_tokens))
    timeout = int(r_cfg.get('timeout', args.timeout))
    max_workers = int(r_cfg.get('max_workers', args.max_workers))
    no_context_yaml = bool(r_cfg.get('no_context', args.no_context))

    config = ResponseGenConfig(
        model_id=model_id,
        region=region,
        system_prompt=system_prompt,
        topic=topic,
        requirements=requirements,
        chunk_size=chunk_size,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_workers=max_workers,
        add_context=not no_context_yaml
    )
    
    output_file_override = r_cfg.get('output')
    output_file = output_file_override if output_file_override else args.output
    if not output_file:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_training_data.jsonl"
    
    print("Response Generation Agent (LangChain)")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Model: {config.model_id}")
    print(f"Timeout: {config.timeout}s")
    print(f"Max workers: {config.max_workers}")
    print("=" * 60)
    
    try:
        agent = ResponseGenerationAgent(config)
        resume_from = r_cfg.get('resume_from', args.resume_from)
        # Generate responses (with incremental writing)
        examples = agent.generate_responses(input_file, output_file=output_file, resume_from=resume_from)
        
        # Save statistics (results already written incrementally)
        agent.save_results(examples, output_file, stats_only=True)
        print(f"\n✅ Success!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
