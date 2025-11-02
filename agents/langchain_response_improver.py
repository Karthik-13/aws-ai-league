#!/usr/bin/env python3
"""
Response Improvement Agent using LangChain
Evaluates and improves responses using a different model for quality assurance.
"""

import json
import argparse
from typing import Any, Dict, List, Optional
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
class ImprovedTrainingExample(BaseModel):
    """An improved training example"""
    instruction: str = Field(description="The question")
    context: str = Field(description="Relevant context or background information")
    response: str = Field(description="Improved, detailed, helpful response")
    improvement_notes: Optional[str] = Field(default=None, description="Notes about what was improved")


class ImprovedTrainingBatch(BaseModel):
    """Batch of improved training examples"""
    examples: List[ImprovedTrainingExample] = Field(description="List of improved training examples")


@dataclass
class ResponseImproverConfig:
    """Configuration for response improvement"""
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"  # Evaluation model (can be different/larger)
    region: str = "us-east-1"
    system_prompt: Optional[str] = None
    evaluation_criteria: Optional[str] = None  # Custom evaluation criteria
    chunk_size: int = 5  # Examples per evaluation call
    temperature: float = 0.3  # Lower temperature for consistent improvements
    max_tokens: int = 4096
    timeout: int = 600  # Request timeout in seconds
    max_workers: int = 1  # Number of parallel workers
    fix_only: bool = False  # If True, only fix issues; if False, always improve even good responses


class ResponseImprovementAgent:
    """LangChain-based agent for evaluating and improving responses"""
    
    def __init__(self, config: ResponseImproverConfig):
        """
        Initialize the response improvement agent
        
        Args:
            config: Configuration for response improvement
        """
        self.config = config
        
        # Configure boto3 client with timeout
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
        llm_kwargs = {
            "model_id": config.model_id,
            "region_name": config.region,
            "client": bedrock_client,
            "model_kwargs": {
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
        }
        
        # Handle ARN vs model ID
        if config.model_id.startswith("arn:aws:bedrock"):
            if "anthropic" in config.model_id.lower() or "claude" in config.model_id.lower():
                llm_kwargs["provider"] = "anthropic"
            elif "amazon" in config.model_id.lower() or "titan" in config.model_id.lower():
                llm_kwargs["provider"] = "amazon"
            elif "meta" in config.model_id.lower() or "llama" in config.model_id.lower():
                llm_kwargs["provider"] = "meta"
            else:
                llm_kwargs["provider"] = "anthropic"
        
        self.llm = ChatBedrock(**llm_kwargs)
        
        # Setup output parser
        self.parser = JsonOutputParser(pydantic_object=ImprovedTrainingBatch)
        
        # Create prompt template
        self.prompt = self._create_prompt_template()
        
        # Create the chain
        self.chain = self._build_chain()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for response improvement"""
        
        # Build system prompt
        system_prompt_parts = []
        base_system = self.config.system_prompt or "You are an expert at evaluating and improving training data responses. You ensure responses are coherent, complete, and meet all quality standards."
        system_prompt_parts.append(base_system)
        
        # Default evaluation criteria
        if self.config.evaluation_criteria:
            criteria = self.config.evaluation_criteria
        else:
            criteria = """Evaluate responses for:
1. **Coherence**: Does the response flow logically and make sense?
2. **Completeness**: Does it include all required sections (Direct Answer, Eligibility/Requirements, Step-by-Step Process, Timeline, Alternative Options, Important Notes, Contact Information)?
3. **Accuracy**: Are specific details (fees, timelines, requirements) accurate and specific?
4. **Clarity**: Is the language clear and professional?
5. **Structure**: Is the formatting clean and readable?
6. **Helpfulness**: Does it fully answer the question and provide actionable guidance?

Improve responses by:
- Fixing grammatical errors or unclear language
- Ensuring all required sections are present and complete
- Adding missing specific details (fees, timelines, contact info)
- Improving clarity and readability
- Maintaining professional tone
- Ensuring logical flow between sections"""

        system_prompt_parts.append("\n\nEvaluation Criteria:\n" + criteria)
        system_message = "\n".join(system_prompt_parts)
        
        user_template = """Review and improve the following training examples. Each example has an instruction (question), context (optional), and response.

TRAINING EXAMPLES TO IMPROVE:
{examples_text}

IMPORTANT INSTRUCTIONS:
- Evaluate each response against the criteria above
- Improve responses that have issues (grammar, clarity, missing sections, incomplete information)
- Enhance good responses to make them even better
- Preserve the original meaning and information
- Ensure responses maintain the required structure
- If a response is already excellent, make minimal changes or note it as already good
- Return improved responses in the same order as input
- Only change what needs improvement - don't rewrite unnecessarily

{format_instructions}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_template)
        ])
    
    def _build_chain(self):
        """Build the LangChain chain"""
        chain_input = {
            "examples_text": RunnablePassthrough(),
            "format_instructions": lambda _: self.parser.get_format_instructions()
        }
        
        return (
            chain_input
            | self.prompt
            | self.llm
            | self.parser
        )
    
    def _format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """Format examples for the evaluation prompt"""
        formatted = []
        for i, ex in enumerate(examples, 1):
            context_str = f"\nContext: {ex.get('context', '')}" if ex.get('context') else ""
            formatted.append(f"""Example {i}:
Instruction: {ex.get('instruction', '')}
{context_str}
Response: {ex.get('response', '')}
---""")
        return "\n\n".join(formatted)
    
    def improve_responses_for_chunk(self, examples: List[Dict]) -> List[Dict]:
        """Improve responses for a chunk of examples"""
        try:
            examples_text = self._format_examples_for_prompt(examples)
            
            chain_input = {
                "examples_text": examples_text
            }
            
            result = self.chain.invoke(chain_input)
            
            # Parse results
            if isinstance(result, dict) and 'examples' in result:
                improved = result['examples']
            elif isinstance(result, list):
                improved = result
            else:
                improved = []
            
            # Ensure we return the same number of examples
            improved_examples = []
            for i, original in enumerate(examples):
                if i < len(improved):
                    imp = improved[i]
                    improved_examples.append({
                        'instruction': imp.get('instruction', original.get('instruction', '')),
                        'context': imp.get('context', original.get('context', '')),
                        'response': imp.get('response', original.get('response', '')),
                        'improvement_notes': imp.get('improvement_notes')
                    })
                else:
                    # Fallback if improvement failed
                    improved_examples.append(original)
            
            return improved_examples
            
        except Exception as e:
            print(f"Error improving chunk: {str(e)}")
            # Return original examples if improvement fails
            return examples
    
    def improve_responses(self, input_file: str, output_file: Optional[str] = None, resume_from: Optional[str] = None) -> List[Dict]:
        """
        Improve responses from input file
        
        Args:
            input_file: Path to input JSONL file with training examples
            output_file: Optional path to output file (for incremental writing)
            resume_from: Optional path to partially completed file
            
        Returns:
            List of improved training examples
        """
        # Determine resume file
        resume_file = resume_from or output_file
        
        # Load examples
        print(f"Loading examples from {input_file}...")
        examples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        print(f"✓ Loaded {len(examples)} examples")
        
        # Check if resuming
        completed = []
        completed_instructions = set()
        file_mode = 'a' if (resume_file and os.path.exists(resume_file)) else 'w'
        
        if resume_file and os.path.exists(resume_file):
            print(f"Resuming from {resume_file}...")
            with open(resume_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        if example.get('response') and example['response'] != '[Improvement failed]':
                            completed.append(example)
                            completed_instructions.add(example['instruction'])
            print(f"✓ Loaded {len(completed)} completed examples")
        
        # Filter remaining
        remaining = [ex for ex in examples if ex['instruction'] not in completed_instructions]
        
        if not remaining:
            print("✓ All examples already improved!")
            return completed
        
        print(f"Improving {len(remaining)} examples...")
        
        # Process in chunks
        num_chunks = (len(remaining) + self.config.chunk_size - 1) // self.config.chunk_size
        print(f"Processing in {num_chunks} chunks of {self.config.chunk_size}")
        if output_file:
            print(f"Incremental writing: Results will be saved to {output_file} after each chunk")
        if self.config.max_workers > 1:
            print(f"Parallel processing: {self.config.max_workers} workers")
        print("=" * 60)
        
        all_examples = completed.copy()
        
        # Thread-safe file writing lock
        file_lock = Lock() if output_file else None
        
        # Open file for incremental writing
        file_handle = None
        if output_file:
            file_handle = open(output_file, file_mode, encoding='utf-8')
            if file_mode == 'w':
                print(f"✓ Opened {output_file} for writing")
            else:
                print(f"✓ Opened {output_file} for appending (resume mode)")
        
        # Worker function for processing a chunk
        def process_chunk(chunk_data):
            """Process a single chunk and return improved results"""
            chunk_idx, chunk = chunk_data
            chunk_num = (chunk_idx // self.config.chunk_size) + 1
            
            try:
                improved = self.improve_responses_for_chunk(chunk)
                
                # Write incrementally (thread-safe)
                if file_handle and file_lock:
                    with file_lock:
                        for example in improved:
                            # Remove improvement_notes from output JSON if present
                            output_example = {k: v for k, v in example.items() if k != 'improvement_notes'}
                            file_handle.write(json.dumps(output_example, ensure_ascii=False) + '\n')
                            file_handle.flush()
                
                improved_count = sum(1 for imp, orig in zip(improved, chunk) 
                                    if imp.get('response') != orig.get('response'))
                
                return {
                    'chunk_num': chunk_num,
                    'examples': improved,
                    'improved_count': improved_count,
                    'success': True
                }
                
            except Exception as e:
                print(f"✗ Error in chunk {chunk_num}: {str(e)}")
                return {
                    'chunk_num': chunk_num,
                    'examples': chunk,  # Return original on error
                    'improved_count': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Prepare chunk data
        chunk_data_list = [
            (chunk_idx, remaining[chunk_idx:chunk_idx + self.config.chunk_size])
            for chunk_idx in range(0, len(remaining), self.config.chunk_size)
        ]
        
        # Process chunks (sequential or parallel)
        if self.config.max_workers == 1:
            # Sequential processing
            for chunk_data in chunk_data_list:
                chunk_idx, chunk = chunk_data
                chunk_num = (chunk_idx // self.config.chunk_size) + 1
                print(f"\nChunk {chunk_num}/{num_chunks} ({len(chunk)} examples)")
                print(f"Progress: {len(all_examples)}/{len(examples)} total")
                
                result = process_chunk(chunk_data)
                all_examples.extend(result['examples'])
                
                if result['success']:
                    if file_handle:
                        print(f"✓ Wrote {len(result['examples'])} examples to file")
                    print(f"✓ Improved {result['improved_count']}/{len(result['examples'])} responses")
                else:
                    print(f"✗ Chunk {chunk_num} failed: {result.get('error', 'Unknown error')}")
                
                if chunk_num < num_chunks:
                    time.sleep(0.5)
        else:
            # Parallel processing
            print(f"Using {self.config.max_workers} parallel workers")
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(process_chunk, chunk_data): chunk_data
                    for chunk_data in chunk_data_list
                }
                
                for future in as_completed(future_to_chunk):
                    result = future.result()
                    all_examples.extend(result['examples'])
                    completed_count += 1
                    
                    if result['success']:
                        print(f"✓ Chunk {result['chunk_num']}/{num_chunks}: Improved {result['improved_count']}/{len(result['examples'])} responses")
                    else:
                        print(f"✗ Chunk {result['chunk_num']}/{num_chunks} failed: {result.get('error', 'Unknown error')}")
                    
                    print(f"Progress: {len(all_examples)}/{len(examples)} total ({completed_count}/{num_chunks} chunks)")
        
        # Close file handle
        if file_handle:
            file_handle.close()
            print(f"\n✓ Closed output file: {output_file}")
        
        print(f"\n{'='*60}")
        print(f"✓ Total examples: {len(all_examples)}")
        
        improved_total = sum(1 for i, ex in enumerate(all_examples) 
                            if i < len(examples) and ex.get('response') != examples[i].get('response'))
        print(f"✓ Improved responses: {improved_total}/{len(all_examples)}")
        
        return all_examples
    
    def save_results(self, examples: List[Dict], output_file: str, stats_only: bool = False):
        """Save improved examples to JSONL file"""
        if not stats_only:
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    # Remove improvement_notes from output
                    output_example = {k: v for k, v in example.items() if k != 'improvement_notes'}
                    f.write(json.dumps(output_example, ensure_ascii=False) + '\n')
            print(f"\n✓ Training data saved to: {output_file}")
        else:
            if os.path.exists(output_file):
                print(f"\n✓ Training data already saved incrementally to: {output_file}")
        
        # Statistics
        stats = {
            'total_examples': len(examples),
            'with_improvement_notes': sum(1 for e in examples if e.get('improvement_notes')),
            'improved_at': datetime.now().isoformat(),
            'config': {
                'model_id': self.config.model_id,
                'chunk_size': self.config.chunk_size,
                'temperature': self.config.temperature
            }
        }
        
        stats_file = output_file.replace('.jsonl', '_improved_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Statistics saved to: {stats_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Improve training data responses using a different model for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Improve responses from training data
  python langchain_response_improver.py fine-tuning-dataset.jsonl
  
  # Use custom evaluation model
  python langchain_response_improver.py fine-tuning-dataset.jsonl --model-id anthropic.claude-3-7-sonnet-20250219-v1:0
  
  # Process in parallel
  python langchain_response_improver.py fine-tuning-dataset.jsonl --max-workers 3
        """
    )
    
    parser.add_argument("input_file", nargs='?', default=None, type=str, 
                       help="Input JSONL file (optional if provided in YAML)")
    parser.add_argument("--output", "-o", type=str, help="Output filename")
    parser.add_argument("--resume-from", "-r", type=str, help="Resume from partial file")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--model-id", type=str, 
                       default="anthropic.claude-3-5-sonnet-20241022-v2:0",
                       help="Evaluation model ID (can be different/larger than generation model)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=5, 
                       help="Examples per evaluation call")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--config", type=str, 
                       help="Path to YAML config. Values override CLI where provided")
    
    args = parser.parse_args()
    
    # Load YAML config
    yaml_cfg: Dict[str, Any] = {}
    def load_yaml_at(path: str) -> Dict[str, Any]:
        import yaml
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
    
    # Load improver-specific section from YAML
    imp_cfg = yaml_cfg.get('improver', {}) if isinstance(yaml_cfg.get('improver'), dict) else {}
    
    input_file = imp_cfg.get('input_file', args.input_file)
    if not input_file or not os.path.exists(input_file):
        print(f"❌ Error: Input file not found or unspecified. Set 'input_file' in YAML improver section or pass as CLI.")
        return 1
    
    model_id = imp_cfg.get('model_id', args.model_id)
    region = imp_cfg.get('region', args.region)
    system_prompt = imp_cfg.get('system_prompt', None)
    evaluation_criteria = imp_cfg.get('evaluation_criteria', None)
    chunk_size = int(imp_cfg.get('chunk_size', args.chunk_size))
    temperature = float(imp_cfg.get('temperature', args.temperature))
    max_tokens = int(imp_cfg.get('max_tokens', args.max_tokens))
    timeout = int(imp_cfg.get('timeout', args.timeout))
    max_workers = int(imp_cfg.get('max_workers', args.max_workers))
    fix_only = bool(imp_cfg.get('fix_only', False))
    
    config = ResponseImproverConfig(
        model_id=model_id,
        region=region,
        system_prompt=system_prompt,
        evaluation_criteria=evaluation_criteria,
        chunk_size=chunk_size,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_workers=max_workers,
        fix_only=fix_only
    )
    
    output_file_override = imp_cfg.get('output')
    output_file = output_file_override if output_file_override else args.output
    if not output_file:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_improved.jsonl"
    
    print("Response Improvement Agent (LangChain)")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Evaluation Model: {config.model_id}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"Timeout: {config.timeout}s")
    print(f"Max workers: {config.max_workers}")
    print("=" * 60)
    
    try:
        agent = ResponseImprovementAgent(config)
        resume_from = imp_cfg.get('resume_from', args.resume_from)
        
        # Improve responses (with incremental writing)
        examples = agent.improve_responses(input_file, output_file=output_file, resume_from=resume_from)
        
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

