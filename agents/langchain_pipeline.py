#!/usr/bin/env python3
"""
LangChain-based Fine-tuning Dataset Pipeline
Orchestrates the three LangChain agents to create complete datasets.
"""

import argparse
from typing import Any, Dict
import subprocess
import sys
import os
from datetime import datetime
from typing import Optional


class LangChainDatasetPipeline:
    """Orchestrates the LangChain-based dataset generation pipeline"""
    
    def __init__(self, topic: str, num_questions: int, output_dir: str = "output"):
        """
        Initialize the pipeline
        
        Args:
            topic: Topic for question generation
            num_questions: Number of questions to generate
            output_dir: Directory for output files
        """
        self.topic = topic
        self.num_questions = num_questions
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = topic.replace(" ", "_").replace("/", "_")[:30]
        
        self.questions_file = os.path.join(output_dir, f"questions_{safe_topic}_{timestamp}.jsonl")
        self.deduplicated_file = os.path.join(output_dir, f"questions_deduplicated_{safe_topic}_{timestamp}.jsonl")
        self.final_file = os.path.join(output_dir, f"training_data_{safe_topic}_{timestamp}.jsonl")
    
    def run_step(self, step_name: str, command: list) -> bool:
        """Run a pipeline step"""
        print(f"\n{'='*60}")
        print(f"STEP: {step_name}")
        print(f"{'='*60}\n")
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            # Print output if there's any
            if result.stdout:
                print(result.stdout)
            print(f"\n✓ {step_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {step_name} failed with error code {e.returncode}")
            # Print error output for debugging
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            return False
    
    def step_1_generate_questions(self, batch_size: int = 20,
                                  system_prompt: Optional[str] = None,
                                  categories: Optional[str] = None,
                                  temperature: float = 0.9,
                                  region: str = "us-east-1") -> bool:
        """Step 1: Generate questions using LangChain"""
        command = [
            sys.executable,
            "langchain_question_generator.py",
            self.topic,
            "--num-questions", str(self.num_questions),
            "--batch-size", str(batch_size),
            "--temperature", str(temperature),
            "--region", region,
            "--output", self.questions_file
        ]
        
        if system_prompt:
            command.extend(["--system-prompt", system_prompt])
        
        if categories:
            command.extend(["--categories", categories])
        
        return self.run_step("Generate Questions (LangChain)", command)
    
    def step_2_deduplicate(self, threshold: float = 0.85,
                          method: str = "threshold",
                          region: str = "us-east-1") -> bool:
        """Step 2: Deduplicate using LangChain embeddings"""
        command = [
            sys.executable,
            "langchain_question_deduplicator.py",
            self.questions_file,
            "--threshold", str(threshold),
            "--method", method,
            "--region", region,
            "--output", self.deduplicated_file
        ]
        
        return self.run_step("Deduplicate Questions (LangChain)", command)
    
    def step_3_generate_responses(self, chunk_size: int = 5,
                                  system_prompt: Optional[str] = None,
                                  no_context: bool = False,
                                  temperature: float = 0.7,
                                  region: str = "us-east-1") -> bool:
        """Step 3: Generate responses using LangChain"""
        command = [
            sys.executable,
            "langchain_response_generator.py",
            self.deduplicated_file,
            "--chunk-size", str(chunk_size),
            "--temperature", str(temperature),
            "--region", region,
            "--output", self.final_file
        ]
        
        if system_prompt:
            command.extend(["--system-prompt", system_prompt])
        
        if no_context:
            command.append("--no-context")
        
        return self.run_step("Generate Responses (LangChain)", command)
    
    def run_full_pipeline(self, **kwargs) -> bool:
        """Run the complete LangChain pipeline"""
        print("\n" + "="*60)
        print("LANGCHAIN FINE-TUNING DATASET PIPELINE")
        print("="*60)
        print(f"Topic: {self.topic}")
        print(f"Target questions: {self.num_questions}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        
        # Step 1
        if not self.step_1_generate_questions(
            batch_size=kwargs.get('question_batch_size', 20),
            system_prompt=kwargs.get('question_system_prompt'),
            categories=kwargs.get('categories'),
            temperature=kwargs.get('question_temperature', 0.9),
            region=kwargs.get('region', 'us-east-1')
        ):
            return False
        
        # Step 2
        if not self.step_2_deduplicate(
            threshold=kwargs.get('dedup_threshold', 0.85),
            method=kwargs.get('dedup_method', 'threshold'),
            region=kwargs.get('region', 'us-east-1')
        ):
            return False
        
        # Step 3
        if not self.step_3_generate_responses(
            chunk_size=kwargs.get('response_chunk_size', 5),
            system_prompt=kwargs.get('response_system_prompt'),
            no_context=kwargs.get('no_context', False),
            temperature=kwargs.get('response_temperature', 0.7),
            region=kwargs.get('region', 'us-east-1')
        ):
            return False
        
        print("\n" + "="*60)
        print("✅ LANGCHAIN PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nFinal training data: {self.final_file}")
        print(f"\nReady for LLM fine-tuning!")
        print("="*60)
        
        return True


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="LangChain-based pipeline for generating LLM fine-tuning datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python langchain_pipeline.py "customer service" --num-questions 500
  
  # With custom parameters
  python langchain_pipeline.py "legal advice" \\
    --num-questions 1000 \\
    --dedup-threshold 0.90 \\
    --response-chunk-size 10
  
  # With system prompts
  python langchain_pipeline.py "medical information" \\
    --num-questions 300 \\
    --question-system-prompt "Generate patient questions" \\
    --response-system-prompt "Provide clear medical information"
        """
    )
    
    parser.add_argument("topic", nargs='?', default=None, type=str, help="Topic for dataset generation (optional if provided in YAML)")
    parser.add_argument("--num-questions", "-n", type=int, default=100)
    parser.add_argument("--output-dir", "-o", type=str, default="output")
    parser.add_argument("--config", type=str, help="Path to YAML config. Values override CLI where provided")
    
    # Question generation options
    parser.add_argument("--question-batch-size", type=int, default=20)
    parser.add_argument("--question-system-prompt", type=str)
    parser.add_argument("--categories", type=str)
    parser.add_argument("--question-temperature", type=float, default=0.9)
    
    # Deduplication options
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--dedup-method", type=str, choices=["threshold", "clustering"], default="threshold")
    
    # Response generation options
    parser.add_argument("--response-chunk-size", type=int, default=5)
    parser.add_argument("--response-system-prompt", type=str)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--response-temperature", type=float, default=0.7)
    
    # AWS options
    parser.add_argument("--region", type=str, default="us-east-1")
    
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
    
    topic = yaml_cfg.get('topic', args.topic)
    num_questions = int(yaml_cfg.get('num_questions', args.num_questions))
    output_dir = yaml_cfg.get('output_dir', args.output_dir)

    if not topic:
        print("❌ Missing 'topic'. Provide it in YAML (topic) or as CLI positional arg.")
        return 1

    pipeline = LangChainDatasetPipeline(
        topic=topic,
        num_questions=num_questions,
        output_dir=output_dir
    )
    
    q_cfg = (yaml_cfg.get('question') or {}) if isinstance(yaml_cfg.get('question'), dict) else {}
    d_cfg = (yaml_cfg.get('dedup') or {}) if isinstance(yaml_cfg.get('dedup'), dict) else {}
    r_cfg = (yaml_cfg.get('response') or {}) if isinstance(yaml_cfg.get('response'), dict) else {}
    region = yaml_cfg.get('region', args.region)

    success = pipeline.run_full_pipeline(
        question_batch_size=int(q_cfg.get('batch_size', args.question_batch_size)),
        question_system_prompt=q_cfg.get('system_prompt', args.question_system_prompt),
        categories=q_cfg.get('categories', args.categories),
        question_temperature=float(q_cfg.get('temperature', args.question_temperature)),
        dedup_threshold=float(d_cfg.get('threshold', args.dedup_threshold)),
        dedup_method=d_cfg.get('method', args.dedup_method),
        response_chunk_size=int(r_cfg.get('chunk_size', args.response_chunk_size)),
        response_system_prompt=r_cfg.get('system_prompt', args.response_system_prompt),
        no_context=bool(r_cfg.get('no_context', args.no_context)),
        response_temperature=float(r_cfg.get('temperature', args.response_temperature)),
        region=region
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
