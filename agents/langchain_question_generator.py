#!/usr/bin/env python3
"""
Question Generation Agent using LangChain
Generates diverse questions while avoiding duplicates.
"""

import json
import argparse
from typing import Any
from typing import List, Set, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import os

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field


# Pydantic models for structured output
class Question(BaseModel):
    """A generated question"""
    instruction: str = Field(description="The question text")
    category: Optional[str] = Field(default=None, description="Optional category")


class QuestionBatch(BaseModel):
    """Batch of questions"""
    questions: List[Question] = Field(description="List of generated questions")


@dataclass
class QuestionGenConfig:
    """Configuration for question generation"""
    topic: str
    num_questions: int = 100
    batch_size: int = 20
    model_id: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    region: str = "us-east-1"
    system_prompt: Optional[str] = None
    temperature: float = 0.9
    categories: Optional[List[str]] = None


class QuestionGenerationAgent:
    """LangChain-based agent for question generation"""
    
    def __init__(self, config: QuestionGenConfig):
        """
        Initialize the question generation agent
        
        Args:
            config: Configuration for question generation
        """
        self.config = config
        self.generated_questions: Set[str] = set()
        self.all_questions: List[Question] = []
        
        # Initialize LangChain components
        self.llm = ChatBedrock(
            model_id=config.model_id,
            region_name=config.region,
            model_kwargs={
                "temperature": config.temperature,
                "max_tokens": 4096
            }
        )
        
        # Setup output parser
        self.parser = JsonOutputParser(pydantic_object=QuestionBatch)
        
        # Create prompt template
        self.prompt = self._create_prompt_template()
        
        # Create the chain
        self.chain = self._build_chain()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for question generation"""
        
        system_message = self.config.system_prompt or "You are an expert at generating diverse, high-quality questions."
        
        user_template = """Generate {batch_size} unique, diverse, and realistic questions about: {topic}

{exclusion_context}

{category_guidance}

Requirements:
1. Each question must be completely unique and different from others
2. Vary question types: how-to, what-is, troubleshooting, comparison, best practices, etc.
3. Vary complexity levels: beginner, intermediate, advanced
4. Include practical, real-world scenarios
5. Make questions specific and detailed
6. This is batch {batch_num}, create very different variations from previous batches

Focus on creating maximum diversity in: phrasing, angle, complexity, and topic coverage.

{format_instructions}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_template)
        ])
    
    def _build_chain(self):
        """Build the LangChain chain"""
        return (
            {
                "topic": RunnablePassthrough(),
                "batch_size": RunnablePassthrough(),
                "batch_num": RunnablePassthrough(),
                "exclusion_context": RunnablePassthrough(),
                "category_guidance": RunnablePassthrough(),
                "format_instructions": lambda _: self.parser.get_format_instructions()
            }
            | self.prompt
            | self.llm
            | self.parser
        )
    
    def _create_exclusion_context(self) -> str:
        """Create context about questions to exclude"""
        if not self.generated_questions:
            return ""
        
        recent_questions = list(self.generated_questions)[-50:]
        return f"""IMPORTANT - AVOID THESE ALREADY GENERATED QUESTIONS:
{chr(10).join(f"- {q}" for q in recent_questions)}

DO NOT generate questions that are similar or variations of the above."""
    
    def _create_category_guidance(self) -> str:
        """Create guidance about categories"""
        if not self.config.categories:
            return ""
        
        return f"""Consider these categories/aspects (vary across batches):
{chr(10).join(f"- {cat}" for cat in self.config.categories)}"""
    
    def _is_duplicate(self, question: str) -> bool:
        """Check if question is a duplicate"""
        normalized = question.lower().strip().rstrip('?').rstrip('.')
        
        if normalized in self.generated_questions:
            return True
        
        # Simple similarity check
        for existing in self.generated_questions:
            existing_normalized = existing.lower().strip().rstrip('?').rstrip('.')
            
            if len(normalized) > 20 and len(existing_normalized) > 20:
                words_new = set(normalized.split())
                words_existing = set(existing_normalized.split())
                
                if len(words_new) > 0:
                    overlap = len(words_new & words_existing) / len(words_new)
                    if overlap > 0.85:
                        return True
        
        return False
    
    def generate_batch(self, batch_num: int) -> List[Question]:
        """
        Generate a batch of questions
        
        Args:
            batch_num: Current batch number
            
        Returns:
            List of Question objects
        """
        try:
            # Prepare input
            chain_input = {
                "topic": self.config.topic,
                "batch_size": self.config.batch_size,
                "batch_num": batch_num,
                "exclusion_context": self._create_exclusion_context(),
                "category_guidance": self._create_category_guidance()
            }
            
            # Invoke chain
            result = self.chain.invoke(chain_input)
            
            # Parse results
            if isinstance(result, dict) and 'questions' in result:
                questions = [Question(**q) if isinstance(q, dict) else q 
                           for q in result['questions']]
            elif isinstance(result, list):
                questions = [Question(**q) if isinstance(q, dict) else q 
                           for q in result]
            else:
                questions = []
            
            return questions
            
        except Exception as e:
            print(f"Error generating batch: {str(e)}")
            return []
    
    def generate_questions(self, resume_from: Optional[str] = None) -> List[Question]:
        """
        Generate all questions with duplicate avoidance
        
        Args:
            resume_from: Optional path to existing questions file
            
        Returns:
            List of Question objects
        """
        # Load existing if resuming
        if resume_from and os.path.exists(resume_from):
            print(f"Loading existing questions from {resume_from}...")
            with open(resume_from, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        question = Question(
                            instruction=data['instruction'],
                            category=data.get('category')
                        )
                        self.all_questions.append(question)
                        self.generated_questions.add(question.instruction.lower().strip())
            print(f"Loaded {len(self.all_questions)} existing questions")
        
        target_count = self.config.num_questions
        current_count = len(self.all_questions)
        
        if current_count >= target_count:
            print(f"Already have {current_count} questions (target: {target_count})")
            return self.all_questions
        
        needed = target_count - current_count
        num_batches = (needed + self.config.batch_size - 1) // self.config.batch_size
        
        print(f"\nGenerating {needed} more questions in {num_batches} batches...")
        print(f"Topic: {self.config.topic}")
        print(f"Starting from: {current_count} questions")
        print("=" * 60)
        
        batch_num = (current_count // self.config.batch_size) + 1
        
        while len(self.all_questions) < target_count:
            samples_needed = min(self.config.batch_size, target_count - len(self.all_questions))
            
            print(f"\nBatch {batch_num} (Need: {samples_needed}, Total: {len(self.all_questions)}/{target_count})")
            
            batch_questions = self.generate_batch(batch_num)
            
            # Filter duplicates
            unique_questions = []
            duplicates = 0
            
            for q in batch_questions:
                if not self._is_duplicate(q.instruction):
                    unique_questions.append(q)
                    self.generated_questions.add(q.instruction.lower().strip())
                    self.all_questions.append(q)
                else:
                    duplicates += 1
            
            print(f"✓ Generated {len(unique_questions)} unique questions")
            if duplicates > 0:
                print(f"  Filtered {duplicates} duplicates")
            
            batch_num += 1
            
            # Safety limit
            if batch_num > num_batches * 3:
                print("\n⚠ Reached maximum batch attempts")
                break
        
        print(f"\n{'='*60}")
        print(f"✓ Total questions generated: {len(self.all_questions)}")
        
        return self.all_questions
    
    def save_questions(self, output_file: str):
        """Save questions to JSONL file"""
        # Don't create empty files
        if not self.all_questions:
            print(f"⚠ No questions to save. Skipping file creation.")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for question in self.all_questions:
                data = {
                    'instruction': question.instruction,
                    'context': '',
                    'response': ''
                }
                if question.category:
                    data['category'] = question.category
                
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"✓ Questions saved to: {output_file}")
        
        # Save statistics
        stats = {
            'total_questions': len(self.all_questions),
            'topic': self.config.topic,
            'generated_at': datetime.now().isoformat(),
            'categories': {}
        }
        
        for q in self.all_questions:
            cat = q.category or 'uncategorized'
            stats['categories'][cat] = stats['categories'].get(cat, 0) + 1
        
        stats_file = output_file.replace('.jsonl', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Statistics saved to: {stats_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Generate diverse questions using LangChain and AWS Bedrock",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("topic", nargs='?', default=None, type=str, help="Topic for question generation (optional if provided in YAML)")
    parser.add_argument("--num-questions", "-n", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=20)
    parser.add_argument("--system-prompt", "-s", type=str, help="Custom system prompt")
    parser.add_argument("--categories", "-c", type=str, help="Comma-separated categories")
    parser.add_argument("--output", "-o", type=str, help="Output filename")
    parser.add_argument("--resume-from", "-r", type=str, help="Resume from existing file")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--model-id", type=str, default="anthropic.claude-3-5-sonnet-20241022-v2:0")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--config", type=str, help="Path to YAML config. Values override CLI where provided")
    
    args = parser.parse_args()

    # Load YAML config (explicit, env, fallback paths)
    yaml_cfg: dict[str, Any] = {}
    def load_yaml_at(path: str) -> dict[str, Any]:
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
    
    # Load question-specific section from YAML
    q_cfg = yaml_cfg.get('question', {}) if isinstance(yaml_cfg.get('question'), dict) else {}
    
    # Parse categories
    # Values from YAML question section (if any)
    topic = q_cfg.get('topic', args.topic)
    num_questions = int(q_cfg.get('num_questions', args.num_questions))
    batch_size = int(q_cfg.get('batch_size', args.batch_size))
    model_id = q_cfg.get('model_id', args.model_id)
    region = q_cfg.get('region', args.region)
    system_prompt = q_cfg.get('system_prompt', args.system_prompt)
    temperature = float(q_cfg.get('temperature', args.temperature))
    output_file_override = q_cfg.get('output')
    resume_from_override = q_cfg.get('resume_from')
    yaml_categories = q_cfg.get('categories')

    categories = None
    raw_categories = yaml_categories if yaml_categories is not None else args.categories
    if isinstance(raw_categories, list):
        categories = [str(c).strip() for c in raw_categories]
    elif isinstance(raw_categories, str):
        categories = [c.strip() for c in raw_categories.split(',') if c.strip()]
    
    # Create configuration
    config = QuestionGenConfig(
        topic=topic,
        num_questions=num_questions,
        batch_size=batch_size,
        model_id=model_id,
        region=region,
        system_prompt=system_prompt,
        temperature=temperature,
        categories=categories
    )

    if not config.topic:
        print("❌ Missing 'topic'. Provide it in YAML (topic) or as CLI positional arg.")
        return 1
    
    # Generate output filename
    output_file = output_file_override if output_file_override else args.output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = topic.replace(" ", "_").replace("/", "_")[:30]
        output_file = f"questions_{safe_topic}_{timestamp}.jsonl"
    
    print("Question Generation Agent (LangChain)")
    print("=" * 60)
    print(f"Topic: {config.topic}")
    print(f"Target: {config.num_questions} questions")
    print(f"Batch size: {config.batch_size}")
    print(f"Model: {config.model_id}")
    print(f"Temperature: {config.temperature}")
    if config.system_prompt:
        print(f"System prompt: {config.system_prompt[:100]}...")
    if categories:
        print(f"Categories: {', '.join(categories)}")
    resume_from = resume_from_override if resume_from_override else args.resume_from
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 60)
    
    try:
        agent = QuestionGenerationAgent(config)
        questions = agent.generate_questions(resume_from=resume_from)
        agent.save_questions(output_file)
        print(f"\n✅ Success! Generated {len(questions)} unique questions")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
