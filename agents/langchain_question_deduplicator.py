#!/usr/bin/env python3
"""
Question Deduplication Agent using LangChain
Removes semantically similar questions using embeddings.
"""

import json
import argparse
from typing import Any, Dict
import numpy as np
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import os

from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication"""
    similarity_threshold: float = 0.85
    model_id: str = "amazon.titan-embed-text-v2:0"
    region: str = "us-east-1"
    clustering_method: str = "threshold"
    eps: float = 0.15


class QuestionDeduplicationAgent:
    """LangChain-based agent for question deduplication"""
    
    def __init__(self, config: DeduplicationConfig):
        """
        Initialize the deduplication agent
        
        Args:
            config: Configuration for deduplication
        """
        self.config = config
        
        # Initialize LangChain embeddings
        self.embeddings = BedrockEmbeddings(
            model_id=config.model_id,
            region_name=config.region
        )
    
    def _embed_questions(self, questions: List[str]) -> np.ndarray:
        """
        Embed questions using LangChain
        
        Args:
            questions: List of question texts
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Generating embeddings for {len(questions)} questions...")
        
        # Use LangChain's batch embedding
        embeddings = self.embeddings.embed_documents(questions)
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        return np.array(embeddings)
    
    def _find_duplicates_threshold(self, questions: List[Dict], 
                                   embeddings: np.ndarray) -> Tuple[List[int], List[Set[int]]]:
        """
        Find duplicates using similarity threshold
        
        Args:
            questions: List of question dictionaries
            embeddings: Embedding vectors
            
        Returns:
            Tuple of (indices to keep, duplicate groups)
        """
        n = len(questions)
        to_remove = set()
        duplicate_groups = []
        
        print(f"\nFinding duplicates (threshold: {self.config.similarity_threshold})...")
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        for i in range(n):
            if i in to_remove:
                continue
            
            similar_indices = []
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                
                similarity = similarity_matrix[i][j]
                if similarity >= self.config.similarity_threshold:
                    similar_indices.append(j)
                    to_remove.add(j)
            
            if similar_indices:
                duplicate_groups.append({i, *similar_indices})
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n} questions...", end='\r')
        
        print(f"\n✓ Found {len(duplicate_groups)} groups of similar questions")
        
        to_keep = [i for i in range(n) if i not in to_remove]
        return to_keep, duplicate_groups
    
    def _find_duplicates_clustering(self, questions: List[Dict], 
                                   embeddings: np.ndarray) -> Tuple[List[int], List[Set[int]]]:
        """
        Find duplicates using DBSCAN clustering
        
        Args:
            questions: List of question dictionaries
            embeddings: Embedding vectors
            
        Returns:
            Tuple of (indices to keep, duplicate groups)
        """
        print(f"\nClustering questions (eps: {self.config.eps})...")
        
        clustering = DBSCAN(eps=self.config.eps, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        print(f"✓ Found {len(clusters)} clusters")
        
        to_keep = []
        duplicate_groups = []
        
        # Add noise points (unique questions)
        to_keep.extend([i for i, label in enumerate(labels) if label == -1])
        
        # Keep one from each cluster
        for cluster_indices in clusters.values():
            if len(cluster_indices) > 1:
                to_keep.append(cluster_indices[0])
                duplicate_groups.append(set(cluster_indices))
        
        return sorted(to_keep), duplicate_groups
    
    def deduplicate_questions(self, input_file: str) -> Tuple[List[Dict], Dict]:
        """
        Remove semantically similar questions
        
        Args:
            input_file: Path to input JSONL file
            
        Returns:
            Tuple of (deduplicated questions, statistics)
        """
        # Load questions
        print(f"Loading questions from {input_file}...")
        questions = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        
        print(f"✓ Loaded {len(questions)} questions")
        
        if len(questions) == 0:
            return [], {'error': 'No questions found'}
        
        # Extract question texts
        texts = [q['instruction'] for q in questions]
        
        # Generate embeddings using LangChain
        embeddings = self._embed_questions(texts)
        
        # Find duplicates
        if self.config.clustering_method == "clustering":
            to_keep, duplicate_groups = self._find_duplicates_clustering(questions, embeddings)
        else:
            to_keep, duplicate_groups = self._find_duplicates_threshold(questions, embeddings)
        
        # Create deduplicated dataset
        deduplicated = [questions[i] for i in to_keep]
        
        # Statistics
        stats = {
            'original_count': len(questions),
            'deduplicated_count': len(deduplicated),
            'removed_count': len(questions) - len(deduplicated),
            'duplicate_groups': len(duplicate_groups),
            'removal_rate': (len(questions) - len(deduplicated)) / len(questions) * 100,
            'method': self.config.clustering_method,
            'similarity_threshold': self.config.similarity_threshold,
            'processed_at': datetime.now().isoformat()
        }
        
        # Sample duplicates
        stats['sample_duplicates'] = []
        for group in duplicate_groups[:5]:
            group_questions = [questions[i]['instruction'] for i in group]
            stats['sample_duplicates'].append(group_questions)
        
        return deduplicated, stats
    
    def save_results(self, questions: List[Dict], stats: Dict, 
                    output_file: str, stats_file: str = None):
        """Save deduplicated questions and statistics"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for question in questions:
                f.write(json.dumps(question, ensure_ascii=False) + '\n')
        
        print(f"\n✓ Deduplicated questions saved to: {output_file}")
        
        if not stats_file:
            stats_file = output_file.replace('.jsonl', '_dedup_stats.json')
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Statistics saved to: {stats_file}")
        
        print(f"\n{'='*60}")
        print(f"Deduplication Summary:")
        print(f"  Original: {stats['original_count']} questions")
        print(f"  Removed: {stats['removed_count']} duplicates ({stats['removal_rate']:.1f}%)")
        print(f"  Final: {stats['deduplicated_count']} unique questions")
        print(f"{'='*60}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Deduplicate questions using LangChain and embeddings"
    )
    
    parser.add_argument("input_file", nargs='?', default=None, type=str, help="Input JSONL file (optional if provided in YAML)")
    parser.add_argument("--threshold", "-t", type=float, default=0.85)
    parser.add_argument("--method", "-m", choices=["threshold", "clustering"], default="threshold")
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--output", "-o", type=str, help="Output filename")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--model-id", type=str, default="amazon.titan-embed-text-v2:0")
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

    # Load dedup-specific section from YAML
    d_cfg = yaml_cfg.get('dedup', {}) if isinstance(yaml_cfg.get('dedup'), dict) else {}
    
    input_file = d_cfg.get('input_file', args.input_file)
    if not input_file or not os.path.exists(input_file):
        print(f"❌ Error: Input file not found or unspecified. Set 'input_file' in YAML dedup section or pass as CLI.")
        return 1
    
    threshold = float(d_cfg.get('threshold', args.threshold))
    model_id = d_cfg.get('model_id', args.model_id)
    region = d_cfg.get('region', args.region)
    method = d_cfg.get('method', args.method)
    eps = float(d_cfg.get('eps', args.eps))

    config = DeduplicationConfig(
        similarity_threshold=threshold,
        model_id=model_id,
        region=region,
        clustering_method=method,
        eps=eps
    )
    
    output_file_override = d_cfg.get('output')
    output_file = output_file_override if output_file_override else args.output
    if not output_file:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_deduplicated.jsonl"
    
    print("Question Deduplication Agent (LangChain)")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Method: {config.clustering_method}")
    print("=" * 60)
    
    try:
        agent = QuestionDeduplicationAgent(config)
        deduplicated, stats = agent.deduplicate_questions(input_file)
        agent.save_results(deduplicated, stats, output_file)
        print(f"\n✅ Success!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
