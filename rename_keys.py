#!/usr/bin/env python3
"""
Rename keys in JSONL dataset: 'output' -> 'response' and 'input' -> 'context'
"""
import json
import sys
import os

def rename_keys(input_file, output_file=None):
    """
    Rename keys in JSONL file
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (optional, defaults to stdout)
    """
    # Open output file or use stdout
    if output_file:
        out_handle = open(output_file, 'w')
        try:
            with open(input_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON
                        data = json.loads(line)
                        
                        # Rename keys
                        if 'output' in data:
                            data['response'] = data.pop('output')
                        
                        if 'input' in data:
                            data['context'] = data.pop('input')
                        
                        # Write to output
                        json.dump(data, out_handle)
                        out_handle.write('\n')
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                        continue
        finally:
            out_handle.close()
    else:
        # Use stdout - no need to close
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON
                    data = json.loads(line)
                    
                    # Rename keys
                    if 'output' in data:
                        data['response'] = data.pop('output')
                    
                    if 'input' in data:
                        data['context'] = data.pop('input')
                    
                    # Write to output
                    json.dump(data, sys.stdout)
                    sys.stdout.write('\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                    continue

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rename_jsonl_keys.py <input.jsonl> [output.jsonl]")
        print("Renames: 'output' -> 'response' and 'input' -> 'context'")
        print("If output file is not specified, writes to stdout")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    rename_keys(input_file, output_file)
    
    if output_file:
        print(f"Successfully processed {input_file} -> {output_file}")