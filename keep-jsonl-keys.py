#!/usr/bin/env python3
"""
Keep only specified keys in JSONL dataset
"""
import json
import sys
import os

def keep_only_keys(input_file, keys_to_keep, output_file=None):
    """
    Keep only specified keys in JSONL file, removing all others
    
    Args:
        input_file: Path to input JSONL file
        keys_to_keep: List of keys to keep
        output_file: Path to output JSONL file (optional, defaults to stdout)
    """
    total_lines = 0
    keys_found = {key: 0 for key in keys_to_keep}
    
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
                        total_lines += 1
                        
                        # Keep only specified keys
                        filtered_data = {}
                        for key in keys_to_keep:
                            if key in data:
                                filtered_data[key] = data[key]
                                keys_found[key] += 1
                        
                        # Write to output (even if empty, to preserve line count)
                        json.dump(filtered_data, out_handle)
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
                    total_lines += 1
                    
                    # Keep only specified keys
                    filtered_data = {}
                    for key in keys_to_keep:
                        if key in data:
                            filtered_data[key] = data[key]
                            keys_found[key] += 1
                    
                    # Write to output (even if empty, to preserve line count)
                    json.dump(filtered_data, sys.stdout)
                    sys.stdout.write('\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                    continue
    
    # Print summary
    if output_file:
        print(f"\nProcessed {total_lines} records from {input_file}")
        print(f"Output saved to {output_file}")
        print("\nKeys retained:")
        for key, count in keys_found.items():
            print(f"  '{key}': found in {count} records")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python keep_jsonl_keys.py <input.jsonl> <key1> [key2 key3 ...] [-o output.jsonl]")
        print("\nExamples:")
        print("  python keep_jsonl_keys.py data.jsonl instruction context response -o filtered.jsonl")
        print("  python keep_jsonl_keys.py data.jsonl context response > output.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Parse arguments
    output_file = None
    keys_to_keep = []
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        else:
            keys_to_keep.append(sys.argv[i])
            i += 1
    
    if not keys_to_keep:
        print("Error: No keys specified to keep")
        sys.exit(1)
    
    keep_only_keys(input_file, keys_to_keep, output_file)
