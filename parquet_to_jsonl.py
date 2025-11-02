#!/usr/bin/env python3
"""
Convert Parquet file to JSONL format using DuckDB
"""
import duckdb
import sys
import os
import json

def parquet_to_jsonl(parquet_file, output_file=None):
    """
    Read a Parquet file and output as JSONL
    
    Args:
        parquet_file: Path to input Parquet file
        output_file: Path to output JSONL file (optional, defaults to stdout)
    """
    # Validate input file path (basic security check)
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Input file '{parquet_file}' not found")
    
    # Normalize paths to prevent directory traversal
    parquet_file = os.path.abspath(os.path.normpath(parquet_file))
    
    # Connect to DuckDB (in-memory)
    con = duckdb.connect(':memory:')
    
    try:
        # Validate file path - ensure it's not a directory traversal attempt
        # Normalize the path to prevent directory traversal attacks
        real_path = os.path.realpath(parquet_file)
        
        # Ensure the real path matches the normalized path (security check)
        if real_path != os.path.abspath(os.path.normpath(real_path)):
            raise ValueError(f"Invalid file path: potential directory traversal detected")
        
        # Read the Parquet file - escape single quotes for SQL safety
        # Replace single quotes with two single quotes (SQL escaping)
        escaped_path = real_path.replace("'", "''")
        
        # Execute query - we've validated and escaped the path
        result = con.execute(f"SELECT * FROM read_parquet('{escaped_path}')")
        
        # Get column names
        columns = [desc[0] for desc in result.description]
        
        # Open output file or use stdout
        if output_file:
            # Validate and normalize output path
            output_file = os.path.abspath(os.path.normpath(output_file))
            out_handle = open(output_file, 'w')
        else:
            out_handle = sys.stdout
        
        try:
            # Write each row as a JSON line
            for row in result.fetchall():
                record = dict(zip(columns, row))
                json.dump(record, out_handle)
                out_handle.write('\n')
        finally:
            if output_file:
                out_handle.close()
    
    finally:
        con.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parquet_to_jsonl.py <input.parquet> [output.jsonl]")
        print("If output file is not specified, writes to stdout")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(parquet_file):
        print(f"Error: File '{parquet_file}' not found")
        sys.exit(1)
    
    parquet_to_jsonl(parquet_file, output_file)
    
    if output_file:
        print(f"Successfully converted {parquet_file} to {output_file}")
