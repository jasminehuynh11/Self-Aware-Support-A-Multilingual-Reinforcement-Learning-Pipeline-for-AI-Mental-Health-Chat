#!/usr/bin/env python3
"""
Script to convert VLLM inference results to simplified format
Extracts question_id, category, turns, answer, and answer_no_tag
Removes content within <think> and <evaluate> tags from answers
"""

import json
import re
import argparse
import os
from typing import Dict, Any, Optional

def remove_tags_from_text(text: str) -> str:
    """
    Remove content within <think> and <evaluate> tags from text
    
    Args:
        text: Input text that may contain tags
        
    Returns:
        Text with tag content removed
    """
    if not text:
        return text
    
    # Remove <think>...</think> content (including empty ones)
    think_pattern = r'<think>.*?</think>'
    text = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove <evaluate>...</evaluate> content 
    evaluate_pattern = r'<evaluate>.*?</evaluate>'
    text = re.sub(evaluate_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newlines
    text = text.strip()
    
    return text

def parse_question_data(question_str: str) -> Dict[str, Any]:
    """
    Parse the question string to extract question_id, category, and turns
    
    Args:
        question_str: String representation of question dictionary
        
    Returns:
        Dictionary with parsed data
    """
    try:
        # The question field contains a string representation of a dictionary
        # We need to safely evaluate it
        question_data = eval(question_str)
        
        return {
            'question_id': question_data.get('question_id', None),
            'category': question_data.get('category', 'unknown'),
            'turns': question_data.get('turns', [])
        }
    except Exception as e:
        print(f"Error parsing question data: {e}")
        print(f"Question string: {question_str[:100]}...")
        return {
            'question_id': None,
            'category': 'unknown',
            'turns': []
        }

def convert_jsonl_file(input_file: str, output_file: str) -> None:
    """
    Convert VLLM inference results to simplified format
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    converted_count = 0
    error_count = 0
    
    print(f"Converting {input_file} to {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the input line
                data = json.loads(line)
                
                # Extract original answer
                original_answer = data.get('answer', '')
                
                # Remove tags to create answer_no_tag
                answer_no_tag = remove_tags_from_text(original_answer)
                
                # Parse question data
                question_str = data.get('question', '')
                question_data = parse_question_data(question_str)
                
                # Create output record
                output_record = {
                    'question_id': question_data['question_id'],
                    'category': question_data['category'],
                    'turns': question_data['turns'],
                    'answer': original_answer,
                    'answer_no_tag': answer_no_tag
                }
                
                # Write to output file
                json.dump(output_record, outfile, ensure_ascii=False)
                outfile.write('\n')
                
                converted_count += 1
                
                # Print progress every 50 lines
                if line_num % 50 == 0:
                    print(f"Processed {line_num} lines...")
                    
            except Exception as e:
                error_count += 1
                print(f"Error processing line {line_num}: {e}")
                print(f"Line content: {line[:100]}...")
                continue
    
    print(f"\nConversion completed!")
    print(f"Successfully converted: {converted_count} records")
    print(f"Errors: {error_count} records")
    print(f"Output saved to: {output_file}")

def preview_conversion(input_file: str, num_examples: int = 3) -> None:
    """
    Preview the conversion by showing first few examples
    
    Args:
        input_file: Path to input JSONL file
        num_examples: Number of examples to show
    """
    print(f"\nPreviewing conversion for {input_file} (first {num_examples} examples):")
    print("=" * 80)
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i >= num_examples:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Parse question data
                question_str = data.get('question', '')
                question_data = parse_question_data(question_str)
                
                # Get answers
                original_answer = data.get('answer', '')
                answer_no_tag = remove_tags_from_text(original_answer)
                
                print(f"\nExample {i+1}:")
                print(f"Question ID: {question_data['question_id']}")
                print(f"Category: {question_data['category']}")
                print(f"Turns: {question_data['turns'][:1] if question_data['turns'] else []}...")  # Show first turn only
                print(f"Original answer length: {len(original_answer)} chars")
                print(f"Answer (no tags) length: {len(answer_no_tag)} chars")
                print(f"Answer preview: {answer_no_tag[:150]}...")
                print("-" * 60)
                
            except Exception as e:
                print(f"Error in example {i+1}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert VLLM inference results to simplified format")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("-o", "--output", help="Output JSONL file path (auto-generated if not specified)")
    parser.add_argument("--preview", action="store_true", help="Preview conversion without saving")
    parser.add_argument("--examples", type=int, default=3, help="Number of examples to show in preview (default: 3)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    # Generate output filename if not specified
    if not args.output:
        input_base = os.path.splitext(args.input_file)[0]
        args.output = f"{input_base}_converted.jsonl"
    
    # Preview mode
    if args.preview:
        preview_conversion(args.input_file, args.examples)
        return
    
    # Check if output file exists
    if os.path.exists(args.output):
        response = input(f"Output file '{args.output}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            return
    
    # Perform conversion
    convert_jsonl_file(args.input_file, args.output)
    
    # Show some statistics
    try:
        with open(args.output, 'r', encoding='utf-8') as f:
            output_lines = sum(1 for line in f if line.strip())
        
        print(f"\nOutput file statistics:")
        print(f"Total records: {output_lines}")
        
        # Show first example from output
        print(f"\nFirst converted record:")
        with open(args.output, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                first_record = json.loads(first_line)
                print(json.dumps(first_record, ensure_ascii=False, indent=2))
                
    except Exception as e:
        print(f"Error reading output file: {e}")

if __name__ == "__main__":
    main()
