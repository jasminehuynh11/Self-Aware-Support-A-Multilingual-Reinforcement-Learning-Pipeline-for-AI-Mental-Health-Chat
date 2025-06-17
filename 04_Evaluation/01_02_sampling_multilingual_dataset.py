#!/usr/bin/env python3
"""
Sample 40 records from each of 5 language-specific JSONL files to create
a new 200-record multilingual dataset.
"""

import json
import random
import os
from pathlib import Path


def load_jsonl_file(filepath):
    """Load and return all records from a JSONL file."""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} records from {filepath}")
        return records
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def sample_records(records, n_samples=40, seed=42):
    """Sample n_samples records from the list."""
    if len(records) < n_samples:
        print(f"Warning: Only {len(records)} records available, sampling all of them")
        return records
    
    random.seed(seed)
    return random.sample(records, n_samples)


def save_jsonl_file(records, filepath):
    """Save records to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    print(f"Saved {len(records)} records to {filepath}")


def save_csv_file(records, filepath):
    """Save records to a CSV file with Interview_Data format."""
    import csv
    
    fieldnames = ['question_id', 'category', 'question', 'original_language', 'translated_to']
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in records:
            # Extract the question from the 'turns' array (first turn)
            question = record.get('turns', [''])[0] if record.get('turns') else ''
            
            csv_record = {
                'question_id': record.get('question_id', ''),
                'category': record.get('category', ''),
                'question': question,
                'original_language': record.get('original_language', ''),
                'translated_to': record.get('translated_to', '')
            }
            writer.writerow(csv_record)
    
    print(f"Saved {len(records)} records to CSV: {filepath}")


def main():
    # Define the language-specific JSONL files
    base_dir = Path('/home/taitran/mq/mq-nlp-group/utils')
    
    language_files = {
        'arabic': base_dir / 'question_arabic.jsonl',
        'chinese': base_dir / 'question_chinese.jsonl',
        'cantonese': base_dir / 'question_cantonese.jsonl',
        'vietnamese': base_dir / 'question_vietnamese.jsonl'
    }
    
    # Check if we have the 5th file (English - might be question.jsonl)
    english_file = base_dir / 'question.jsonl'
    if english_file.exists():
        language_files['english'] = english_file
    else:
        print(f"Warning: Could not find English JSONL file at {english_file}")
        # Let's check what files are available
        jsonl_files = list(base_dir.glob('question*.jsonl'))
        print(f"Available JSONL files: {[f.name for f in jsonl_files]}")
        
        # If we don't have exactly 5 files, we'll work with what we have
        if len(language_files) < 5:
            print("Proceeding with 4 language files (160 records total)")
    
    print(f"\nProcessing {len(language_files)} language files:")
    for lang, filepath in language_files.items():
        print(f"  {lang}: {filepath}")
    
    # Sample from each file
    all_sampled_records = []
    
    for language, filepath in language_files.items():
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue
            
        records = load_jsonl_file(filepath)
        if not records:
            continue
            
        sampled = sample_records(records, n_samples=40)
        
        # Add language identifier to each record
        for record in sampled:
            record['sampled_language'] = language
            
        all_sampled_records.extend(sampled)
        print(f"Sampled {len(sampled)} records from {language}")
    
    print(f"\nTotal sampled records: {len(all_sampled_records)}")
    
    # Shuffle the combined dataset
    random.seed(42)
    random.shuffle(all_sampled_records)
    
    # Create output directory
    output_dir = base_dir / 'sampled_multilingual_data'
    output_dir.mkdir(exist_ok=True)
    
    # Save in both JSONL and CSV formats
    output_jsonl = output_dir / 'sampled_multilingual_200.jsonl'
    output_csv = output_dir / 'sampled_multilingual_200.csv'
    
    save_jsonl_file(all_sampled_records, output_jsonl)
    save_csv_file(all_sampled_records, output_csv)
    
    # Generate summary statistics
    language_counts = {}
    category_counts = {}
    
    for record in all_sampled_records:
        lang = record.get('sampled_language', 'unknown')
        category = record.get('category', 'unknown')
        
        language_counts[lang] = language_counts.get(lang, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\n=== SAMPLING SUMMARY ===")
    print(f"Total records: {len(all_sampled_records)}")
    print(f"\nRecords by language:")
    for lang, count in sorted(language_counts.items()):
        print(f"  {lang}: {count}")
    print(f"\nRecords by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    
    print(f"\nOutput files created:")
    print(f"  JSONL: {output_jsonl}")
    print(f"  CSV: {output_csv}")


if __name__ == "__main__":
    main()
