#!/usr/bin/env python3
"""
Script to convert full_synthetic_counseling_data.json to Interview_Data format (CSV).
Converts JSON format to CSV with instruction, input (patient_message), and output (doctor_message).
"""

import json
import pandas as pd
import os
from typing import List, Dict

def load_json_data(json_file: str) -> List[Dict]:
    """Load the synthetic counseling data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_interview_format(json_data: List[Dict]) -> pd.DataFrame:
    """Convert JSON data to Interview_Data CSV format."""
    
    # Standard instruction used in Interview_Data files
    standard_instruction = """You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. 
The assistant gives helpful, comprehensive, and appropriate answers to the user's questions. """
    
    converted_data = []
    
    print(f"Converting {len(json_data)} records...")
    
    for i, record in enumerate(json_data):
        try:
            # Extract required fields
            patient_message = record.get('patient_message', '')
            doctor_message = record.get('doctor_message', '')
            nationality = record.get('nationality', 'Unknown')
            topic = record.get('topic', 'Unknown')
            
            # Skip records with empty messages
            if not patient_message or not doctor_message:
                print(f"Skipping record {i+1}: Empty patient or doctor message")
                continue
            
            # Create the converted record
            converted_record = {
                'instruction': standard_instruction,
                'input': patient_message,
                'output': doctor_message,
                'nationality': nationality,  # Keep for reference
                'topic': topic  # Keep for reference
            }
            
            converted_data.append(converted_record)
            
            # Progress update
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(json_data)} records")
                
        except Exception as e:
            print(f"Error processing record {i+1}: {str(e)}")
            continue
    
    print(f"Successfully converted {len(converted_data)} records")
    
    # Create DataFrame
    df = pd.DataFrame(converted_data)
    return df

def save_by_nationality(df: pd.DataFrame, output_dir: str):
    """Save data separated by nationality, similar to the original Interview_Data files."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group by nationality
    nationality_groups = df.groupby('nationality')
    
    for nationality, group in nationality_groups:
        # Clean nationality name for filename
        nationality_clean = nationality.replace(' ', '_').lower()
        
        # Create output filename
        output_file = os.path.join(output_dir, f"Interview_Data_synthetic_{nationality_clean}.csv")
        
        # Select only the required columns for CSV output
        output_df = group[['instruction', 'input', 'output']].copy()
        
        # Save to CSV
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Saved {len(output_df)} {nationality} records to: {output_file}")

def save_combined_format(df: pd.DataFrame, output_file: str):
    """Save all data in combined format similar to Interview_Data_1K_cantonese.csv."""
    
    # Select only the required columns
    output_df = df[['instruction', 'input', 'output']].copy()
    
    # Save to CSV
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Saved {len(output_df)} total records to: {output_file}")

def print_statistics(df: pd.DataFrame):
    """Print conversion statistics."""
    print("\n=== Conversion Statistics ===")
    print(f"Total records: {len(df)}")
    
    # Nationality distribution
    if 'nationality' in df.columns:
        print("\nNationality distribution:")
        nationality_counts = df['nationality'].value_counts()
        for nationality, count in nationality_counts.items():
            print(f"  {nationality}: {count:,}")
    
    # Topic distribution (top 10)
    if 'topic' in df.columns:
        print("\nTop 10 topics:")
        topic_counts = df['topic'].value_counts().head(10)
        for topic, count in topic_counts.items():
            print(f"  {topic}: {count}")
    
    # Data quality checks
    print(f"\nData quality:")
    print(f"  Records with input: {df['input'].notna().sum()}")
    print(f"  Records with output: {df['output'].notna().sum()}")
    print(f"  Average input length: {df['input'].str.len().mean():.0f} characters")
    print(f"  Average output length: {df['output'].str.len().mean():.0f} characters")

def main():
    """Main function to convert JSON to CSV format."""
    
    # Configuration
    json_file = "/home/taitran/mq/mq-nlp-group/2_1_Synthetic_data/full_synthetic_counseling_data.json"
    output_dir = "/home/taitran/mq/mq-nlp-group/2_1_Synthetic_data/converted_interview_data"
    combined_output = "/home/taitran/mq/mq-nlp-group/2_1_Synthetic_data/synthetic_interview_data_combined.csv"
    
    print("JSON to Interview_Data CSV Converter")
    print("=" * 50)
    print(f"Input file: {json_file}")
    print(f"Output directory: {output_dir}")
    print(f"Combined output: {combined_output}")
    print()
    
    # Check if input file exists
    if not os.path.exists(json_file):
        print(f"Error: Input file not found: {json_file}")
        return
    
    try:
        # Load JSON data
        print("Loading JSON data...")
        json_data = load_json_data(json_file)
        print(f"Loaded {len(json_data)} records from JSON")
        
        # Convert to Interview_Data format
        print("\nConverting to Interview_Data format...")
        df = convert_to_interview_format(json_data)
        
        # Print statistics
        print_statistics(df)
        
        # Save data in different formats
        print(f"\nSaving data...")
        
        # Save combined format
        save_combined_format(df, combined_output)
        
        # Save separated by nationality
        save_by_nationality(df, output_dir)
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"✓ Combined file: {combined_output}")
        print(f"✓ Separated files in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return

if __name__ == "__main__":
    main()
