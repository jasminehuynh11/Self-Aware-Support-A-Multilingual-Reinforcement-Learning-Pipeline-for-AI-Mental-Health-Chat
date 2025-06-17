#!/usr/bin/env python3
"""
Script to combine 3 CSV files into one using the column standard of Interview_Data_6K.csv.
Combines: Interview_Data_6K.csv, combined_interview_data.csv, and synthetic_interview_data_combined.csv
"""

import pandas as pd
import os
from typing import List, Dict

def load_and_standardize_csv(file_path: str, file_name: str) -> pd.DataFrame:
    """Load CSV file and standardize to Interview_Data_6K format."""
    
    print(f"Processing {file_name}...")
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        print(f"  Loaded {len(df):,} rows")
        print(f"  Original columns: {list(df.columns)}")
        
        # Standardize to the required format: instruction, input, output
        standardized_df = pd.DataFrame()
        
        # All files should have these columns, but check and map appropriately
        if 'instruction' in df.columns and 'input' in df.columns and 'output' in df.columns:
            standardized_df = df[['instruction', 'input', 'output']].copy()
        else:
            print(f"  Error: Required columns not found in {file_name}")
            return pd.DataFrame()
        
        # Remove any rows with missing values in critical columns
        initial_count = len(standardized_df)
        standardized_df = standardized_df.dropna(subset=['instruction', 'input', 'output'])
        removed_count = initial_count - len(standardized_df)
        
        if removed_count > 0:
            print(f"  Removed {removed_count} rows with missing values")
        
        # Add source information for tracking
        standardized_df['source_file'] = file_name
        
        print(f"  Standardized: {len(standardized_df):,} rows")
        
        return standardized_df
        
    except Exception as e:
        print(f"  Error loading {file_name}: {str(e)}")
        return pd.DataFrame()

def combine_csv_files():
    """Combine the 3 CSV files into one standardized format."""
    
    print("Combining 3 CSV Files into Interview_Data_6K Standard Format")
    print("=" * 70)
    
    # Define the files to combine
    files_to_combine = [
        {
            'path': '/home/taitran/mq/mq-nlp-group/utils/Interview_Data_6K.csv',
            'name': 'Interview_Data_6K.csv'
        },
        {
            'path': '/home/taitran/mq/mq-nlp-group/utils/translated_data/combined_interview_data.csv',
            'name': 'combined_interview_data.csv'
        },
        {
            'path': '/home/taitran/mq/mq-nlp-group/2_1_Synthetic_data/synthetic_interview_data_combined.csv',
            'name': 'synthetic_interview_data_combined.csv'
        }
    ]
    
    # Process each file
    combined_dataframes = []
    total_files_processed = 0
    
    for file_info in files_to_combine:
        df = load_and_standardize_csv(file_info['path'], file_info['name'])
        if not df.empty:
            combined_dataframes.append(df)
            total_files_processed += 1
        print()
    
    if not combined_dataframes:
        print("❌ No data was successfully loaded from any files!")
        return
    
    # Combine all dataframes
    print("Combining all data...")
    final_df = pd.concat(combined_dataframes, ignore_index=True)
    
    # Remove the source_file column for final output (keeping standard format)
    source_tracking = final_df['source_file'].value_counts()
    final_df = final_df[['instruction', 'input', 'output']].copy()
    
    # Save combined file
    output_file = '/home/taitran/mq/mq-nlp-group/utils/combined_3_datasets.csv'
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Print statistics
    print("=" * 70)
    print("COMBINATION RESULTS")
    print("=" * 70)
    print(f"✓ Files successfully processed: {total_files_processed}/3")
    print(f"✓ Total combined records: {len(final_df):,}")
    print(f"✓ Output file: {output_file}")
    
    print(f"\nSource breakdown:")
    for source, count in source_tracking.items():
        print(f"  {source}: {count:,} records")
    
    # Data quality check
    print(f"\nData quality:")
    print(f"  Records with instruction: {final_df['instruction'].notna().sum():,}")
    print(f"  Records with input: {final_df['input'].notna().sum():,}")
    print(f"  Records with output: {final_df['output'].notna().sum():,}")
    print(f"  Average input length: {final_df['input'].str.len().mean():.0f} characters")
    print(f"  Average output length: {final_df['output'].str.len().mean():.0f} characters")
    
    # File size
    file_size = os.path.getsize(output_file)
    print(f"  File size: {file_size:,} bytes ({file_size / (1024*1024):.1f} MB)")
    
    print(f"\n🎉 Successfully combined 3 datasets!")
    print(f"Final file follows Interview_Data_6K.csv column standard: instruction, input, output")

def verify_output_format():
    """Verify the output format matches Interview_Data_6K.csv standard."""
    
    print("\n" + "=" * 70)
    print("FORMAT VERIFICATION")
    print("=" * 70)
    
    original_file = '/home/taitran/mq/mq-nlp-group/utils/Interview_Data_6K.csv'
    combined_file = '/home/taitran/mq/mq-nlp-group/utils/combined_3_datasets.csv'
    
    if not os.path.exists(combined_file):
        print("❌ Combined file not found for verification")
        return
    
    try:
        # Load both files
        original_df = pd.read_csv(original_file)
        combined_df = pd.read_csv(combined_file)
        
        # Check column structure
        original_cols = list(original_df.columns)
        combined_cols = list(combined_df.columns)
        
        print(f"Original file columns: {original_cols}")
        print(f"Combined file columns: {combined_cols}")
        
        if original_cols == combined_cols:
            print("✅ Column structure matches perfectly!")
        else:
            print("❌ Column structure mismatch")
            return
        
        # Check instruction format
        original_instruction = original_df['instruction'].iloc[0]
        combined_instruction = combined_df['instruction'].iloc[0]
        
        if original_instruction.strip() == combined_instruction.strip():
            print("✅ Instruction format matches!")
        else:
            print("⚠️  Instruction format differs (may be normal if from different sources)")
        
        # Sample data preview
        print(f"\nSample from combined file:")
        print(f"Input: {combined_df['input'].iloc[0][:100]}...")
        print(f"Output: {combined_df['output'].iloc[0][:100]}...")
        
        print(f"\n✅ Format verification completed successfully!")
        
    except Exception as e:
        print(f"❌ Verification error: {str(e)}")

def main():
    """Main function to combine CSV files."""
    try:
        combine_csv_files()
        verify_output_format()
    except Exception as e:
        print(f"❌ Error during combination: {str(e)}")

if __name__ == "__main__":
    main()
