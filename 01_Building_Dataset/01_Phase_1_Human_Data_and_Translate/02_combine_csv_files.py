#!/usr/bin/env python3
"""
Script to combine all CSV files in the translated_data folder.
All files should have the same structure: instruction, input, output
"""

import pandas as pd
import os
import glob
from pathlib import Path

def combine_csv_files(source_folder, output_file=None):
    """
    Combine all CSV files in the source folder into one file.
    
    Args:
        source_folder (str): Path to folder containing CSV files
        output_file (str): Output file name (optional)
    
    Returns:
        str: Path to the combined file
    """
    
    # Get all CSV files in the folder
    csv_pattern = os.path.join(source_folder, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {source_folder}")
        return None
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Read and combine all CSV files
    combined_data = []
    total_rows = 0
    
    for csv_file in csv_files:
        print(f"\nProcessing {os.path.basename(csv_file)}...")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Add a source column to track which file each row came from
            df['source_file'] = os.path.basename(csv_file)
            
            # Add language information based on filename
            if 'arabic' in csv_file.lower():
                df['language'] = 'Arabic'
            elif 'cantonese' in csv_file.lower():
                df['language'] = 'Cantonese'
            elif 'mandarin' in csv_file.lower():
                df['language'] = 'Mandarin'
            elif 'vietnamese' in csv_file.lower():
                df['language'] = 'Vietnamese'
            else:
                df['language'] = 'Unknown'
            
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            combined_data.append(df)
            total_rows += len(df)
            
        except Exception as e:
            print(f"  Error reading {csv_file}: {str(e)}")
            continue
    
    if not combined_data:
        print("No data was successfully read from any files!")
        return None
    
    # Combine all dataframes
    print(f"\nCombining all data...")
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = os.path.join(source_folder, "combined_interview_data.csv")
    
    # Save combined data
    try:
        combined_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ Successfully combined {len(csv_files)} files")
        print(f"✓ Total rows: {len(combined_df)}")
        print(f"✓ Output saved to: {output_file}")
        
        # Print summary statistics
        print(f"\nSummary by language:")
        language_counts = combined_df['language'].value_counts()
        for language, count in language_counts.items():
            print(f"  {language}: {count:,} rows")
        
        print(f"\nSummary by source file:")
        source_counts = combined_df['source_file'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} rows")
            
        return output_file
        
    except Exception as e:
        print(f"Error saving combined file: {str(e)}")
        return None

def main():
    """Main function to combine CSV files."""
    
    # Set the source folder
    source_folder = "/home/taitran/mq/mq-nlp-group/utils/translated_data"
    
    print("CSV File Combiner")
    print("=" * 50)
    print(f"Source folder: {source_folder}")
    
    # Check if folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Folder {source_folder} does not exist!")
        return
    
    # Combine files
    output_file = combine_csv_files(source_folder)
    
    if output_file:
        print(f"\n🎉 Success! Combined file created at:")
        print(f"   {output_file}")
        
        # Show file size
        file_size = os.path.getsize(output_file)
        print(f"   File size: {file_size:,} bytes ({file_size / (1024*1024):.1f} MB)")
        
    else:
        print("\n❌ Failed to combine files!")

if __name__ == "__main__":
    main()
