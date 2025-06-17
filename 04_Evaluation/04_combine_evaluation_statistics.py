#!/usr/bin/env python3
"""
Script to combine all evaluation statistics JSON files into a single CSV file.
This creates a comprehensive comparison of different model performances.
"""

import pandas as pd
import json
import os
import glob
from typing import List, Dict

def extract_model_info(filename: str) -> Dict[str, str]:
    """Extract model information from filename."""
    basename = os.path.basename(filename)
    # Remove extension and prefix
    model_part = basename.replace('evaluation_statistics_', '').replace('.json', '')
    
    if 'samantha' in model_part:
        model_type = 'Samantha'
        version = model_part.replace('samantha_model_', '')
    elif 'vllm' in model_part:
        model_type = 'VLLM'
        version = model_part.replace('vllm_model_', '').replace('vllm_model', 'base')
    else:
        model_type = 'Unknown'
        version = model_part
    
    return {
        'model_type': model_type,
        'version': version,
        'filename': basename
    }

def load_and_process_json_files(directory: str) -> pd.DataFrame:
    """Load all evaluation statistics JSON files and combine them."""
    
    # Find all evaluation statistics JSON files
    pattern = os.path.join(directory, "evaluation_statistics_*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        raise ValueError(f"No evaluation statistics JSON files found in {directory}")
    
    print(f"Found {len(json_files)} JSON files to process:")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")
    
    combined_data = []
    
    for json_file in sorted(json_files):
        try:
            # Extract model information from filename
            model_info = extract_model_info(json_file)
            
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create a row for this model
            row = {
                'model_type': model_info['model_type'],
                'version': model_info['version'],
                'filename': model_info['filename']
            }
            
            # Add all statistics from the JSON
            row.update(data)
            
            combined_data.append(row)
            print(f"✓ Processed: {model_info['model_type']} {model_info['version']}")
            
        except Exception as e:
            print(f"✗ Error processing {json_file}: {str(e)}")
            continue
    
    if not combined_data:
        raise ValueError("No valid data was processed from JSON files")
    
    # Convert to DataFrame
    df = pd.DataFrame(combined_data)
    
    # Reorder columns to put model info first
    info_columns = ['model_type', 'version', 'filename']
    other_columns = [col for col in df.columns if col not in info_columns]
    df = df[info_columns + sorted(other_columns)]
    
    return df

def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table with key metrics for easy comparison."""
    
    summary_metrics = [
        'avg_overall_score_mean',
        'avg_active_listening_mean',
        'avg_empathy_validation_mean',
        'avg_safety_trustworthiness_mean',
        'avg_openness_nonjudgment_mean',
        'avg_clarity_encouragement_mean',
        'avg_boundaries_ethical_mean',
        'avg_holistic_approach_mean'
    ]
    
    # Select relevant columns
    summary_columns = ['model_type', 'version'] + summary_metrics
    summary_df = df[summary_columns].copy()
    
    # Rename columns for better readability
    column_mapping = {
        'avg_overall_score_mean': 'Overall Score',
        'avg_active_listening_mean': 'Active Listening',
        'avg_empathy_validation_mean': 'Empathy & Validation',
        'avg_safety_trustworthiness_mean': 'Safety & Trustworthiness',
        'avg_openness_nonjudgment_mean': 'Openness & Non-judgment',
        'avg_clarity_encouragement_mean': 'Clarity & Encouragement',
        'avg_boundaries_ethical_mean': 'Boundaries & Ethical',
        'avg_holistic_approach_mean': 'Holistic Approach'
    }
    
    summary_df = summary_df.rename(columns=column_mapping)
    
    # Round to 2 decimal places
    numeric_columns = [col for col in summary_df.columns if col not in ['model_type', 'version']]
    summary_df[numeric_columns] = summary_df[numeric_columns].round(2)
    
    # Sort by overall score descending
    summary_df = summary_df.sort_values('Overall Score', ascending=False)
    
    return summary_df

def main():
    """Main function to combine evaluation JSON files."""
    
    # Configuration
    input_directory = "/home/taitran/mq/mq-nlp-group/4_Evaluation"
    output_file = "/home/taitran/mq/mq-nlp-group/4_Evaluation/combined_evaluation_statistics.csv"
    summary_file = "/home/taitran/mq/mq-nlp-group/4_Evaluation/evaluation_summary_comparison.csv"
    
    print("Evaluation Statistics Combiner")
    print("=" * 50)
    print(f"Input directory: {input_directory}")
    print(f"Output file: {output_file}")
    print(f"Summary file: {summary_file}")
    print()
    
    try:
        # Load and process all JSON files
        print("Loading and processing JSON files...")
        df = load_and_process_json_files(input_directory)
        
        print(f"\nSuccessfully combined {len(df)} model evaluations")
        print(f"Total columns: {len(df.columns)}")
        
        # Save detailed results
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ Detailed results saved to: {output_file}")
        
        # Create and save summary
        print("\\nCreating summary comparison...")
        summary_df = create_summary_statistics(df)
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"✓ Summary comparison saved to: {summary_file}")
        
        # Display summary
        print(f"\\n=== MODEL PERFORMANCE SUMMARY ===")
        print(summary_df.to_string(index=False))
        
        # Show top performers by metric
        print(f"\\n=== TOP PERFORMERS BY METRIC ===")
        metrics = ['Overall Score', 'Active Listening', 'Empathy & Validation', 
                  'Safety & Trustworthiness', 'Openness & Non-judgment']
        
        for metric in metrics:
            if metric in summary_df.columns:
                top_model = summary_df.loc[summary_df[metric].idxmax()]
                print(f"{metric:25}: {top_model['model_type']} {top_model['version']} ({top_model[metric]:.2f})")
        
        print(f"\\n🎉 Evaluation statistics combination completed successfully!")
        print(f"Detailed results: {output_file}")
        print(f"Summary comparison: {summary_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
