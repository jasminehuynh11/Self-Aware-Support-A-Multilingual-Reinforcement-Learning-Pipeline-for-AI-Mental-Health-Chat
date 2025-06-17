#!/usr/bin/env python3
"""
Usage example for the synthetic data generator.
This script demonstrates how to use the SyntheticDataGenerator class.
"""

import os
from generate_synthetic_data import SyntheticDataGenerator

def run_small_test():
    """Run a small test with 20 samples to verify everything works."""
    print("Running small test with 20 samples...")
    
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize generator
    generator = SyntheticDataGenerator(api_key)
    
    # Generate test data
    results = generator.generate_batch(
        total_samples=20,
        num_threads=2,
        output_file="test_synthetic_data.json"
    )
    
    print(f"Test completed! Generated {len(results)} samples.")
    print("Check 'test_synthetic_data.json' for results.")

def run_full_generation():
    """Run the full generation with 5000 samples."""
    print("Running full generation with 5000 samples...")
    print("This will take a while (estimated 30-60 minutes)...")
    
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize generator
    generator = SyntheticDataGenerator(api_key)
    
    # Generate full dataset
    results = generator.generate_batch(
        total_samples=5000,
        num_threads=4,
        output_file="full_synthetic_counseling_data.json"
    )
    
    print(f"Full generation completed! Generated {len(results)} samples.")
    print("Check 'full_synthetic_counseling_data.json' for results.")

def main():
    """Main function with user choice."""
    print("Synthetic Mental Health Counseling Data Generator")
    print("=" * 50)
    print("1. Run small test (20 samples)")
    print("2. Run full generation (5000 samples)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        run_small_test()
    elif choice == "2":
        run_full_generation()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main()
