#!/usr/bin/env python3
"""
JSONL Translation Script using OpenAI API
Translates the 'turns' content in a JSONL file to Chinese, Vietnamese, and Arabic.
"""

import json
import os
import time
from typing import List, Dict, Any
from openai import OpenAI
import argparse
from pathlib import Path

class JSONLTranslator:
    def __init__(self, api_key: str = None):
        """
        Initialize the translator with OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
        """
        if api_key:
            self.client = OpenAI(
                api_key=api_key)
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            # Try to get from environment variable
            self.client = OpenAI(api_key=api_key)  # Will use OPENAI_API_KEY env var
        
        self.languages = {
            # 'chinese': 'Chinese (Simplified)',
            # 'vietnamese': 'Vietnamese', 
            # 'arabic': 'Arabic'
            'cantonese': 'Cantonese Chinese (Traditional)'
        }
        
        self.rate_limit_delay = 0.1  # Delay between requests to avoid rate limiting
    
    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate a single text using OpenAI's API.
        
        Args:
            text: Text to translate
            target_language: Target language name
            
        Returns:
            Translated text
        """
        try:
            prompt = f"Translate the following text to {target_language}. Maintain the original tone and meaning:\n\n{text}"
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate the given text to {target_language} while preserving the original meaning, tone, and context. For sensitive or mental health content, maintain the appropriate tone and meaning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1  # Lower temperature for more consistent translations
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Add rate limiting delay
            time.sleep(self.rate_limit_delay)
            
            return translated_text
            
        except Exception as e:
            print(f"Error translating text: {e}")
            return f"[TRANSLATION_ERROR: {str(e)}]"
    
    def process_jsonl_file(self, input_file: str, output_dir: str = None) -> None:
        """
        Process the JSONL file and create translated versions.
        
        Args:
            input_file: Path to input JSONL file
            output_dir: Directory to save translated files (default: same as input)
        """
        input_path = Path(input_file)
        
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Read the original file
        print(f"Reading input file: {input_file}")
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"Found {total_lines} records to translate")
        
        # Process each language
        for lang_code, lang_name in self.languages.items():
            print(f"\n=== Translating to {lang_name} ===")
            
            output_file = output_dir / f"{input_path.stem}_{lang_code}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as out_f:
                for i, line in enumerate(lines, 1):
                    try:
                        # Parse JSON line
                        data = json.loads(line.strip())
                        
                        print(f"Processing record {i}/{total_lines} for {lang_name}...")
                        
                        # Translate each turn
                        if 'turns' in data and isinstance(data['turns'], list):
                            translated_turns = []
                            for turn in data['turns']:
                                if isinstance(turn, str) and turn.strip():
                                    translated_turn = self.translate_text(turn, lang_name)
                                    translated_turns.append(translated_turn)
                                    print(f"Translated turn: {turn} -> {translated_turn} \n")
                                else:
                                    translated_turns.append(turn)  # Keep non-string or empty turns as-is
                            
                            # Update the data with translated turns
                            data['turns'] = translated_turns
                            data['original_language'] = 'english'
                            data['translated_to'] = lang_code
                        
                        # Write translated record
                        out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {i}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing line {i}: {e}")
                        continue
            
            print(f"✓ {lang_name} translation completed: {output_file}")
        
        print(f"\n All translations completed!")
        print(f"Output files saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Translate JSONL file content using OpenAI API')
    parser.add_argument('input_file', help='Path to input JSONL file')
    parser.add_argument('--output-dir', help='Output directory for translated files')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    parser.add_argument('--rate-limit', type=float, default=1.0, help='Delay between API calls in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return
    
    # Initialize translator
    try:
        translator = JSONLTranslator(api_key=args.api_key)
        translator.rate_limit_delay = args.rate_limit
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable or provided --api-key")
        return
    
    # Process the file
    try:
        translator.process_jsonl_file(args.input_file, args.output_dir)
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
