import pandas as pd
import openai
import os
import time
import json
import sys

def setup_openai_client():
    """Setup OpenAI client with API key from environment variable"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    return openai.OpenAI(api_key=api_key)

def translate_text_multiple_languages(client, text: str) -> dict:
    """Translate text to all target languages in a single API call"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a professional translator. Translate the given text to the following languages:
1. Vietnamese
2. Mandarin Chinese (Simplified)
3. Arabic
4. Cantonese Chinese (Traditional)

Return the translations in JSON format with the following structure:
{
    "vietnamese": "translated text in Vietnamese",
    "mandarin": "translated text in Mandarin Chinese",
    "arabic": "translated text in Arabic", 
    "cantonese": "translated text in Cantonese Chinese"
}

Only return the JSON, no explanations or additional content."""
                },
                {
                    "role": "user", 
                    "content": text
                }
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        try:
            translations = json.loads(result)
            return translations
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON response for text: {text[:50]}...")
            # Return empty translations if JSON parsing fails
            return {
                "vietnamese": text,
                "mandarin": text,
                "arabic": text,
                "cantonese": text
            }
    
    except Exception as e:
        print(f"Error translating text: {e}")
        # Return original text for all languages if translation fails
        return {
            "vietnamese": text,
            "mandarin": text,
            "arabic": text,
            "cantonese": text
        }

def main():
    # Setup OpenAI client
    client = setup_openai_client()
    
    # Read the CSV file (first 100 rows)
    print("Reading CSV file...")
    csv_file_path = '/home/taitran/mq/mq-nlp-group/utils/Interview_Data_6K.csv'
    
    try:
        # Read only first 100 rows (plus header)
        df = pd.read_csv(csv_file_path, nrows=100)
        print(f"Loaded {len(df)} rows from CSV")
        
        # Check if 'input' column exists
        if 'input' not in df.columns:
            print("Error: 'input' column not found in CSV file")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Languages to translate to
        languages = ['vietnamese', 'mandarin', 'arabic', 'cantonese']
        
        # Initialize dictionaries to store translations for each language
        translations = {lang: [] for lang in languages}
        
        # Process each row
        total_rows = len(df)
        for idx, row in df.iterrows():
            input_text = row['input']
            
            print(f"Translating row {idx + 1}/{total_rows}...")
            
            if pd.isna(input_text) or input_text == "":
                # Handle empty/null values
                for lang in languages:
                    translations[lang].append("")
            else:
                # Translate to all languages
                row_translations = translate_text_multiple_languages(client, str(input_text))
                
                for lang in languages:
                    translations[lang].append(row_translations.get(lang, str(input_text)))
            
            # Rate limiting - wait between requests
            time.sleep(1)
        
        # Create and save CSV files for each language
        for language in languages:
            print(f"\nCreating CSV file for {language.title()}...")
            
            # Create a copy of the dataframe
            translated_df = df.copy()
            
            # Update the input column with translations
            translated_df['input'] = translations[language]
            
            # Save to new CSV file
            output_filename = f'/home/taitran/mq/mq-nlp-group/utils/Interview_Data_6K_{language}.csv'
            translated_df.to_csv(output_filename, index=False)
            print(f"Saved translated data to: {output_filename}")
        
        print("\n✅ Translation completed successfully!")
        print("Created files:")
        for language in languages:
            print(f"  - Interview_Data_6K_{language}.csv")
    
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
