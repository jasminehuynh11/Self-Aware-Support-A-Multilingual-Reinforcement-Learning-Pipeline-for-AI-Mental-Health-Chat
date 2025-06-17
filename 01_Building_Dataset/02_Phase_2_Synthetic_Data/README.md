# Synthetic Mental Health Counseling Data Generator

This Python script generates synthetic conversations between patients and mental health professionals using GPT-4.1-mini. The generated data follows specific distribution requirements and cultural considerations.

## Features

- **Even Distribution**: Ensures balanced sampling across:
  - 500 diverse user profiles from `diverse_sample_500.json`
  - 5 language groups (English, Vietnamese, Arabic, Cantonese, Mandarin) - 1000 samples each
  - 33 topics from `social_category.csv`

- **Parallel Processing**: Uses 4 threads by default for faster generation

- **Cultural Sensitivity**: Incorporates cultural expressions and understandings of mental distress for each language group

- **Comprehensive Conversations**: Each generated conversation includes:
  - Patient's goals and emotions
  - Specific triggering situations
  - Symptom descriptions (frequency, intensity, duration)
  - Life events and family dynamics
  - Coping strategies
  - Questions about the therapeutic process

## Requirements

```bash
pip install -r requirements.txt
```

## Setup

1. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Ensure Required Files Exist**:
   - `diverse_sample_500.json` - User profiles
   - `02_social_category.csv` - Topic categories
   - `02_sample_full_prompt.md` - Base prompt template

## Usage

### Option 1: Using the example script
```bash
python 03_run_generator.py
```
Choose option 1 for a small test (20 samples) or option 2 for full generation (5000 samples).

### Option 2: Direct usage
```python
from generate_synthetic_data import SyntheticDataGenerator
import os

# Initialize
generator = SyntheticDataGenerator(os.getenv("OPENAI_API_KEY"))

# Generate data
results = generator.generate_batch(
    total_samples=5000,
    num_threads=4,
    output_file="synthetic_counseling_data.json"
)
```

## Output Format

Each generated sample contains:
```json
{
  "sample_id": 1,
  "profile": {...},
  "nationality": "Vietnamese",
  "topic": "Family",
  "patient_message": "...",
  "doctor_message": "...",
  "full_response": "...",
  "timestamp": 1234567890.12
}
```

## Distribution Targets

- **Total Samples**: 5,000
- **Language Distribution**: 1,000 samples per language
  - English: 1,000
  - Vietnamese: 1,000
  - Arabic: 1,000
  - Cantonese: 1,000
  - Mandarin: 1,000
- **Profile Distribution**: Even usage of all 500 profiles
- **Topic Distribution**: Even coverage of all 33 categories

## Performance

- **Speed**: ~4-8 samples per minute with 4 threads
- **Estimated Time**: 30-60 minutes for 5,000 samples
- **API Costs**: Approximately $15-25 for 5,000 samples (depending on response length)

## Cultural Considerations

The generator incorporates specific cultural understandings for each language group:

- **Vietnamese**: Considers shame, "loss of face", somatization, indirect communication
- **Mandarin**: Accounts for denial, somatic expression, family consultation patterns
- **Arabic**: Respects family decision-making, religious context, stigma concerns
- **Cantonese**: Similar to Mandarin with regional variations
- **English**: Standard Western therapeutic communication patterns

## Error Handling

- Automatic retry for failed API calls
- Progress tracking and reporting
- Failed sample logging
- Graceful handling of API rate limits

## Files Generated

- `synthetic_counseling_data.json` - Main output file
- `test_synthetic_data.json` - Test output (if running small test)
- Progress logs printed to console

## Customization

You can modify:
- Number of samples per language group
- Number of threads for parallel processing
- Output file names
- Prompt templates in `total_prompt.md`
- Additional topics in `social_category.csv`

## Troubleshooting

1. **API Key Issues**: Ensure `OPENAI_API_KEY` is set correctly
2. **File Not Found**: Check that all required files exist in the correct locations
3. **Rate Limits**: The script handles rate limits automatically with delays
4. **Memory Issues**: For very large datasets, consider processing in smaller batches

## License

This tool is for research and educational purposes in mental health data generation.
