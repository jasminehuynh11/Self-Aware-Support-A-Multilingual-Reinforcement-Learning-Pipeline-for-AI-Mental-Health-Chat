#!/usr/bin/env python3
"""
VLLM inference script for Samantha models (full models, no LoRA)
Usage: python samantha_vllm_inference.py
"""

import json
import os
from datetime import datetime

def main():
    # Import libraries (install with: pip install vllm)
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Please install VLLM: pip install vllm")
        return

    # Configuration - Samantha models
    MODELS = {
        "samantha-1.11": {
            "path": "Samantha-1.11-7b",  # Your local model path
            "description": "Samantha 1.11 7B model"
        },
        "samantha-1.2": {
            "path": "samantha-1.2-mistral-7b",  # Your local model path  
            "description": "Samantha 1.2 Mistral 7B model"
        },
        "qwenbase": {
            "path": "unsloth/Qwen3-4B",  # Example for Qwen base model
            "description": "Qwen 3 4B model"
        }
    }
    
    # Select which model to use (change this as needed)
    MODEL_KEY = "samantha-1.11"  # or "samantha-1.2"
    MODEL_PATH = MODELS[MODEL_KEY]["path"]
    MODEL_DESC = MODELS[MODEL_KEY]["description"]
    
    QUESTIONS_FILE = "./dataset/sampled_multilingual_200.jsonl"
    BATCH_SIZE = 4
    MAX_TOKENS = 7000
    
    # System prompt for Samantha (adjust as needed for the model)
    SYSTEM_PROMPT = """You are Samantha, a helpful AI assistant. Please provide thoughtful, comprehensive, and appropriate responses to user questions. Be empathetic, clear, and supportive in your communication."""

    print("=== Samantha VLLM Inference ===")
    print(f"Model: {MODEL_DESC}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Questions file: {QUESTIONS_FILE}")
    
    # Check if files exist
    if not os.path.exists(QUESTIONS_FILE):
        print(f"Error: {QUESTIONS_FILE} not found!")
        return
    
    # if not os.path.exists(MODEL_PATH):
    #     print(f"Error: Model path {MODEL_PATH} not found!")
    #     print("Available models:")
    #     for key, model_info in MODELS.items():
    #         exists = "✓" if os.path.exists(model_info["path"]) else "✗"
    #         print(f"  {exists} {key}: {model_info['path']}")
    #     return
    
    # Load questions
    print("\nLoading questions...")
    questions = []
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    question_data = json.loads(line)
                    # Extract question text from different possible formats
                    if isinstance(question_data, dict):
                        if 'turns' in question_data and question_data['turns']:
                            # Handle the format with 'turns' array
                            question_text = question_data['turns'][0]
                        else:
                            # Handle other formats
                            question_text = question_data.get('question', 
                                                            question_data.get('text', 
                                                                             question_data.get('content', str(question_data))))
                    else:
                        question_text = str(question_data)
                    
                    questions.append({
                        'line_number': line_num,
                        'data': question_data,
                        'text': question_text,
                        'question_id': question_data.get('question_id', line_num),
                        'category': question_data.get('category', 'unknown'),
                        'original_language': question_data.get('original_language', 'unknown')
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {line_num}")
    
    print(f"Loaded {len(questions)} questions")
    
    if not questions:
        print("No valid questions found!")
        return
    
    # Initialize VLLM (no LoRA needed for full models)
    print("\nInitializing VLLM...")
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=4096,  # Adjust based on model capabilities
        gpu_memory_utilization=0.85,  # Can use more memory since no LoRA
        trust_remote_code=True,
        dtype="float16",  # Use float16 for better performance
        tensor_parallel_size=1,  # Adjust if you have multiple GPUs
    )
    
    tokenizer = llm.get_tokenizer()
    
    # Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=MAX_TOKENS,
        repetition_penalty=1.1,
        stop=[tokenizer.eos_token] if hasattr(tokenizer, 'eos_token') else None
    )
    
    print("VLLM initialized successfully!")
    
    # Format prompts (adjust for Samantha's expected format)
    def format_prompt(question_text):
        # Simple format for Samantha - adjust based on model's training
        # Some Samantha models use specific chat formats
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question_text}
                ]
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback to simple format
                return f"System: {SYSTEM_PROMPT}\n\nUser: {question_text}\n\nAssistant:"
        else:
            # Simple format if no chat template available
            return f"System: {SYSTEM_PROMPT}\n\nUser: {question_text}\n\nAssistant:"
    
    # Process questions in batches
    print(f"\nProcessing {len(questions)} questions in batches of {BATCH_SIZE}...")
    results = []
    
    for i in range(0, len(questions), BATCH_SIZE):
        batch = questions[i:i+BATCH_SIZE]
        batch_prompts = [format_prompt(q['text']) for q in batch]
        
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(questions)-1)//BATCH_SIZE + 1}")
        
        # Generate responses (no LoRA request needed)
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Collect results
        for j, output in enumerate(outputs):
            question = batch[j]
            result = {
                'line_number': question['line_number'],
                'question_id': question['question_id'],
                'category': question['category'],
                'original_language': question['original_language'],
                'question': question['text'],
                'answer': output.outputs[0].text,
                'tokens_prompt': len(output.prompt_token_ids),
                'tokens_completion': len(output.outputs[0].token_ids),
                'finish_reason': output.outputs[0].finish_reason,
                'model': MODEL_KEY,
                'model_path': MODEL_PATH
            }
            results.append(result)
            
            # Show sample results
            if len(results) <= 3:
                print(f"Q{result['line_number']}: {result['question'][:80]}...")
                print(f"A{result['line_number']}: {result['answer'][:120]}...")
                print("-" * 40)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"samantha_results_{MODEL_KEY}_{timestamp}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    # Print summary
    print(f"\n=== Results Summary ===")
    print(f"Model: {MODEL_DESC}")
    print(f"Total questions processed: {len(results)}")
    print(f"Average prompt tokens: {sum(r['tokens_prompt'] for r in results) / len(results):.1f}")
    print(f"Average completion tokens: {sum(r['tokens_completion'] for r in results) / len(results):.1f}")
    print(f"Results saved to: {output_file}")
    
    # Show final few results
    print(f"\nLast few results:")
    for result in results[-2:]:
        print(f"Q{result['line_number']}: {result['question'][:60]}...")
        print(f"A{result['line_number']}: {result['answer'][:100]}...")
        print("-" * 40)

def run_both_models():
    """Function to run inference on both Samantha models"""
    print("=== Running inference on both Samantha models ===")
    
    models_to_run = ["samantha-1.11", "samantha-1.2", "qwenbase"]
    
    for model_key in models_to_run:
        print(f"\n{'='*60}")
        print(f"Starting inference with {model_key}")
        print(f"{'='*60}")
        
        # Temporarily modify the global MODEL_KEY
        global MODEL_KEY
        original_model = MODEL_KEY
        MODEL_KEY = model_key
        
        try:
            main()
        except Exception as e:
            print(f"Error with {model_key}: {e}")
        finally:
            MODEL_KEY = original_model
        
        print(f"Completed inference with {model_key}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--both":
        run_both_models()
    else:
        main()
