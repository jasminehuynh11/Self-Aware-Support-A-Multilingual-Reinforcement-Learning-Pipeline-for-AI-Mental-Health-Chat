#!/usr/bin/env python3
"""
Simple VLLM inference script for processing questions with LoRA adapter
Usage: python simple_vllm_inference.py
"""

import json
import os
from datetime import datetime

def main():
    # Import libraries (install with: pip install vllm)
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        print("Please install VLLM: pip install vllm")
        return

    # Configuration
    BASE_MODEL = "unsloth/Qwen3-4B"
    LORA_PATH = "trained_model_v3_2_grpo"  # Your LoRA adapter path
    QUESTIONS_FILE = "./dataset/sampled_multilingual_200.jsonl"
    BATCH_SIZE = 4
    MAX_TOKENS = 7000
    
    # System prompt for counseling assistant
    SYSTEM_PROMPT = """You are a helpful mental health counselling assistant. Please answer mental health questions based on the patient's description.
Provide helpful, comprehensive, and appropriate answers to the user's questions.

After your counselling response, you must include a self-evaluation in the following format:
<evaluate>
{"Active Listening" : score, "Empathy & Validation" : score, "Safety & Trustworthiness" : score, "Open-mindedness & Non-judgment" : score, "Clarity & Encouragement" : score, "Boundaries & Ethical" : score, "Holistic Approach" : score, "Explaination for Scoring": "Your explanation here"}
</evaluate>

Where score is a number from 1-10, and provide a clear explanation for your scoring."""

    print("=== VLLM LoRA Inference ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapter: {LORA_PATH}")
    print(f"Questions file: {QUESTIONS_FILE}")
    
    # Check if files exist
    if not os.path.exists(QUESTIONS_FILE):
        print(f"Error: {QUESTIONS_FILE} not found!")
        return
    
    if not os.path.exists(LORA_PATH):
        print(f"Error: LoRA adapter {LORA_PATH} not found!")
        return
    
    # Load questions
    print("\nLoading questions...")
    questions = []
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    question_data = json.loads(line)
                    questions.append({
                        'line_number': line_num,
                        'data': question_data,
                        'text': question_data.get('question', question_data.get('text', str(question_data)))
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {line_num}")
    
    print(f"Loaded {len(questions)} questions")
    
    if not questions:
        print("No valid questions found!")
        return
    
    # Initialize VLLM
    print("\nInitializing VLLM...")
    llm = LLM(
        model=BASE_MODEL,
        max_model_len=8012,
        gpu_memory_utilization=0.8,
        enable_lora=True,
        max_lora_rank=128,
        trust_remote_code=True,
    )
    
    tokenizer = llm.get_tokenizer()
    lora_request = LoRARequest("counseling_lora", 1, LORA_PATH)
    
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
    
    # Format prompts
    def format_prompt(question_text):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Process questions in batches
    print(f"\nProcessing {len(questions)} questions in batches of {BATCH_SIZE}...")
    results = []
    
    for i in range(0, len(questions), BATCH_SIZE):
        batch = questions[i:i+BATCH_SIZE]
        batch_prompts = [format_prompt(q['text']) for q in batch]
        
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(questions)-1)//BATCH_SIZE + 1}")
        
        # Generate responses
        outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
        
        # Collect results
        for j, output in enumerate(outputs):
            question = batch[j]
            result = {
                'line_number': question['line_number'],
                'question': question['text'],
                'answer': output.outputs[0].text,
                'tokens_prompt': len(output.prompt_token_ids),
                'tokens_completion': len(output.outputs[0].token_ids),
                'finish_reason': output.outputs[0].finish_reason
            }
            results.append(result)
            
            # Show sample results
            if len(results) <= 3:
                print(f"Q{result['line_number']}: {result['question'][:80]}...")
                print(f"A{result['line_number']}: {result['answer'][:120]}...")
                print("-" * 40)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"vllm_results_{LORA_PATH}_{timestamp}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    # Print summary
    print(f"\n=== Results Summary ===")
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

if __name__ == "__main__":
    main()
