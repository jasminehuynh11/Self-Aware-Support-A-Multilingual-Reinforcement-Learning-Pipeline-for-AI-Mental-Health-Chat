#!/usr/bin/env python3
"""
Demo script to show a single response generation with evaluation metrics.
"""

import openai
import os
import json

def demo_single_response():
    """Demo a single response generation."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set your OPENAI_API_KEY environment variable")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
    # Demo Vietnamese input (as specified in the user's example)
    demo_input = "Tôi đã gặp khó khăn trong việc tìm kiếm sự cân bằng giữa trách nhiệm công việc và vai trò làm mẹ đơn thân của một cậu con trai 12 tuổi. Tôi nhận thấy mình thường cảm thấy quá tải và lo lắng, và tôi đang cân nhắc việc thiết lập một lịch trình có cấu trúc hơn cho cả hai mẹ con. Tuy nhiên, tôi còn do dự trong việc đặt ra giới hạn nghiêm ngặt về thời gian sử dụng thiết bị điện tử, vì tôi nhận thấy rằng việc đặt giới hạn chặt chẽ đôi khi có thể dẫn đến tranh cãi và sự oán giận. Thay vào đó, tôi đang nghĩ đến các chiến lược linh hoạt hơn, như thưởng thêm thời gian sử dụng thiết bị hoặc các vật phẩm trong trò chơi khi con có hành vi tốt. Nhưng tôi không chắc liệu những phần thưởng này có còn là lựa chọn khả thi hay không, vì chúng có thể củng cố thói quen sử dụng thiết bị điện tử không lành mạnh."
    
    system_message = """You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description.The assistant gives helpful, comprehensive, and appropriate answers to the user's questions. At the end of answer, add tag <evaluate>{"Active Listening" : score, "Empathy & Validation": score, "Safety & Trustworthiness" : score, "Open-mindedness & Non-judgment" : score, "Clarity & Encouragement" : score, "Boundaries & Ethical" : score, "Holistic Approach" : score, "Explaination for Scoring": explain} </evaluate> evaluate your consultant answer in 7 metrics and explain for that evaluation with score from 1 to 10 in json format, where 1 is the worst and 10 is the best and explain is clearly explain why has that score. 

Consultation Metrics:
1. Active Listening: Responses should show careful consideration of the user's concerns, reflecting an understanding and capturing the essence of the issue. Avoid making assumptions or jumping to conclusions.
2. Empathy & Validation: Responses should convey deep understanding and compassion, validating the user's feelings and emotions without being dismissive or minimizing their experiences.
3. Safety & Trustworthiness: Prioritize user safety in responses, refraining from potentially harmful or insensitive language. Ensure that information provided is consistent and trustworthy.
4. Open-mindedness & Non-judgment: Approach concerns without any inherent bias or judgment. Answers should be free from biases related to personal attributes and convey respect, demonstrating unconditional positive regard.
5. Clarity & Encouragement: Provide clear, concise, and easily understandable answers. Where appropriate, motivate or highlight strengths, offering encouragement while maintaining a neutral stance.
6. Boundaries & Ethical: It's vital to clarify the role of the response, emphasizing its informational nature. In complex scenarios, guiding users to seek human professional assistance is essential.
7. Holistic Approach: Responses should be comprehensive, addressing concerns from various angles, be it emotional, cognitive, or situational. Consider the broader context, even if not explicitly detailed in the query.

Please respond in Vietnamese language to match the user's input language."""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": demo_input}
    ]
    
    print("Demo: Single Response Generation with Evaluation")
    print("=" * 60)
    print("Input (Vietnamese):")
    print(demo_input[:200] + "..." if len(demo_input) > 200 else demo_input)
    print("\nGenerating response...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )
        
        generated_response = response.choices[0].message.content
        
        print("\nGenerated Response:")
        print("-" * 40)
        print(generated_response)
        
        # Try to extract and parse the evaluation
        if "<evaluate>" in generated_response and "</evaluate>" in generated_response:
            start_idx = generated_response.find("<evaluate>") + 10
            end_idx = generated_response.find("</evaluate>")
            eval_json = generated_response[start_idx:end_idx]
            
            try:
                eval_data = json.loads(eval_json)
                print("\n" + "=" * 60)
                print("EVALUATION METRICS:")
                print("=" * 60)
                for metric, score in eval_data.items():
                    if metric != "Explaination for Scoring":
                        print(f"{metric}: {score}/10")
                
                if "Explaination for Scoring" in eval_data:
                    print(f"\nExplanation: {eval_data['Explaination for Scoring']}")
                    
            except json.JSONDecodeError:
                print("\nNote: Evaluation JSON could not be parsed, but evaluation tag is present")
        else:
            print("\nNote: No evaluation tag found in response")
            
        print(f"\nResponse length: {len(generated_response)} characters")
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    demo_single_response()

if __name__ == "__main__":
    main()
