#!/usr/bin/env python3
"""
Multilingual Mental Health Chatbot with DASS-21 Integration
Built for Qwen3-4B LoRA model via VLLM API
"""

import gradio as gr
import requests
import json
import time
import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

class MentalHealthChatbot:
    def __init__(self, vllm_base_url: str = "http://127.0.0.1:8000"):
        self.vllm_base_url = vllm_base_url
        self.model_name = "unsloth/Qwen3-4B"  # The best model (v3.2) but we still use the original name for compatibility
        self.conversation_history = []
        self.dass21_results = []
        
        # DASS-21 evaluation prompt (with escaped braces for JSON structure)
        self.dass21_prompt = """**Role:** You are an expert AI assistant with a specialization in computational linguistics and psychological assessment. Your task is to analyze a patient's self-reported text from a mental health context and evaluate it based on the Depression Anxiety Stress Scales 21 (DASS-21) questionnaire.

**Objective:** Read the provided text, and for each of the 21 items on the DASS-21, assign a score that reflects the patient's experiences **over the past week** as described in their statement. Then, calculate the total scores for the Depression, Anxiety, and Stress subscales. Finally, format the entire output as a single JSON object.

**Context: DASS-21 Framework**

The DASS-21 is a self-report questionnaire designed to measure the negative emotional states of depression, anxiety, and stress.

* **Scoring Scale:** You must use the following 4-point severity/frequency scale for each item:
    * **0:** "Did not apply to me at all" - NEVER
    * **1:** "Applied to me to some degree, or some of the time" - SOMETIMES
    * **2:** "Applied to me to a considerable degree, or a good part of time" - OFTEN
    * **3:** "Applied to me very much, or most of the time" - ALMOST ALWAYS

**Task Instructions:**

1.  **Analyze the Input Text:** Carefully read the patient's text provided below.
2.  **Score Each Item:** For each of the 21 statements listed in the "JSON Output Structure" below, evaluate the patient's text and assign the most appropriate score (0, 1, 2, or 3).
3.  **Provide Justification:** For each score, include a brief `justification` string, quoting or referencing the part of the text that supports your scoring decision. If the text does not contain information for a specific item, score it as 0 and state that in the justification.
4.  **Generate JSON Output:** Present the final results in a single, valid JSON object, following the precise structure outlined below.

**JSON Output Structure:**

```json
{{
  "assessment": {{
    "instrument": "DASS-21",
    "scores": {{
      "depression": {{
        "items": [
          {{"item": 3, "statement": "I couldn't seem to experience any positive feeling at all", "score": null, "justification": ""}},
          {{"item": 5, "statement": "I found it difficult to work up the initiative to do things", "score": null, "justification": ""}},
          {{"item": 10, "statement": "I felt that I had nothing to look forward to", "score": null, "justification": ""}},
          {{"item": 13, "statement": "I felt down-hearted and blue", "score": null, "justification": ""}},
          {{"item": 16, "statement": "I was unable to become enthusiastic about anything", "score": null, "justification": ""}},
          {{"item": 17, "statement": "I felt I wasn't worth much as a person", "score": null, "justification": ""}},
          {{"item": 21, "statement": "I felt that life was meaningless", "score": null, "justification": ""}}
        ],
        "total_score": null
      }},
      "anxiety": {{
        "items": [
          {{"item": 2, "statement": "I was aware of dryness of my mouth", "score": null, "justification": ""}},
          {{"item": 4, "statement": "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness in the absence of physical exertion)", "score": null, "justification": ""}},
          {{"item": 7, "statement": "I experienced trembling (e.g., in the hands)", "score": null, "justification": ""}},
          {{"item": 9, "statement": "I was worried about situations in which I might panic and make a fool of myself", "score": null, "justification": ""}},
          {{"item": 15, "statement": "I felt I was close to panic", "score": null, "justification": ""}},
          {{"item": 19, "statement": "I was aware of the action of my heart in the absence of physical exertion (e.g., sense of heart rate increase, heart missing a beat)", "score": null, "justification": ""}},
          {{"item": 20, "statement": "I felt scared without any good reason", "score": null, "justification": ""}}
        ],
        "total_score": null
      }},
      "stress": {{
        "items": [
          {{"item": 1, "statement": "I found it hard to wind down", "score": null, "justification": ""}},
          {{"item": 6, "statement": "I tended to over-react to situations", "score": null, "justification": ""}},
          {{"item": 8, "statement": "I felt that I was using a lot of nervous energy", "score": null, "justification": ""}},
          {{"item": 11, "statement": "I found myself getting agitated", "score": null, "justification": ""}},
          {{"item": 12, "statement": "I found it difficult to relax", "score": null, "justification": ""}},
          {{"item": 14, "statement": "I was intolerant of anything that kept me from getting on with what I was doing", "score": null, "justification": ""}},
          {{"item": 18, "statement": "I felt I was rather touchy", "score": null, "justification": ""}}
        ],
        "total_score": null
      }}
    }}
  }}
}}
```

Patient sharing: {user_text}"""

    def call_vllm_api(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call VLLM API for inference"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "top_p": 0.9,
                "stop": ["</s>", "<|im_end|>"]
            }
            
            response = requests.post(
                f"{self.vllm_base_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["text"].strip()
            else:
                return f"Error: API call failed with status {response.status_code}"
                
        except Exception as e:
            return f"Error calling VLLM API: {str(e)}"

    def call_vllm_chat_api(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        """Call VLLM API for chat completions with proper message structure"""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "top_p": 0.9,
                "stop": ["</s>", "<|im_end|>"]
            }
            
            response = requests.post(
                f"{self.vllm_base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return f"Error: Chat API call failed with status {response.status_code}"
                
        except Exception as e:
            return f"Error calling VLLM Chat API: {str(e)}"

    def get_counseling_response(self, user_message: str) -> str:
        """Get counseling response from the model using chat format with fallback"""
        # First try chat completions API
        messages = [
            {
                "role": "system",
                "content": """/no_think You are a compassionate, multilingual mental health counselor. Provide supportive, empathetic responses that demonstrate active listening, cultural sensitivity, and professional boundaries. Always maintain a warm, non-judgmental tone while offering practical guidance when appropriate.

Key guidelines:
- Show empathy and understanding
- Validate the user's feelings
- Provide helpful, evidence-based suggestions when appropriate
- Maintain professional boundaries
- Be culturally sensitive
- Keep responses concise but comprehensive (limit to 500 words)
- If the user expresses crisis thoughts, encourage seeking immediate professional help"""
            },
            {
                "role": "user", 
                "content": user_message
            }
        ]
        
        response = self.call_vllm_chat_api(messages, max_tokens=1024)
        
        # If chat API fails, fallback to completions API
        if response.startswith("Error:"):
            print("DEBUG - Chat API failed, falling back to completions API")
            fallback_prompt = f"""You are a compassionate, multilingual mental health counselor. Provide supportive, empathetic responses that demonstrate active listening, cultural sensitivity, and professional boundaries. Always maintain a warm, non-judgmental tone while offering practical guidance when appropriate.

User: {user_message}

Counselor:"""
            return self.call_vllm_api(fallback_prompt, max_tokens=1024)
        
        return response

    def evaluate_dass21(self, user_message: str) -> Dict:
        """Evaluate user message using DASS-21 criteria"""
        full_prompt = self.dass21_prompt.format(user_text=user_message)
        
        response = self.call_vllm_api(full_prompt, max_tokens=4000)  # Increased token limit
        
        # Debug: Print the raw response (truncated for readability)
        print(f"DEBUG - Raw DASS-21 response length: {len(response)}")
        print(f"DEBUG - Raw DASS-21 response preview: {response[:200]}...")
        print(f"DEBUG - Raw DASS-21 response ending: ...{response[-200:]}")
        
        try:
            # Clean up the response first
            response = response.strip()
            
            # Multiple strategies for JSON extraction
            json_str = None
            
            # Strategy 1: Look for ```json blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_end > json_start:
                    json_str = response[json_start:json_end].strip()
                    print("DEBUG - Using strategy 1: ```json blocks")
            
            # Strategy 2: Look for complete JSON object
            if not json_str and "{" in response and "}" in response:
                # Find the first { and try to match braces
                start = response.find("{")
                if start >= 0:
                    brace_count = 0
                    end = start
                    for i, char in enumerate(response[start:], start):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    
                    if brace_count == 0:  # Found complete JSON
                        json_str = response[start:end]
                        print("DEBUG - Using strategy 2: brace matching")
            
            # Strategy 3: Try to find JSON between assessment tags or similar
            if not json_str:
                # Look for patterns that might indicate JSON structure
                patterns = [
                    r'\{[^{}]*"assessment"[^{}]*\{.*?\}\s*\}',
                    r'\{.*?"instrument".*?"DASS-21".*?\}'
                ]
                for pattern in patterns:
                    match = re.search(pattern, response, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        print("DEBUG - Using strategy 3: regex pattern matching")
                        break
            
            if not json_str:
                return {
                    "error": "No valid JSON structure found in response", 
                    "raw_response": response,
                    "response_length": len(response)
                }
            
            print(f"DEBUG - Extracted JSON length: {len(json_str)}")
            print(f"DEBUG - Extracted JSON preview: {json_str[:300]}...{json_str[-100:]}")
            
            # Try to parse the JSON
            try:
                parsed_result = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                print(f"DEBUG - Initial JSON parse failed: {e}")
                
                # Fix common issues
                json_str_fixed = json_str
                
                # Fix trailing commas
                json_str_fixed = re.sub(r',\s*}', '}', json_str_fixed)
                json_str_fixed = re.sub(r',\s*]', ']', json_str_fixed)
                
                # Fix incomplete strings (add closing quotes)
                # This is more complex, so let's try a simpler approach
                
                try:
                    parsed_result = json.loads(json_str_fixed)
                    print("DEBUG - JSON fixed and parsed successfully")
                except json.JSONDecodeError as e2:
                    # Last resort: create a simple assessment based on text analysis
                    print("DEBUG - Using fallback simple assessment")
                    return self.create_fallback_assessment(user_message, str(e2))
            
            # Validate the structure and calculate total scores if missing
            if "assessment" in parsed_result and "scores" in parsed_result["assessment"]:
                scores = parsed_result["assessment"]["scores"]
                
                # Calculate total scores if they're null
                for category in ["depression", "anxiety", "stress"]:
                    if category in scores and "items" in scores[category]:
                        if scores[category].get("total_score") is None:
                            total = sum(
                                item.get("score", 0) or 0 
                                for item in scores[category]["items"] 
                                if isinstance(item.get("score"), (int, float))
                            )
                            scores[category]["total_score"] = total
                            print(f"DEBUG - Calculated {category} total: {total}")
                        else:
                            print(f"DEBUG - {category} total already set: {scores[category]['total_score']}")
            
            return parsed_result
            
        except Exception as e:
            return {
                "error": f"Failed to parse DASS-21 evaluation: {str(e)}", 
                "raw_response": response[:1000] + "..." if len(response) > 1000 else response,
                "exception_type": type(e).__name__
            }
    
    def create_fallback_assessment(self, user_message: str, parse_error: str) -> Dict:
        """Create a simple fallback assessment when JSON parsing fails"""
        text_lower = user_message.lower()
        
        # Simple keyword-based scoring (very basic fallback)
        depression_keywords = ["sad", "depressed", "hopeless", "worthless", "meaningless", "down", "blue", "empty"]
        anxiety_keywords = ["anxious", "worried", "panic", "scared", "nervous", "tense", "fear"]
        stress_keywords = ["stressed", "overwhelmed", "pressure", "agitated", "irritated", "touchy", "energy"]
        
        depression_score = min(3, sum(1 for word in depression_keywords if word in text_lower))
        anxiety_score = min(3, sum(1 for word in anxiety_keywords if word in text_lower))
        stress_score = min(3, sum(1 for word in stress_keywords if word in text_lower))
        
        return {
            "assessment": {
                "instrument": "DASS-21",
                "scores": {
                    "depression": {"total_score": depression_score},  # Raw score, will be doubled later
                    "anxiety": {"total_score": anxiety_score},
                    "stress": {"total_score": stress_score}
                }
            },
            "fallback": True,
            "parse_error": parse_error,
            "note": "This is a simplified assessment due to JSON parsing issues"
        }

    def create_dass21_visualization(self, assessment_data: Dict) -> go.Figure:
        """Create visualization for DASS-21 results"""
        if "error" in assessment_data:
            print(f"DEBUG - Visualization error: {assessment_data.get('error', 'Unknown error')}")
            return None
            
        try:
            if "assessment" not in assessment_data:
                print("DEBUG - No 'assessment' key in data")
                return None
                
            if "scores" not in assessment_data["assessment"]:
                print("DEBUG - No 'scores' key in assessment data")
                return None
                
            scores = assessment_data["assessment"]["scores"]
            
            # Extract total scores with validation
            depression_score = scores.get("depression", {}).get("total_score", 0)
            anxiety_score = scores.get("anxiety", {}).get("total_score", 0)
            stress_score = scores.get("stress", {}).get("total_score", 0)
            
            # Handle None values
            depression_score = depression_score if depression_score is not None else 0
            anxiety_score = anxiety_score if anxiety_score is not None else 0
            stress_score = stress_score if stress_score is not None else 0
            
            print(f"DEBUG - Raw scores: D={depression_score}, A={anxiety_score}, S={stress_score}")
            
            # Multiply by 2 for DASS-21 (as per standard scoring)
            depression_final = depression_score * 2
            anxiety_final = anxiety_score * 2
            stress_final = stress_score * 2
            
            # Define severity levels
            def get_severity(score, scale_type):
                if scale_type == "depression":
                    if score <= 9: return "Normal"
                    elif score <= 13: return "Mild"
                    elif score <= 20: return "Moderate"
                    elif score <= 27: return "Severe"
                    else: return "Extremely Severe"
                elif scale_type == "anxiety":
                    if score <= 7: return "Normal"
                    elif score <= 9: return "Mild"
                    elif score <= 14: return "Moderate"
                    elif score <= 19: return "Severe"
                    else: return "Extremely Severe"
                elif scale_type == "stress":
                    if score <= 14: return "Normal"
                    elif score <= 18: return "Mild"
                    elif score <= 25: return "Moderate"
                    elif score <= 33: return "Severe"
                    else: return "Extremely Severe"
            
            print(f"DEBUG - Final scores: D={depression_final}, A={anxiety_final}, S={stress_final}")
            
            # Create bar chart
            categories = ['Depression', 'Anxiety', 'Stress']
            values = [depression_final, anxiety_final, stress_final]
            severities = [
                get_severity(depression_final, "depression"),
                get_severity(anxiety_final, "anxiety"),
                get_severity(stress_final, "stress")
            ]
            
            print(f"DEBUG - Severities: {severities}")
            
            colors = []
            for severity in severities:
                if severity == "Normal":
                    colors.append("#2E8B57")  # Green
                elif severity == "Mild":
                    colors.append("#FFD700")  # Gold
                elif severity == "Moderate":
                    colors.append("#FF8C00")  # Orange
                elif severity == "Severe":
                    colors.append("#FF4500")  # Red
                else:
                    colors.append("#8B0000")  # Dark Red
            
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=colors,
                    text=[f"{v}<br>({s})" for v, s in zip(values, severities)],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="DASS-21 Assessment Results",
                xaxis_title="Psychological Dimensions",
                yaxis_title="Score",
                template="plotly_white",
                height=400
            )
            
            print("DEBUG - Visualization created successfully")
            return fig
            
        except Exception as e:
            print(f"DEBUG - Visualization error: {str(e)}")
            print(f"DEBUG - Assessment data: {assessment_data}")
            return None

def create_chatbot_interface():
    """Create the Gradio interface"""
    chatbot_instance = MentalHealthChatbot()
    
    def chat_response(message, history):
        """Handle chat response with dual LLM calls"""
        if not message.strip():
            return history, ""
        
        # Add user message to history
        history.append([message, ""])
        
        # Call 1: Get counseling response
        counseling_response = chatbot_instance.get_counseling_response(message)
        
        # Call 2: Get DASS-21 evaluation (in background)
        dass21_evaluation = chatbot_instance.evaluate_dass21(message)
        
        # Update history with counseling response
        history[-1][1] = counseling_response
        
        # Store DASS-21 results for visualization
        chatbot_instance.dass21_results.append({
            "timestamp": datetime.now(),
            "user_message": message,
            "evaluation": dass21_evaluation
        })
        
        return history, ""
    
    def get_dass21_visualization():
        """Get the latest DASS-21 visualization"""
        if not chatbot_instance.dass21_results:
            return None, "No DASS-21 evaluations available yet. Start chatting to generate assessments!"
        
        latest_result = chatbot_instance.dass21_results[-1]
        fig = chatbot_instance.create_dass21_visualization(latest_result["evaluation"])
        
        if fig is None:
            error_msg = "Unable to generate DASS-21 visualization. Check console for debug information."
            if "error" in latest_result["evaluation"]:
                error_msg += f"\nError: {latest_result['evaluation']['error']}"
            return None, error_msg
        
        # Create summary text
        try:
            evaluation = latest_result["evaluation"]
            if "error" in evaluation:
                summary = f"**DASS-21 Evaluation Error**: {evaluation['error']}"
            else:
                scores = evaluation["assessment"]["scores"]
                depression_score = scores.get("depression", {}).get("total_score", 0) or 0
                anxiety_score = scores.get("anxiety", {}).get("total_score", 0) or 0
                stress_score = scores.get("stress", {}).get("total_score", 0) or 0
                
                depression_final = depression_score * 2
                anxiety_final = anxiety_score * 2
                stress_final = stress_score * 2
                
                summary = f"""
**Latest DASS-21 Assessment Summary**
- **Depression Score**: {depression_final}/42 ({depression_score}/21 raw)
- **Anxiety Score**: {anxiety_final}/42 ({anxiety_score}/21 raw)  
- **Stress Score**: {stress_final}/42 ({stress_score}/21 raw)
- **Assessment Time**: {latest_result["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}

*Note: DASS-21 scoring: Raw scores (0-21) are doubled for final scores (0-42) as per standard guidelines.*
                """
        except Exception as e:
            summary = f"Assessment completed but unable to generate detailed summary. Error: {str(e)}"
        
        return fig, summary
    
    def clear_conversation():
        """Clear conversation history and DASS-21 results"""
        chatbot_instance.conversation_history = []
        chatbot_instance.dass21_results = []
        return [], None, "Conversation and assessments cleared."
    
    # Create Gradio interface
    with gr.Blocks(
        title="Multilingual Mental Health Chatbot with DASS-21",
        theme=gr.themes.Soft(),
        css="""
        .chatbot-container { height: 500px; }
        .dass21-container { height: 450px; }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🧠 Multilingual Mental Health Chatbot with DASS-21 Assessment
        
        **Features:**
        - 💬 Culturally-sensitive counseling responses in 5 languages (EN, VI, AR, ZH-CN, ZH-HK)
        - 📊 Real-time DASS-21 psychological assessment
        - 🎯 Powered by fine-tuned Qwen3-4B model (v3.2)
        - 🔒 Privacy-focused on-premise deployment
        
        **Instructions:** Share your thoughts and feelings. The AI will provide supportive responses while automatically evaluating your mental health status using the DASS-21 scale.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Mental Health Counselor",
                    elem_classes=["chatbot-container"],
                    height=500,
                    show_label=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Share what's on your mind...",
                        label="Your Message",
                        lines=2,
                        max_lines=5
                    )
                    
                with gr.Row():
                    submit_btn = gr.Button("Send Message", variant="primary")
                    clear_btn = gr.Button("Clear Conversation", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📈 DASS-21 Assessment")
                
                dass21_plot = gr.Plot(
                    label="Latest Assessment Results",
                    elem_classes=["dass21-container"]
                )
                
                dass21_summary = gr.Markdown(
                    value="Start chatting to see your DASS-21 assessment results here.",
                    label="Assessment Summary"
                )
                
                refresh_btn = gr.Button("Refresh Assessment", variant="secondary")
        
        # Event handlers
        msg.submit(
            chat_response,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        ).then(
            get_dass21_visualization,
            outputs=[dass21_plot, dass21_summary]
        )
        
        submit_btn.click(
            chat_response,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        ).then(
            get_dass21_visualization,
            outputs=[dass21_plot, dass21_summary]
        )
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, dass21_plot, dass21_summary]
        )
        
        refresh_btn.click(
            get_dass21_visualization,
            outputs=[dass21_plot, dass21_summary]
        )
        
        gr.Markdown("""
        ---
        **Disclaimer:** This AI chatbot is designed for research and educational purposes. It should not replace professional mental health care. If you're experiencing a mental health crisis, please contact emergency services or a mental health professional immediately.
        
        **Model:** Qwen3-4B v3.2 (LoRA fine-tuned) | **Assessment:** DASS-21 (Depression, Anxiety, Stress Scales)
        """)
    
    return interface

if __name__ == "__main__":
    # Configuration
    VLLM_URL = "http://localhost:8000"  # Adjust based on your VLLM server
    
    print("🚀 Starting Multilingual Mental Health Chatbot...")
    print(f"📡 VLLM API URL: {VLLM_URL}")
    print("🔄 Testing VLLM connection...")
    
    # Test VLLM connection
    try:
        response = requests.get(f"{VLLM_URL}/health")
        if response.status_code == 200:
            print("✅ VLLM server is running!")
        else:
            print("⚠️  VLLM server may not be ready")
    except:
        print("❌ Cannot connect to VLLM server. Please ensure it's running.")
    
    # Create and launch interface
    interface = create_chatbot_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Set to True if you want a public link
        debug=True
    )
