# Configuration for Multilingual Mental Health Chatbot
# Update these values according to your setup

# VLLM Server Configuration
VLLM_BASE_URL = "http://localhost:8000"
MODEL_NAME = "unsloth/Qwen3-4B"  # Your best model (v3.2)

# Model Paths (update these to your actual paths)
BASE_MODEL_PATH = "unsloth/Qwen3-4B"
LORA_ADAPTER_PATH = "trained_model_v3_2_grpo"  # Update this path

# Gradio Configuration
GRADIO_HOST = "0.0.0.0"
GRADIO_PORT = 7860
SHARE_GRADIO = False  # Set to True for public sharing

# Model Inference Parameters
MAX_TOKENS_COUNSELING = 4096
MAX_TOKENS_DASS21 = 2048
TEMPERATURE = 0.8
TOP_P = 0.9

# DASS-21 Scoring Multiplier (standard is 2 for DASS-21)
DASS21_SCORE_MULTIPLIER = 2

# UI Configuration
CHATBOT_HEIGHT = 500
ASSESSMENT_HEIGHT = 450

# Logging
ENABLE_LOGGING = True
LOG_CONVERSATIONS = False  # Set to True to log conversations (be mindful of privacy)
