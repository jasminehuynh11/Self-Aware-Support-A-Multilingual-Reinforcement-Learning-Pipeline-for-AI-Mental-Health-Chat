#!/bin/bash

# Setup script for Multilingual Mental Health Chatbot
# This script helps you set up and run the VLLM server and Gradio app

echo "🚀 Setting up Multilingual Mental Health Chatbot..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Install requirements
echo "📦 Installing Python requirements..."
pip install -r chatbot_requirements.txt

# Install VLLM if not present
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "📦 Installing VLLM..."
    pip install vllm
fi

echo "✅ Requirements installed"

# Configuration
MODEL_PATH="unsloth/Qwen3-4B"  # Update this to your model path
LORA_PATH="trained_model_v3_2_grpo"  # Update this to your LoRA adapter path if needed
PORT=8000

echo "🔧 Configuration:"
echo "   Model: $MODEL_PATH"
echo "   Port: $PORT"

# Function to start VLLM server
start_vllm() {
    echo "🚀 Starting VLLM server..."
    
    if [ -n "$LORA_PATH" ]; then
        # With LoRA adapter
        python -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --enable-lora \
            --lora-modules qwen3-lora=$LORA_PATH \
            --port $PORT \
            --host 0.0.0.0 \
            --trust-remote-code &
    else
        # Base model only
        python -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --port $PORT \
            --host 0.0.0.0 \
            --trust-remote-code &
    fi
    
    VLLM_PID=$!
    echo "🔄 VLLM server starting with PID: $VLLM_PID"
    
    # Wait for server to be ready
    echo "⏳ Waiting for VLLM server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "✅ VLLM server is ready!"
            break
        fi
        echo "   Attempt $i/30... waiting 2 seconds"
        sleep 2
    done
    
    if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "❌ VLLM server failed to start properly"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
}

# Function to start Gradio app
start_gradio() {
    echo "🌐 Starting Gradio chatbot application..."
    python gradio_chatbot_app.py
}

# Function to cleanup
cleanup() {
    echo "🧹 Cleaning up..."
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null
        echo "   VLLM server stopped"
    fi
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Main execution
case "${1:-both}" in
    "vllm")
        start_vllm
        wait
        ;;
    "gradio")
        start_gradio
        ;;
    "both"|*)
        start_vllm
        sleep 3
        start_gradio
        ;;
esac
