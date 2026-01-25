"""Streamlit demo for Visual Reasoning Systems."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import base64

from src.models import ModelFactory
from src.data.vqa_dataset import VQADataset
from src.utils.device import get_device, set_seed
from src.utils.config import load_config


# Page configuration
st.set_page_config(
    page_title="Visual Reasoning Systems",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 Visual Reasoning Systems")
st.markdown("""
This demo showcases advanced Visual Question Answering (VQA) models that can understand images and answer questions about them.
The system uses state-of-the-art vision-language models like VisualBERT and CLIP-based architectures.
""")

# Sidebar for model selection
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["visualbert", "clip_vqa"],
    help="Choose between different VQA model architectures"
)

# Load configuration
@st.cache_resource
def load_model_config():
    """Load model configuration."""
    config = load_config("configs/config.yaml")
    config.model.name = model_type
    return config

config = load_model_config()

# Initialize model
@st.cache_resource
def load_model():
    """Load the selected model."""
    set_seed(42)
    device = get_device("auto")
    model = ModelFactory.create_model(config.model)
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Upload Image")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to ask questions about"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert image to tensor format expected by model
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Question input
        st.header("❓ Ask a Question")
        question = st.text_input(
            "Enter your question about the image:",
            placeholder="e.g., What color is the car? How many people are in the image?",
            help="Ask any question about the uploaded image"
        )
        
        # Tokenize question
        if question:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            question_tokens = tokenizer(
                question,
                max_length=50,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            question_input_ids = question_tokens["input_ids"].to(device)
            question_attention_mask = question_tokens["attention_mask"].to(device)

with col2:
    st.header("🤖 Model Response")
    
    if uploaded_file is not None and question:
        # Generate answer
        with st.spinner("Processing your question..."):
            try:
                # Get model prediction
                with torch.no_grad():
                    outputs = model(
                        images=image_tensor,
                        question_input_ids=question_input_ids,
                        question_attention_mask=question_attention_mask
                    )
                    
                    logits = outputs["logits"]
                    predicted_class = torch.argmax(logits, dim=-1).item()
                
                # Map prediction to answer
                answer_map = {0: "No", 1: "Yes"}
                answer = answer_map.get(predicted_class, "Unknown")
                
                # Display results
                st.success(f"**Answer:** {answer}")
                
                # Show confidence scores
                probabilities = torch.softmax(logits, dim=-1)
                confidence = probabilities[0][predicted_class].item()
                
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Show all class probabilities
                st.subheader("📊 Prediction Probabilities")
                prob_data = {
                    "Class": ["No", "Yes"],
                    "Probability": [probabilities[0][0].item(), probabilities[0][1].item()]
                }
                st.bar_chart(prob_data)
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
    
    else:
        st.info("Please upload an image and enter a question to get started.")

# Model information
st.header("ℹ️ Model Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Type", model_type.upper())
    st.metric("Device", str(device))

with col2:
    from src.utils.device import get_model_size
    model_info = get_model_size(model)
    st.metric("Parameters", f"{model_info['total_parameters_millions']:.1f}M")
    st.metric("Trainable", f"{model_info['trainable_parameters_millions']:.1f}M")

with col3:
    st.metric("Model Status", "Ready")
    st.metric("Input Size", "224x224")

# Example questions
st.header("💡 Example Questions")
st.markdown("""
Try asking questions like:
- **Object recognition:** "What objects do you see in the image?"
- **Counting:** "How many people are in the image?"
- **Colors:** "What color is the car?"
- **Actions:** "What is the person doing?"
- **Spatial relations:** "Is the dog on the left or right side?"
- **Yes/No questions:** "Is there a cat in the image?"
""")

# Technical details
with st.expander("🔧 Technical Details"):
    st.markdown(f"""
    **Model Architecture:** {model_type}
    **Input Resolution:** 224x224 pixels
    **Question Length:** Up to 50 tokens
    **Answer Format:** Binary classification (Yes/No)
    **Device:** {device}
    **Framework:** PyTorch
    """)

# Footer
st.markdown("---")
st.markdown("""
**Visual Reasoning Systems** - Advanced Computer Vision Project  
Built with PyTorch, Transformers, and Streamlit
""")
