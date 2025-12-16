"""
Anvil: A Custom AI Model Environment
Main entry point for the application
"""

import streamlit as st
import os
from pathlib import Path
from ui.pages import training_mode, converse_mode
from utils.model_manager import ModelManager
from utils.config import ensure_directories

# Page configuration
st.set_page_config(
    page_title="Anvil - Custom AI Models",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve sidebar layout
st.markdown("""
<style>
    /* Sidebar base styling */
    section[data-testid="stSidebar"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Remove top spacing from sidebar content */
    section[data-testid="stSidebar"] > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Logo container styling */
    #anvil-logo-container {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    #anvil-logo-container img {
        max-width: 80% !important;
        height: auto !important;
        display: block !important;
        margin: 0 auto !important;
    }
    
    /* Ensure proper spacing for sidebar elements */
    section[data-testid="stSidebar"] .stMarkdown {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* First markdown (logo) - zero top margin and padding */
    section[data-testid="stSidebar"] .stMarkdown:first-of-type {
        margin-top: 0 !important;
        margin-bottom: 0.1rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Override any Streamlit default spacing on first element */
    section[data-testid="stSidebar"] > div > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Horizontal rule spacing */
    section[data-testid="stSidebar"] hr {
        margin-top: 0.75rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Subheader spacing */
    section[data-testid="stSidebar"] .stSubheader {
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0.5rem !important;
    }
    
    /* Prevent text overlap */
    section[data-testid="stSidebar"] * {
        line-height: 1.5 !important;
    }
    
    /* Caption spacing */
    section[data-testid="stSidebar"] .stCaption {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Ensure directories exist
ensure_directories()

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

if 'mode' not in st.session_state:
    st.session_state.mode = None

def main():
    """Main application entry point"""
    
    # Sidebar for model selection and navigation
    with st.sidebar:
        # Display logo if available
        from utils.config import ASSETS_DIR
        logo_path = ASSETS_DIR / "logo.png"
        if logo_path.exists():
            import base64
            with open(logo_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            st.markdown(
                f'<div id="anvil-logo-container"><img src="data:image/png;base64,{img_data}" alt="Anvil Logo"></div>',
                unsafe_allow_html=True
            )
        else:
            st.title("🔨 Anvil")
        st.markdown("<hr style='margin-top: 0.1rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        
        # Model Profile Selection
        st.subheader("Model Profile Selection")
        model_manager = st.session_state.model_manager
        
        # Get available models (non-archived)
        available_models = model_manager.get_available_models()
        
        if available_models:
            selected = st.selectbox(
                "Select Model Profile",
                options=available_models,
                index=0 if available_models else None,
                key="model_selector"
            )
            st.session_state.selected_model = selected
        else:
            st.info("No models available. Create a new model profile.")
        
        # Create new model
        with st.expander("Create New Model Profile"):
            new_model_name = st.text_input("Model Name", key="new_model_name")
            
            # Curated list of HuggingFace models good for writing and general prompting
            # All support LoRA fine-tuning
            hf_models = [
                {
                    "name": "Gemma 3 4B (Recommended)",
                    "hf_id": "google/gemma-3-4b-it",
                    "description": "Excellent for writing, general tasks. 4B parameters, fast inference."
                },
                {
                    "name": "Llama 3.1 8B Instruct",
                    "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "description": "Strong general-purpose model, great for writing and reasoning."
                },
                {
                    "name": "Llama 3.2 3B Instruct",
                    "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
                    "description": "Smaller, faster model. Good balance of quality and speed."
                },
                {
                    "name": "Mistral 7B Instruct",
                    "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
                    "description": "Excellent instruction following, great for structured tasks."
                },
                {
                    "name": "Qwen2.5 7B Instruct",
                    "hf_id": "Qwen/Qwen2.5-7B-Instruct",
                    "description": "Strong multilingual support, good for diverse content."
                },
                {
                    "name": "Phi-3 Mini 4K",
                    "hf_id": "microsoft/Phi-3-mini-4k-instruct",
                    "description": "Compact model, efficient for smaller tasks."
                },
                {
                    "name": "Gemma 2 9B IT",
                    "hf_id": "google/gemma-2-9b-it",
                    "description": "Larger Gemma model, more capable for complex tasks."
                },
                {
                    "name": "Llama 3 8B Instruct",
                    "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "description": "Solid general-purpose model, well-tested."
                },
            ]
            
            # Create display options with descriptions
            model_options = [f"{m['name']} ({m['hf_id']})" for m in hf_models]
            
            # Find default index (Gemma 3 4B)
            default_index = 0
            for i, m in enumerate(hf_models):
                if "gemma-3-4b-it" in m["hf_id"]:
                    default_index = i
                    break
            
            selected_model_display = st.selectbox(
                "Base Model (HuggingFace)",
                options=model_options,
                index=default_index,
                key="base_model_selector",
                help="Select a HuggingFace model. All models support LoRA fine-tuning and work well for writing and general tasks."
            )
            
            # Extract HuggingFace ID from selection
            selected_index = model_options.index(selected_model_display)
            selected_model_info = hf_models[selected_index]
            base_model_hf_id = selected_model_info["hf_id"]
            
            # Show model description
            st.caption(f"💡 {selected_model_info['description']}")
            
            # Allow manual override for advanced users
            with st.expander("🔧 Advanced: Custom HuggingFace Model", expanded=False):
                custom_hf_model = st.text_input(
                    "Custom HuggingFace Model ID",
                    value="",
                    placeholder="e.g., microsoft/Phi-3-medium-4k-instruct",
                    key="custom_hf_model",
                    help="Override with any HuggingFace model ID. Must support LoRA/PEFT."
                )
                if custom_hf_model:
                    base_model_hf_id = custom_hf_model
                    st.info(f"Using custom model: {base_model_hf_id}")
            
            # Store the HuggingFace ID as the base_model (for backward compatibility with existing code)
            # The system will map this to the actual HF model when needed
            base_model = base_model_hf_id
            
            if st.button("Create Model Profile", key="create_model_btn"):
                if new_model_name:
                    if base_model:
                        model_manager.create_model_profile(new_model_name, base_model)
                        st.success(f"Model profile '{new_model_name}' created with base model: {base_model_hf_id}")
                        st.rerun()
                    else:
                        st.error("Please select a base model")
                else:
                    st.error("Please enter a model name")
        
        st.markdown("---")
    
    # Main content area - always show model configuration (with tabs)
    if st.session_state.selected_model:
        training_mode.render()
    else:
        st.info("👈 Please select a model profile from the sidebar to begin")

if __name__ == "__main__":
    main()


