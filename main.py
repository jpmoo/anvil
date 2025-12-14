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
        padding-top: 1rem !important;
    }
    
    /* Logo container styling */
    #anvil-logo-container {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0.5rem 0 !important;
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
    
    /* First markdown (logo) - minimal top margin */
    section[data-testid="stSidebar"] .stMarkdown:first-of-type {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
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
        st.markdown("<hr style='margin-top: 0.25rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
        
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
            
            # Get installed Ollama models
            from utils.ollama_client import OllamaClient
            ollama_client = OllamaClient()
            installed_models = ollama_client.get_available_models()
            
            # Filter to get base model names (remove duplicates and tags)
            base_models = []
            seen = set()
            for model in installed_models:
                base = model.split(':')[0] if ':' in model else model
                if base.lower() not in seen:
                    base_models.append(base)
                    seen.add(base.lower())
            
            if base_models:
                base_model = st.selectbox(
                    "Base Model",
                    options=sorted(base_models),
                    key="base_model_selector",
                    help="Select from your installed Ollama models"
                )
            else:
                st.warning("No Ollama models found. Please install at least one model first.")
                base_model = st.text_input("Base Model Name", key="base_model_text", placeholder="e.g., llama2")
            
            if st.button("Create Model Profile", key="create_model_btn"):
                if new_model_name:
                    if base_model:
                        model_manager.create_model_profile(new_model_name, base_model)
                        st.success(f"Model profile '{new_model_name}' created!")
                        st.rerun()
                    else:
                        st.error("Please select or enter a base model")
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


