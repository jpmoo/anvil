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

# Custom CSS to reduce sidebar padding and improve layout
st.markdown("""
<style>
    /* Remove all top padding from sidebar */
    section[data-testid="stSidebar"] {
        padding-top: 0 !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 0 !important;
    }
    
    /* Remove padding from sidebar content container */
    .css-1d391kg {
        padding-top: 0 !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Remove all margins from logo/image */
    .stImage {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    .stImage > img {
        margin-top: 0 !important;
        margin-bottom: 0.25rem !important;
        padding: 0 !important;
    }
    
    /* Scale logo to 50% and center - more specific selector */
    section[data-testid="stSidebar"] .stImage {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    section[data-testid="stSidebar"] .stImage img {
        max-width: 50% !important;
        width: 50% !important;
        height: auto !important;
        margin: 0 auto !important;
    }
    
    /* Remove padding from sidebar header area */
    .css-1lcbmhc .css-1d391kg {
        padding-top: 0 !important;
    }
    
    /* Target the first element in sidebar to remove top spacing */
    section[data-testid="stSidebar"] > div > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Center icon buttons - target Streamlit buttons specifically */
    button[data-testid*="baseButton"],
    button[data-testid*="button"],
    button.stButton > button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* Center button content - target Streamlit's button structure */
    button[data-testid*="baseButton"] > p,
    button[data-testid*="button"] > p,
    button.stButton > button > p,
    button > p {
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
        line-height: 1 !important;
        text-indent: 0 !important;
    }
    
    /* Target the button wrapper */
    .stButton > button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0.25rem 0.5rem !important;
    }
    
    .stButton > button > p {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        text-align: center !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
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
        # Display logo if available (with minimal padding)
        from utils.config import ASSETS_DIR
        logo_path = ASSETS_DIR / "logo.png"
        if logo_path.exists():
            # Use HTML to display logo at 62.5% size (25% bigger than 50%)
            import base64
            with open(logo_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            st.markdown(
                f'<div style="display: flex; justify-content: center; margin-bottom: 0.25rem;"><img src="data:image/png;base64,{img_data}" style="width: 78.125%; height: auto;"></div>',
                unsafe_allow_html=True
            )
        else:
            st.title("🔨 Anvil")
        st.markdown("---")
        
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


