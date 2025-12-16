"""Converse Mode page (placeholder for future development)"""

import streamlit as st
from utils.ollama_client import OllamaClient
from utils.model_manager import ModelManager

def render():
    """Render Converse Mode interface"""
    
    if not st.session_state.selected_model:
        st.warning("⚠️ Please select a model from the sidebar first")
        return
    
    model_name = st.session_state.selected_model
    
    st.header("Converse Mode")
    st.info("🚧 This mode is a placeholder for future development. Use Model Configuration > Learning Session for now.")
    
    # Get base model and check for fine-tuned models
    model_manager = ModelManager()
    metadata = model_manager.get_model_metadata(model_name)
    base_model = metadata.get("base_model", "llama2") if metadata else "llama2"
    
    # Always try to use fine-tuned model first
    fine_tuned_path = model_manager.get_fine_tuned_model_path(model_name)
    has_fine_tuned = fine_tuned_path is not None
    
    # Initialize client - always prefer fine-tuned model
    client = None
    use_fine_tuned = False
    
    # REMOTE-ONLY MODE: Check for remote inference server
    remote_inference_url = None
    if 'active_job' in st.session_state and st.session_state.active_job:
        active_job = st.session_state.active_job
        remote_inference_ready = active_job.get("inference_ready", False)
        remote_inference_url = active_job.get("inference_url", "")
        if remote_inference_url and remote_inference_ready:
            # Verify the URL is accessible
            try:
                import requests
                health_check = requests.get(f"{remote_inference_url}/health", timeout=5)
                if health_check.status_code != 200:
                    remote_inference_url = None
            except:
                remote_inference_url = None
    
    if remote_inference_url:
        try:
            from utils.fine_tuned_client import FineTunedModelClient
            # Use remote server - determine which model to use
            # Try to use fine-tuned version if available, otherwise base
            model_for_remote = "v1" if has_fine_tuned else "base"
            client = FineTunedModelClient(remote_url=remote_inference_url, remote_model=model_for_remote)
            use_fine_tuned = True
            st.success(f"✅ Using remote GPU inference server (model: {model_for_remote})")
        except Exception as e:
            st.error(f"❌ Failed to connect to remote server: {str(e)}")
            st.info("💡 Please set up the remote inference server in Phase 4 of the training workflow.")
            use_fine_tuned = False
            client = None
    elif has_fine_tuned:
        # Local mode disabled - show error
        st.error("❌ **Remote inference server required**")
        st.markdown("""
        This application is configured for **remote-only mode**.
        
        **To use fine-tuned models:**
        1. Complete training (Phase 3)
        2. Go to Phase 4: Finalize  
        3. Click "🔧 Setup Inference Server" to deploy the server on your Vast.ai instance
        4. Once the server is running, you can use this mode
        """)
        use_fine_tuned = False
        client = None
    
    if not use_fine_tuned:
        ollama_client = OllamaClient(base_model)
        
        # Check if model is available
        if not ollama_client.model_exists():
            st.error(f"⚠️ Model '{base_model}' is not available in Ollama.")
            st.markdown(f"**To fix this:**")
            st.code(f"ollama pull {base_model}", language="bash")
            
            available_models = ollama_client.get_available_models()
            if available_models:
                st.info(f"**Available models:** {', '.join(available_models)}")
            else:
                st.warning("No models found in Ollama. Please install at least one model.")
            
            st.stop()
        
        client = ollama_client
    
    # Simple chat interface
    st.markdown("### Chat with your model")
    
    # Prepend text configuration (collapsible)
    with st.expander("⚙️ Prompt Settings", expanded=False):
        prepend_key = f"prepend_text_{model_name}"
        if prepend_key not in st.session_state:
            st.session_state[prepend_key] = ""
        
        prepend_text = st.text_area(
            "Prepend Text (added invisibly to all prompts)",
            value=st.session_state[prepend_key],
            help="This text will be prepended to all prompts before sending to the model. It's not visible in the chat but affects all responses.",
            key=f"prepend_textarea_{model_name}",
            height=100
        )
        st.session_state[prepend_key] = prepend_text
        
        include_summary = st.checkbox(
            "Include conversation summary request",
            value=st.session_state.get(f"include_summary_{model_name}", False),
            help="If checked, the model will be asked to attach a summary of the conversation at the end of each response, marked with ###SUMMARY###",
            key=f"include_summary_checkbox_{model_name}"
        )
        st.session_state[f"include_summary_{model_name}"] = include_summary
    
    # Initialize chat history
    if 'converse_history' not in st.session_state:
        st.session_state.converse_history = []
    
    # Display chat history
    for message in st.session_state.converse_history:
        with st.chat_message(message["role"]):
            # Strip summary section if present in assistant messages
            content = message["content"]
            if message["role"] == "assistant" and "###SUMMARY###" in content:
                content = content.split("###SUMMARY###")[0].strip()
            st.write(content)
    
    # Ensure client is initialized
    if client is None:
        st.error("❌ Failed to initialize model client. Please check your model configuration.")
        st.stop()
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Add user message (without prepend text - that's added invisibly)
        st.session_state.converse_history.append({"role": "user", "content": user_input})
        
        # Prepare messages for model (with prepend text and summary request)
        messages = []
        
        # Get prepend text and summary setting
        prepend_key = f"prepend_text_{model_name}"
        prepend_text = st.session_state.get(prepend_key, "")
        include_summary = st.session_state.get(f"include_summary_{model_name}", False)
        
        # Get the index of the last message (the current user input)
        last_message_idx = len(st.session_state.converse_history) - 1
        
        for idx, msg in enumerate(st.session_state.converse_history):
            # Ensure role is valid (user or assistant)
            role = msg.get("role", "user")
            if role not in ["user", "assistant"]:
                role = "user"  # Default to user if invalid
            
            content = str(msg.get("content", ""))
            
            # For user messages, prepend the prepend text (invisibly)
            # Add summary request only to the latest user message if enabled
            if role == "user":
                # Build the full content with prepend and summary request
                full_content_parts = []
                
                # Add prepend text first (if exists) - to all user messages
                if prepend_text:
                    full_content_parts.append(prepend_text)
                
                # Add the original user message
                full_content_parts.append(content)
                
                # Add summary request if enabled AND this is the latest user message
                if include_summary and idx == last_message_idx:
                    # Create a summary of the conversation so far (up to this message)
                    conversation_summary = ""
                    for prev_msg in st.session_state.converse_history[:idx]:  # All messages before this one
                        prev_role = prev_msg.get("role", "user")
                        prev_content = str(prev_msg.get("content", ""))
                        if prev_role == "user":
                            conversation_summary += f"User: {prev_content}\n"
                        elif prev_role == "assistant":
                            # Strip any existing summaries from previous responses
                            prev_content_clean = prev_content.split("###SUMMARY###")[0].strip()
                            conversation_summary += f"Assistant: {prev_content_clean}\n"
                    
                    summary_request = "\n\nPlease attach a summary of the conversation so far to the end of your response, set off with the text '###SUMMARY###'. The conversation so far:\n" + conversation_summary
                    full_content_parts.append(summary_request)
                
                # Join all parts
                content = "\n\n".join(full_content_parts)
            
            messages.append({
                "role": role,
                "content": content
            })
        
        # Get response from model
        with st.spinner("Thinking..."):
            if use_fine_tuned:
                response = client.chat(messages, max_length=1024, temperature=0.7)
            else:
                response = client.chat(messages)
        
        # Strip summary section if present
        if response and isinstance(response, str):
            if "###SUMMARY###" in response:
                parts = response.split("###SUMMARY###")
                main_response = parts[0].strip()
                # Store the summary separately if needed (for future use)
                summary = parts[1].strip() if len(parts) > 1 else ""
                response = main_response
        
        # Add assistant response (with summary stripped for display)
        st.session_state.converse_history.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    if st.button("Clear Chat", key="clear_converse"):
        st.session_state.converse_history = []
        st.rerun()


