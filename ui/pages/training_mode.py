"""Model Configuration page"""

import streamlit as st
import json
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from utils.training_data import TrainingDataManager
from utils.fine_tuner import FineTuner
from utils.model_status import ModelStatus
from utils.ollama_client import OllamaClient
from utils.config import get_model_context_dir, get_model_behavioral_path, get_model_preferences, save_model_preferences, get_hf_token


def filter_malloc_warnings(text):
    """
    Filter out macOS MallocStackLogging warnings from subprocess output.
    These are harmless warnings that clutter the terminal output.
    """
    if not text:
        return text
    
    lines = text.split('\n')
    filtered_lines = [
        line for line in lines 
        if 'MallocStackLogging' not in line
    ]
    return '\n'.join(filtered_lines)


def extract_dataset_stats(log_content: str) -> dict:
    """
    Extract dataset statistics from Axolotl training logs
    
    Args:
        log_content: Full training log content
        
    Returns:
        Dictionary with dataset statistics
    """
    import re
    stats = {}
    
    if not log_content:
        return stats
    
    lines = log_content.split('\n')
    
    # Look for patterns in the logs
    # Pattern 1: "Dropping Long Sequences" progress - extract final count
    # Pattern 2: "Drop Samples with Zero Trainable Tokens" progress
    # Pattern 3: Dataset size information
    # Pattern 4: Training dataset summary
    
    # Try to find original count from "Dropping Long Sequences" progress bar
    # Format: "Dropping Long Sequences (>2048) (num_proc=192): 100%|██████████| 223/223"
    dropping_pattern = r'Dropping Long Sequences.*?(\d+)/(\d+)'
    for line in lines:
        match = re.search(dropping_pattern, line)
        if match:
            # The second number is usually the total
            stats['original_count'] = int(match.group(2))
            break
    
    # Look for "Drop Samples with Zero Trainable Tokens" final count
    zero_tokens_pattern = r'Drop Samples with Zero Trainable Tokens.*?(\d+)/(\d+)'
    for line in lines:
        match = re.search(zero_tokens_pattern, line)
        if match:
            # Update original count if found, or use this as reference
            if 'original_count' not in stats:
                stats['original_count'] = int(match.group(2))
            break
    
    # Look for dataset size information
    # Patterns like: "Dataset size: 200", "Training on 200 examples", "num_train_examples: 200"
    # Also look for Axolotl-specific patterns after filtering
    dataset_size_patterns = [
        r'[Dd]ataset\s+[Ss]ize[:\s]+(\d+)',
        r'[Tt]raining\s+on\s+(\d+)\s+examples?',
        r'num_train_examples[:\s]+(\d+)',
        r'train.*?examples?[:\s]+(\d+)',
        r'Total.*?samples?[:\s]+(\d+)',
        r'Found\s+(\d+)\s+examples?',
        r'(\d+)\s+examples?\s+after\s+filtering',
        r'(\d+)\s+examples?\s+remaining',
        r'(\d+)\s+examples?\s+will\s+be\s+used',
    ]
    
    for pattern in dataset_size_patterns:
        for line in lines:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                stats['final_count'] = int(match.group(1))
                break
        if 'final_count' in stats:
            break
    
    # Try to extract from progress bar completion - look for the last number after filtering steps
    # After "Drop Samples with Zero Trainable Tokens" completes, look for subsequent dataset info
    # Or look for lines that mention the count after all filtering
    if 'final_count' not in stats:
        # Look for lines after the filtering steps that mention dataset size
        # Axolotl often outputs something like "Dataset has X examples" or "X train examples"
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Skip the filtering progress bars themselves
            if 'dropping long' in line_lower or 'drop samples' in line_lower or 'group by length' in line_lower:
                continue
            # Look for dataset size mentions after filtering
            if any(keyword in line_lower for keyword in ['dataset', 'train', 'example', 'sample']) and 'size' in line_lower:
                match = re.search(r'(\d+)\s*(?:examples?|samples?)', line_lower)
                if match:
                    potential_count = int(match.group(1))
                    # Only accept if it's reasonable (less than or equal to original)
                    if 'original_count' in stats:
                        if potential_count <= stats['original_count']:
                            stats['final_count'] = potential_count
                            break
                    else:
                        stats['final_count'] = potential_count
                        break
    
    # Look for dropped counts
    dropped_patterns = [
        r'(\d+)\s+samples?\s+dropped',
        r'[Dd]ropped[:\s]+(\d+)',
        r'[Ff]iltered[:\s]+(\d+)',
    ]
    
    total_dropped = 0
    for pattern in dropped_patterns:
        for line in lines:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                dropped = int(match.group(1))
                if dropped > total_dropped:
                    total_dropped = dropped
    
    # Calculate dropped counts if we have original and final
    if 'original_count' in stats and 'final_count' in stats:
        calculated_dropped = stats['original_count'] - stats['final_count']
        if calculated_dropped > 0:
            stats['total_dropped'] = calculated_dropped
    
    # Try to find specific drop reasons - be careful not to confuse sequence length with drop count
    # Look for lines mentioning "long sequences" or "zero tokens" but avoid sequence length numbers
    for line in lines:
        line_lower = line.lower()
        if ('long sequence' in line_lower or 'dropping long' in line_lower) and '>' in line:
            # This is the "Dropping Long Sequences (>2048)" line - don't extract 2048 as drop count
            # Look for actual drop counts in summary lines
            pass
        elif 'dropped' in line_lower and ('long' in line_lower or 'sequence' in line_lower):
            # Look for actual drop counts
            match = re.search(r'(\d+)\s+(?:samples?|examples?)\s+dropped', line_lower)
            if match:
                try:
                    stats['dropped_long'] = int(match.group(1))
                except:
                    pass
    
    return stats


def render():
    """Render Model Configuration interface"""
    # Explicitly declare we're using the module-level subprocess import
    # This prevents Python from treating it as a local variable when local imports exist
    global subprocess
    
    if not st.session_state.selected_model:
        st.warning("⚠️ Please select a model from the sidebar first")
        return
    
    # Initialize HF token in session state if not already set
    if 'hf_token' not in st.session_state:
        from utils.config import get_hf_token
        st.session_state['hf_token'] = get_hf_token()
    
    model_name = st.session_state.selected_model
    
    # Get model metadata for title
    from utils.model_manager import ModelManager
    model_manager = ModelManager()
    metadata = model_manager.get_model_metadata(model_name)
    model_title = metadata.get("name", model_name) if metadata else model_name
    
    st.header(model_title)
    
    # Tab selection - removed Fine-Tuning, using rule-based system instead
    # Sidebar: API Key Configuration
    with st.sidebar:
        st.markdown("## 🔑 Vast.ai API Key")
        from utils.config import get_vast_api_key, save_vast_api_key, delete_vast_api_key
        
        # Load saved API key
        saved_key = get_vast_api_key()
        
        if saved_key:
            # Key is saved - show status and delete option
            st.caption(f"✅ Saved (ends with: {saved_key[-4:] if len(saved_key) > 4 else '***'})")
            
            # Delete key button with confirmation
            delete_confirm_key = "delete_api_key_confirm"
            if delete_confirm_key not in st.session_state:
                st.session_state[delete_confirm_key] = False
            
            if not st.session_state[delete_confirm_key]:
                if st.button("🗑️ Delete API Key", key="delete_api_key", type="secondary", use_container_width=True):
                    st.session_state[delete_confirm_key] = True
                    st.rerun()
            else:
                st.warning("⚠️ Are you sure you want to delete the API key?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Confirm", key="confirm_delete_api_key", type="primary", use_container_width=True):
                        delete_vast_api_key()
                        if "vast_api_key" in st.session_state:
                            del st.session_state["vast_api_key"]
                        st.session_state[delete_confirm_key] = False
                        st.success("✅ API key deleted")
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", key="cancel_delete_api_key", use_container_width=True):
                        st.session_state[delete_confirm_key] = False
                        st.rerun()
            
            # Use saved key
            vast_api_key = saved_key
            st.session_state.vast_api_key = vast_api_key
        else:
            # No key saved - show input box
            vast_api_key = st.text_input(
                "API Key",
                type="password",
                value="",  # Don't show the actual key in the input
                help="Enter your Vast.ai API key to launch training jobs. It will be saved permanently.",
                key="vast_api_key_input",
                placeholder="Enter API key"
            )
            
            if vast_api_key:
                # Save the key permanently
                save_vast_api_key(vast_api_key)
                st.session_state.vast_api_key = vast_api_key
                st.success("✅ Saved!")
                st.rerun()
            
            # Try environment variable as fallback
            if not vast_api_key:
                import os
                vast_api_key = os.getenv("VAST_API_KEY", "")
                if vast_api_key:
                    st.caption("ℹ️ Using environment variable")
                    st.session_state.vast_api_key = vast_api_key
        
        st.caption("Find your API key in your [Vast.ai account settings](https://vast.ai/account)")
        st.markdown("---")
        
        # Hugging Face Token Configuration
        st.markdown("## 🤗 Hugging Face Token")
        from utils.config import get_hf_token, save_hf_token, delete_hf_token
        
        # Load saved token
        saved_token = get_hf_token()
        
        if saved_token:
            # Token is saved - show status and delete option
            st.caption(f"✅ Saved (ends with: {saved_token[-4:] if len(saved_token) > 4 else '***'})")
            
            # Delete token button with confirmation
            delete_confirm_hf_token = "delete_hf_token_confirm"
            if delete_confirm_hf_token not in st.session_state:
                st.session_state[delete_confirm_hf_token] = False
            
            if not st.session_state[delete_confirm_hf_token]:
                if st.button("🗑️ Delete Token", key="delete_hf_token", type="secondary", use_container_width=True):
                    st.session_state[delete_confirm_hf_token] = True
                    st.rerun()
            else:
                st.warning("⚠️ Are you sure you want to delete the token?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Confirm", key="confirm_delete_hf_token", type="primary", use_container_width=True):
                        delete_hf_token()
                        if "hf_token" in st.session_state:
                            del st.session_state["hf_token"]
                        st.session_state[delete_confirm_hf_token] = False
                        st.success("✅ Token deleted")
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", key="cancel_delete_hf_token", use_container_width=True):
                        st.session_state[delete_confirm_hf_token] = False
                        st.rerun()
            
            # Use saved token
            hf_token = saved_token
            st.session_state['hf_token'] = hf_token
        else:
            # No token saved - show input box
            hf_token = st.text_input(
                "Token",
                type="password",
                value="",  # Don't show the actual token in the input
                help="Enter your Hugging Face token for gated models (Gemma, Llama, etc.). Get it from https://huggingface.co/settings/tokens. It will be saved permanently.",
                key="hf_token_input",
                placeholder="Enter Hugging Face token"
            )
            
            if hf_token:
                # Save the token permanently
                save_hf_token(hf_token)
                st.session_state['hf_token'] = hf_token
                st.success("✅ Saved!")
                st.rerun()
            
            # Try environment variable as fallback
            if not hf_token:
                import os
                hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
                if hf_token:
                    st.caption("ℹ️ Using environment variable")
                    st.session_state['hf_token'] = hf_token
        
        st.caption("Get your token from [Hugging Face settings](https://huggingface.co/settings/tokens)")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Training", "📄 Context Upload", "💬 Interact"])
    
    # Tab 1: Training
    with tab1:
        # Training Data Summary
        st.markdown("### 📊 Training Data Summary")
        
        from utils.training_data import TrainingDataManager
        from utils.config import get_model_queue_dir
        from pathlib import Path  # Import here to ensure it's available in this scope
        
        data_manager = TrainingDataManager(model_name)
        
        # Get queued files - check both metadata files and actual files
        queue_dir = get_model_queue_dir(model_name)
        queued_files = []
        total_queue_size = 0
        
        if queue_dir.exists():
            # First, try to get from metadata files (exclude YAML files)
            import json as json_module  # Use alias to avoid any scoping issues
            for metadata_file in queue_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json_module.load(f)
                        file_model = metadata.get("model")
                        if file_model == model_name:
                            # Skip YAML files - they're only used when attached to training files
                            if metadata.get("is_yaml", False):
                                continue
                            # Verify the actual file exists
                            filename = metadata.get("filename")
                            if filename:
                                # Also check file extension to be safe
                                file_ext = Path(filename).suffix.lower()
                                if file_ext in ['.yaml', '.yml']:
                                    continue
                                file_path = queue_dir / filename
                                if file_path.exists():
                                    # Ensure metadata dict is preserved as-is (including attached_yaml)
                                    # Create a fresh dict to avoid any reference issues
                                    file_metadata = dict(metadata)  # Explicit dict copy
                                    queued_files.append(file_metadata)
                                    total_queue_size += metadata.get("size", 0)
                except Exception as e:
                    # Log errors for debugging
                    import sys
                    error_type = type(e).__name__
                    print(f"[DEBUG] Error reading {metadata_file.name}: {error_type}: {e}", file=sys.stderr)
            
            # Also check for files without metadata (fallback, exclude YAML files)
            # BUT: Check if metadata file exists first - if it does, we should have already loaded it above
            for file_path in queue_dir.iterdir():
                if file_path.is_file() and not file_path.name.endswith("_metadata.json"):
                    # Skip YAML files
                    file_ext = file_path.suffix.lower()
                    if file_ext in ['.yaml', '.yml']:
                        continue
                    # Check if we already have this file in our list (from metadata)
                    filename = file_path.name
                    # Check if metadata file exists for this file
                    metadata_file_path = queue_dir / f"{file_path.stem}_metadata.json"
                    if metadata_file_path.exists():
                        # Metadata file exists - we should have loaded it above, skip this fallback
                        continue
                    # Only create metadata for files that truly don't have metadata files
                    if not any(f.get("filename") == filename for f in queued_files):
                        # Create metadata from file (files without metadata don't have attached_yaml)
                        file_size = file_path.stat().st_size
                        file_type = file_ext[1:] if file_ext.startswith('.') else file_ext
                        queued_files.append({
                            "filename": filename,
                            "file_type": file_type,
                            "date": "Unknown",
                            "model": model_name,
                            "size": file_size,
                            "queued": True,
                            "attached_yaml": None  # Explicitly set to None for files without metadata
                        })
                        total_queue_size += file_size
        
        # Debug: Show what we found
        if len(queued_files) == 0 and queue_dir.exists():
            # Additional debugging - try reading files directly
            debug_files = []
            import json as json_module  # Use alias to avoid scoping issues
            for metadata_file in queue_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        meta = json_module.load(f)
                        if meta.get("model") == model_name and not meta.get("is_yaml", False):
                            filename = meta.get("filename")
                            if filename and Path(filename).suffix.lower() not in ['.yaml', '.yml']:
                                file_path = queue_dir / filename
                                if file_path.exists():
                                    debug_files.append({"file": metadata_file.name, "model": meta.get("model"), "filename": filename})
                except:
                    pass
            
            # If debug found files but queued_files is empty, there's a logic issue
            if len(debug_files) > 0:
                st.warning(f"⚠️ Debug: Found {len(debug_files)} files but they weren't added to queued_files list!")
                with st.expander("Debug Details", expanded=True):
                    st.write("Files that should have been detected:")
                    for df in debug_files:
                        st.write(f"  - {df['file']} -> {df['filename']} (model: {df['model']})")
        
        # Display summary
        st.markdown("#### 📄 Queued Context Files")
        if len(queued_files) > 0:
            st.metric("Files Queued", len(queued_files))
            st.caption(f"Total size: {total_queue_size:,} bytes ({total_queue_size / 1024 / 1024:.2f} MB)")
            
            # Group files by YAML config
            files_by_yaml = {}
            for file_meta in queued_files:
                # Get attached_yaml, handling None, empty string, or missing key
                attached_yaml = file_meta.get('attached_yaml')
                
                if attached_yaml and str(attached_yaml).strip():
                    yaml_key = str(attached_yaml).strip()
                else:
                    yaml_key = 'No YAML'
                if yaml_key not in files_by_yaml:
                    files_by_yaml[yaml_key] = []
                files_by_yaml[yaml_key].append(file_meta)
            
            # Show file list grouped by YAML
            with st.expander(f"View {len(queued_files)} file(s)", expanded=False):
                # Sort YAML groups: "No YAML" last, others alphabetically
                sorted_yaml_keys = sorted([k for k in files_by_yaml.keys() if k != 'No YAML']) + (['No YAML'] if 'No YAML' in files_by_yaml else [])
                
                for yaml_key in sorted_yaml_keys:
                    files_in_group = files_by_yaml[yaml_key]
                    if yaml_key == 'No YAML':
                        st.markdown(f"**📦 No YAML Config ({len(files_in_group)} file(s)):**")
                    else:
                        st.markdown(f"**📋 YAML: {yaml_key} ({len(files_in_group)} file(s)):**")
                    
                    for file_meta in files_in_group:
                        file_type = file_meta.get('file_type', 'unknown').upper()
                        st.write(f"  • **{file_meta['filename']}** ({file_type}, {file_meta.get('size', 0):,} bytes)")
        else:
            st.info("No files queued yet")
            st.caption("Upload files in Tab 2 to queue them for training")
        
        # Overall status
        has_training_data = len(queued_files) > 0
        if has_training_data:
            st.success(f"✅ Ready to train with {len(queued_files)} file(s)")
        else:
            st.warning("⚠️ No training data available. Add files in Tab 2 before training.")
        
        st.markdown("")  # Break between sections
        
        # Check if there are active jobs that haven't been dismissed
        has_active_jobs = False
        active_jobs = []
        active_job = None
        if vast_api_key:
            try:
                from utils.vast_training_manager import VastTrainingManager
            except Exception as import_error:
                st.error(f"❌ Could not import VastTrainingManager: {str(import_error)}")
                has_active_jobs = False
                active_jobs = []
                active_job = None
            else:
                try:
                    training_manager = VastTrainingManager(model_name, vast_api_key)
                except Exception as init_error:
                    import traceback
                    st.error(f"❌ Could not initialize training manager: {str(init_error)}")
                    with st.expander("🔍 Error Details", expanded=False):
                        st.code(traceback.format_exc())
                    has_active_jobs = False
                    active_jobs = []
                    active_job = None
                else:
                    try:
                        jobs = training_manager.list_jobs()
                        if jobs:
                            # Get all active jobs (not finalized, dismissed, or cancelled)
                            for job in jobs:
                                is_finalized = job.get("finalized", False)
                                is_dismissed = job.get("dismissed", False)
                                is_cancelled = job.get("status") == "cancelled"
                                if not is_finalized and not is_dismissed and not is_cancelled:
                                    active_jobs.append(job)
                            
                            if active_jobs:
                                has_active_jobs = True
                                # Sort by creation date (newest first)
                                active_jobs = sorted(active_jobs, key=lambda x: x.get("created_at", ""), reverse=True)
                                # Default to latest job
                                active_job = active_jobs[0]
                    except Exception as list_error:
                        import traceback
                        error_msg = str(list_error)
                        error_traceback = traceback.format_exc()
                        print(f"[DEBUG] Error loading jobs: {error_msg}")
                        print(f"[DEBUG] Full traceback:\n{error_traceback}")
                        # Show error to user with details
                        st.error(f"❌ Could not load training jobs: {error_msg}")
                        with st.expander("🔍 Error Details (Click to expand)", expanded=False):
                            st.code(error_traceback)
                        # Don't set has_active_jobs if we can't load jobs
                        has_active_jobs = False
                        active_jobs = []
                        active_job = None
        
        # Launch Training Section - only show if no active (non-dismissed) jobs
        # Show this section if there's training data AND no active jobs
        if len(queued_files) > 0 and not has_active_jobs:
            st.markdown("### 🚀 Launch Training")
            
            # Instance selection: Only allow existing instances
            st.info("ℹ️ **Note:** You must have an existing Vast.ai instance running. The program will validate the instance before starting training.")
            
            selected_existing_instance = None
            
            # List existing instances
            if vast_api_key:
                try:
                    from utils.vast_ai_client import VastAIClient
                    vast_client = VastAIClient(api_key=vast_api_key)
                    existing_instances = vast_client.list_instances()
                    
                    if existing_instances:
                        # Filter instances by status
                        instance_options = []
                        for inst in existing_instances:
                            # Skip if inst is not a dictionary
                            if not isinstance(inst, dict):
                                continue
                            
                            inst_id = inst.get("id") or inst.get("instance_id")
                            if not inst_id:
                                continue
                            
                            # Convert inst_id to string if it's not already
                            inst_id = str(inst_id)
                            
                            status = inst.get("actual_status") or inst.get("status", "unknown")
                            if not isinstance(status, str):
                                status = str(status) if status else "unknown"
                            
                            gpu_info = inst.get("gpu_name") or inst.get("gpu", "Unknown GPU")
                            if not isinstance(gpu_info, str):
                                gpu_info = str(gpu_info) if gpu_info else "Unknown GPU"
                            
                            price = inst.get("dph_total") or inst.get("price_per_hour", 0)
                            if not isinstance(price, (int, float)):
                                try:
                                    price = float(price) if price else 0
                                except (ValueError, TypeError):
                                    price = 0
                            
                            # Show all instances, but indicate status
                            status_emoji = {
                                "running": "🟢",
                                "stopped": "🔴",
                                "loading": "🟡",
                                "starting": "🟡"
                            }.get(status.lower(), "⚪")
                            
                            # Safely format inst_id (handle if it's too short)
                            inst_id_display = inst_id[:8] + "..." if len(inst_id) > 8 else inst_id
                            label = f"{status_emoji} {inst_id_display} - {gpu_info} - ${price:.2f}/hr - {status}"
                            instance_options.append((label, inst_id, status))
                        
                        if instance_options:
                            selected_label = st.selectbox(
                                "Select Existing Instance",
                                options=[opt[0] for opt in instance_options],
                                key="existing_instance_select"
                            )
                            
                            # Find the selected instance
                            selected_idx = [opt[0] for opt in instance_options].index(selected_label)
                            selected_existing_instance = instance_options[selected_idx][1]
                            selected_status = instance_options[selected_idx][2].lower()
                            
                            if selected_status == "stopped":
                                if st.button("▶️ Start Instance", key="start_existing_instance"):
                                    try:
                                        with st.spinner("Starting instance..."):
                                            vast_client.start_instance(selected_existing_instance)
                                        st.success(f"✅ Instance {selected_existing_instance[:8]}... started!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error starting instance: {str(e)}")
                                
                                st.warning("⚠️ Selected instance is stopped. Start it before launching training.")
                                selected_existing_instance = None  # Don't allow using stopped instance
                            elif selected_status in ["loading", "starting"]:
                                st.info("ℹ️ Instance is starting up. Wait for it to be running before launching training.")
                                selected_existing_instance = None  # Don't allow using starting instance
                            else:
                                st.success(f"✅ Instance {selected_existing_instance[:8]}... is ready!")
                        else:
                            st.warning("No instances available")
                    else:
                        st.error("❌ No existing instances found. Please create an instance in Vast.ai first.")
                except Exception as e:
                    st.error(f"❌ Error listing instances: {str(e)}")
            else:
                st.warning("⚠️ Vast.ai API key required to list existing instances")
            
            # Show current base model and allow override
            from utils.model_manager import ModelManager
            model_manager = ModelManager()
            model_metadata = model_manager.get_model_metadata(model_name)
            current_base_model = model_metadata.get("base_model", "llama2") if model_metadata else "llama2"
            
            st.info(f"📌 Current base model: **{current_base_model}**")
            
            # Allow manual override of Hugging Face model identifier
            hf_model_override = st.text_input(
                "Hugging Face Model (optional override)",
                value="",
                help="If your model isn't in the mapping, manually specify the Hugging Face model identifier (e.g., 'microsoft/Phi-3-mini-4k-instruct'). Leave empty to use automatic mapping.",
                key="hf_model_override"
            )
            
            # Training configuration
            col1, col2 = st.columns(2)
            
            with col1:
                    epochs = st.number_input(
                    "Epochs (auto-calculated if not specified)", 
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Number of training epochs. Will be auto-calculated based on dataset size to target 200-500 training steps. You can override this value if needed.",
                    key="epochs_input"
                    )
            
            with col2:
                    learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=2e-4,
                    format="%.6f",
                    help="Learning rate for training",
                    key="learning_rate_input"
                    )
            
            # SSH Port configuration (optional override)
            st.markdown("---")
            st.markdown("#### SSH Connection Settings")
            ssh_port_override = st.number_input(
                "SSH Port (optional override)",
                min_value=1,
                max_value=65535,
                value=22,
                help="Override the SSH port if the default (22) or auto-detected port is incorrect. Leave as 22 if unsure. This will be used for all SSH connections during training.",
                key="ssh_port_override_input"
            )
            if ssh_port_override != 22:
                st.info(f"ℹ️ SSH port override set to **{ssh_port_override}**. This will be used instead of the auto-detected port.")
            
            # Launch button - requires existing instance
            button_label = "🚀 Launch Training Job"
            button_disabled = (selected_existing_instance is None)
            
            if st.button(button_label, key="launch_training", type="primary", disabled=button_disabled):
                if not vast_api_key:
                    st.error("❌ Vast.ai API key is required. Please enter it above.")
                else:
                    try:
                        from utils.vast_training_manager import VastTrainingManager
                        training_manager = VastTrainingManager(model_name, vast_api_key)
                        
                        # Group queued files by YAML attachment
                        from utils.config import get_model_queue_dir
                        queue_dir = get_model_queue_dir(model_name)
                        
                        # Get all training files (exclude YAML files)
                        training_file_groups = {}  # key: yaml_filename or None, value: list of file metadata
                        
                        if queue_dir.exists():
                            import json as json_module_launch  # Use alias to avoid scoping issues
                            for metadata_file in queue_dir.glob("*_metadata.json"):
                                try:
                                    with open(metadata_file, 'r', encoding='utf-8') as f:
                                        metadata = json_module_launch.load(f)
                                        if metadata.get("model") == model_name and not metadata.get("is_yaml"):
                                            filename = metadata.get("filename")
                                            if filename:
                                                file_path = queue_dir / filename
                                                if file_path.exists():
                                                    attached_yaml = metadata.get("attached_yaml")
                                                    yaml_key = attached_yaml if attached_yaml else None
                                                    if yaml_key not in training_file_groups:
                                                        training_file_groups[yaml_key] = []
                                                    training_file_groups[yaml_key].append(metadata)
                                except Exception as e:
                                    pass
                        
                        if not training_file_groups:
                            st.warning("⚠️ No training files found in queue. Please upload files in Tab 2 first.")
                        else:
                            # Build job queue - serialize all groups through one instance
                            job_queue = []
                            total_files = 0
                            
                            for yaml_key, file_group in training_file_groups.items():
                                yaml_filename = yaml_key if yaml_key else None
                                yaml_path = None
                                
                                # If YAML is attached, get the YAML file path
                                if yaml_filename:
                                    yaml_file_path = queue_dir / yaml_filename
                                    if yaml_file_path.exists():
                                        yaml_path = str(yaml_file_path)
                                    else:
                                        st.warning(f"⚠️ YAML file '{yaml_filename}' not found. Skipping this group.")
                                        continue
                                # If no YAML attached, yaml_filename and yaml_path remain None
                                # This will cause prepare_training_package to create a stock/default Axolotl config
                                
                                # Add to job queue
                                job_queue.append({
                                    "yaml_filename": yaml_filename,
                                    "yaml_path": yaml_path,
                                    "file_group": file_group,
                                    "file_count": len(file_group)
                                })
                                total_files += len(file_group)
                            
                            if job_queue:
                                # Launch single instance with job queue
                                with st.spinner(f"Launching training instance with {len(job_queue)} job(s) queued ({total_files} total files)..."):
                                    try:
                                        # Get HF token from session state or config
                                        from utils.config import get_hf_token
                                        hf_token = st.session_state.get('hf_token') or get_hf_token()
                                        
                                        # Must use existing instance
                                        if not selected_existing_instance:
                                            st.error("❌ Please select an existing instance.")
                                        else:
                                            # Get SSH port override if set (only use if not default 22)
                                            ssh_port_override_value = None
                                            if ssh_port_override != 22:
                                                ssh_port_override_value = ssh_port_override
                                            
                                            job_info = training_manager.launch_training_job(
                                                gpu_name=None,
                                                min_gpu_ram=16,
                                                max_price=None,
                                                disk_space=100,
                                                epochs=epochs,
                                                learning_rate=learning_rate,
                                                hf_model_override=hf_model_override.strip() if hf_model_override and hf_model_override.strip() else None,
                                                num_gpus=None,
                                                job_queue=job_queue,  # Pass entire job queue
                                                hf_token=hf_token if hf_token else None,
                                                existing_instance_id=selected_existing_instance,
                                                ssh_port_override=ssh_port_override_value
                                            )
                                            
                                            success_msg = f"✅ Launched training job with {len(job_queue)} job(s) queued:\n"
                                            for idx, queue_item in enumerate(job_queue, 1):
                                                job_yaml = queue_item.get('yaml_filename')
                                                yaml_desc = f" (YAML: {job_yaml})" if job_yaml else " (stock/default config)"
                                                success_msg += f"  • Job {idx}/{len(job_queue)}: {queue_item['file_count']} file(s){yaml_desc}\n"
                                            st.success(success_msg)
                                            
                                            # Clear terminal output for new job
                                            instance_id = job_info.get("instance_id")
                                            if instance_id:
                                                terminal_output_key = f"terminal_output_{instance_id}"
                                                st.session_state[terminal_output_key] = []
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error launching training instance: {str(e)}")
                    except ValueError as e:
                        # Model mapping error - show helpful message
                        error_msg = str(e)
                        st.error(f"❌ Model Mapping Error:\n\n{error_msg}")
                        st.info("💡 Tip: Make sure your model profile uses a supported base model name. You can check your model's base model in the model metadata.")
                    except Exception as e:
                        st.error(f"❌ Error launching training: {str(e)}")
        
        st.markdown("")  # Break between sections
        
        # Active Training Jobs - Phase-Based UI
        if has_active_jobs and active_jobs:
            # With serialized jobs, there should only be one active job (with a queue)
            # But keep selector for backward compatibility if multiple instances exist
            if len(active_jobs) > 1:
                st.markdown("### 📋 Active Training Jobs")
                job_options = []
                for idx, job in enumerate(active_jobs):
                    instance_id = job.get("instance_id", "unknown")
                    # Convert to string if it's an integer
                    instance_id_str = str(instance_id) if instance_id != "unknown" else "unknown"
                    job_queue = job.get("job_queue")
                    current_job_index = job.get("current_job_index")
                    
                    # Show queue info if available
                    if job_queue and current_job_index is not None:
                        queue_display = f"Job {current_job_index + 1}/{len(job_queue)} in queue"
                    else:
                        yaml_info = job.get("package_info", {}).get("yaml_config")
                        queue_display = f"YAML: {yaml_info}" if yaml_info else "No YAML"
                    
                    created = job.get("created_at", "Unknown")
                    # Create display name
                    if instance_id_str != "unknown" and len(instance_id_str) > 8:
                        instance_display = f"{instance_id_str[:8]}..."
                    else:
                        instance_display = instance_id_str
                    display_name = f"Job {instance_display} ({queue_display}) - {created[:10] if len(created) > 10 else created}"
                    job_options.append((display_name, idx))
                
                selected_job_idx = st.selectbox(
                    "Select Job to View",
                    options=range(len(active_jobs)),
                    format_func=lambda x: job_options[x][0],
                    key="job_selector"
                )
                active_job = active_jobs[selected_job_idx]
            else:
                active_job = active_jobs[0]
        
        if has_active_jobs and active_job:
            st.markdown("### 📊 Active Training Job")
            
            # Add dismiss button to reset stuck jobs (with confirmation)
            dismiss_confirm_key = f"dismiss_confirm_{active_job.get('instance_id')}"
            if dismiss_confirm_key not in st.session_state:
                st.session_state[dismiss_confirm_key] = False
            
            col_dismiss, col_spacer = st.columns([1, 5])
            with col_dismiss:
                if not st.session_state[dismiss_confirm_key]:
                    if st.button("🗑️ Dismiss Job", key="dismiss_job", help="Dismiss this job (instance will remain running - you must stop or destroy it manually in Vast.ai)", type="secondary"):
                        st.session_state[dismiss_confirm_key] = True
                        st.rerun()
                else:
                    st.warning("⚠️ Are you sure you want to dismiss this job? The Vast.ai instance will remain running - you must stop or destroy it manually in Vast.ai to stop charges.")
                    col_confirm, col_cancel = st.columns([1, 1])
                    with col_confirm:
                        if st.button("✅ Confirm Dismiss", key="confirm_dismiss_job", type="primary"):
                            try:
                                from datetime import datetime
                                from utils.vast_training_manager import VastTrainingManager
                                training_manager = VastTrainingManager(model_name, vast_api_key)
                                
                                # Mark job as dismissed (do NOT stop or destroy instance)
                                instance_id = active_job.get("instance_id")
                                
                                # Mark job as dismissed
                                active_job["dismissed"] = True
                                active_job["dismissed_at"] = datetime.now().isoformat()
                                training_manager._save_job(active_job)
                                st.session_state[dismiss_confirm_key] = False
                                st.success("✅ Job dismissed. You can now launch a new training job.")
                                if instance_id:
                                    st.warning(f"⚠️ **Important:** The Vast.ai instance ({instance_id[:8]}...) is still running. You must stop or destroy it manually in Vast.ai to stop charges.")
                                st.rerun()
                            except Exception as e:
                                st.session_state[dismiss_confirm_key] = False
                                st.error(f"Error dismissing job: {str(e)}")
                    with col_cancel:
                        if st.button("❌ Cancel", key="cancel_dismiss_job"):
                            st.session_state[dismiss_confirm_key] = False
                            st.rerun()
            
            try:
                from utils.vast_training_manager import VastTrainingManager
                training_manager = VastTrainingManager(model_name, vast_api_key)
                
                # Initialize phase tracking in session state
                phase_key = f"training_phase_{active_job.get('instance_id')}"
                terminal_output_key = f"terminal_output_{active_job.get('instance_id')}"
                
                # Always check current job status to determine phase (don't rely on cached phase)
                # This ensures that if status changes, phase updates accordingly
                job_status = active_job.get('status', 'unknown')
                current_phase = st.session_state.get(phase_key)
                
                # Determine what phase should be based on current job status
                if job_status == 'launching':
                    target_phase = 1  # Validate instance
                elif job_status == 'validated' and not active_job.get('files_uploaded'):
                    target_phase = 2  # Upload file
                elif job_status == 'validated' and active_job.get('files_uploaded'):
                    target_phase = 3  # Do training
                elif job_status == 'running' and not active_job.get('files_uploaded'):
                    # If status is 'running' but not validated, go to Phase 1 to validate
                    target_phase = 1  # Validate instance
                elif job_status == 'running' and active_job.get('files_uploaded'):
                    target_phase = 3  # Do training
                elif job_status == 'completed':
                    target_phase = 4  # Finalize
                else:
                    target_phase = 1  # Default to phase 1 (validate instance)
                
                # Set phase if not set, or if it needs to be updated based on status
                phase_changed = False
                if phase_key not in st.session_state or current_phase != target_phase:
                    # Phase is changing - clear terminal output
                    if phase_key in st.session_state and current_phase != target_phase:
                        phase_changed = True
                    st.session_state[phase_key] = target_phase
                
                if terminal_output_key not in st.session_state:
                    st.session_state[terminal_output_key] = []
                
                # Clear terminal output if phase changed
                if phase_changed:
                    st.session_state[terminal_output_key] = []
                
                current_phase = st.session_state[phase_key]
                terminal_output = st.session_state[terminal_output_key]
                
                # Phase definitions
                phases = {
                    1: {"name": "Validate Instance", "icon": "✅", "description": "Validating that the existing instance is running and meets all requirements"},
                    2: {"name": "Upload Files", "icon": "📤", "description": "Uploading training files to the instance via SSH/SCP"},
                    3: {"name": "Do Training", "icon": "⚙️", "description": "Monitoring training progress"},
                    4: {"name": "Finalize", "icon": "✅", "description": "Downloading weights and cleaning up (instance remains running - shut down manually)"}
                }
                
                # Display phase progress
                st.markdown("#### Training Progress")
                phase_cols = st.columns(4)
                for i, (phase_num, phase_info) in enumerate(phases.items()):
                    with phase_cols[i]:
                        if phase_num < current_phase:
                            st.success(f"{phase_info['icon']} {phase_info['name']}")
                        elif phase_num == current_phase:
                            st.info(f"**{phase_info['icon']} {phase_info['name']}**")
                        else:
                            st.caption(f"{phase_info['icon']} {phase_info['name']}")
                
                st.markdown("---")
                
                # Phase 1: Validate Instance
                if current_phase == 1:
                    # Ensure datetime is available in this scope
                    from datetime import datetime
                    
                    st.markdown(f"### {phases[1]['icon']} Phase 1: {phases[1]['name']}")
                    st.caption("Validate that the existing instance is running and meets all requirements for training.")
                    
                    # Automatically fetch and display SSH info when entering Phase 1
                    instance_id = active_job.get("instance_id")
                    ssh_host = active_job.get("ssh_host")
                    
                    # Check for SSH port override first (user-specified port takes precedence)
                    ssh_port_override = active_job.get("ssh_port_override")
                    if ssh_port_override:
                        ssh_port = ssh_port_override
                    else:
                        ssh_port = active_job.get("ssh_port", 22)
                    
                    # If SSH info not in job, try to fetch from API
                    if instance_id and not ssh_host:
                        try:
                            ssh_info = training_manager.get_instance_ssh_info(instance_id)
                            ssh_host = ssh_info.get("host")
                            api_ssh_port = ssh_info.get("port", 22)
                            
                            # Use override port if set, otherwise use API port
                            if ssh_port_override:
                                ssh_port = ssh_port_override
                            else:
                                ssh_port = api_ssh_port
                            
                            if ssh_host:
                                # Save to job for future use
                                active_job["ssh_host"] = ssh_host
                                active_job["ssh_port"] = ssh_port
                                training_manager._save_job(active_job)
                            else:
                                st.warning("⚠️ SSH info not yet available from Vast.ai. The instance may still be initializing.")
                        except Exception as e:
                            st.warning(f"⚠️ Could not fetch SSH info from API: {str(e)}")
                    
                    # Display SSH info if available
                    if ssh_host:
                        port_note = " (override)" if ssh_port_override else ""
                        st.success(f"🔐 **SSH Connection Info:** `ssh -p {ssh_port} root@{ssh_host}`{port_note}")
                    
                    st.info("💡 **Click 'Check Instance'** to validate the instance. Phase 2 will only start when all checks pass.")
                    
                    # Terminal output area (scrollable)
                    st.markdown("#### Validation Output")
                    terminal_container = st.container()
                    with terminal_container:
                        # Show existing output in a scrollable code block
                        if terminal_output:
                            # Keep only last 200 lines to prevent UI slowdown
                            display_output = terminal_output[-200:] if len(terminal_output) > 200 else terminal_output
                            output_text = "\n".join(display_output)
                            st.code(output_text, language="text")
                            if len(terminal_output) > 200:
                                st.caption(f"Showing last 200 of {len(terminal_output)} lines")
                        else:
                            st.info("No output yet. Click 'Check Instance' to validate the instance.")
                    
                    # Action buttons
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("✅ Check Instance", key="check_instance_status", type="primary"):
                            try:
                                instance_id = active_job.get("instance_id")
                                
                                if not instance_id:
                                    terminal_output.append(f"[ERROR] No instance ID found in job.")
                                    st.error("No instance ID found in job.")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.rerun()
                                    return
                                
                                # Run validation with progress updates
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting instance validation...")
                                terminal_output.append(f"[VALIDATION] Instance ID: {instance_id}")
                                st.session_state[terminal_output_key] = terminal_output  # Save immediately so user sees progress
                                
                                # Get SSH info: try job first, then API
                                ssh_host = active_job.get("ssh_host")
                                
                                # Check for SSH port override first (user-specified port takes precedence)
                                ssh_port_override = active_job.get("ssh_port_override")
                                if ssh_port_override:
                                    ssh_port = ssh_port_override
                                    terminal_output.append(f"[VALIDATION] Using SSH port override: {ssh_port}")
                                else:
                                    ssh_port = active_job.get("ssh_port", 22)
                                
                                if ssh_host:
                                    port_source = "override" if ssh_port_override else "saved"
                                    terminal_output.append(f"[VALIDATION] Using saved SSH info: {ssh_host}:{ssh_port} ({port_source})")
                                    st.session_state[terminal_output_key] = terminal_output
                                else:
                                    # Try to get from API
                                    terminal_output.append(f"[VALIDATION] Fetching SSH info from Vast.ai API...")
                                    st.session_state[terminal_output_key] = terminal_output
                                    try:
                                        # Get SSH info from API
                                        ssh_info = training_manager.get_instance_ssh_info(instance_id)
                                        ssh_host = ssh_info.get("host")
                                        api_ssh_port = ssh_info.get("port", 22)
                                        
                                        # Use override port if set, otherwise use API port
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                            terminal_output.append(f"[VALIDATION] Using SSH port override: {ssh_port} (instead of API port: {api_ssh_port})")
                                        else:
                                            ssh_port = api_ssh_port
                                        
                                        # Log raw data for debugging
                                        raw_data = ssh_info.get("raw_data", {})
                                        if raw_data:
                                            terminal_output.append(f"[DEBUG] Available SSH fields from API:")
                                            connection_type = raw_data.get("connection_type", "unknown")
                                            terminal_output.append(f"[DEBUG]   Connection type: {connection_type}")
                                            if raw_data.get("public_ipaddr"):
                                                terminal_output.append(f"[DEBUG]   public_ipaddr: {raw_data.get('public_ipaddr')}")
                                            if raw_data.get("ipaddr"):
                                                terminal_output.append(f"[DEBUG]   ipaddr: {raw_data.get('ipaddr')}")
                                            if raw_data.get("ssh_host"):
                                                terminal_output.append(f"[DEBUG]   ssh_host (gateway): {raw_data.get('ssh_host')}")
                                            if raw_data.get("ssh_port") is not None:
                                                terminal_output.append(f"[DEBUG]   ssh_port (gateway): {raw_data.get('ssh_port')} (type: {type(raw_data.get('ssh_port')).__name__})")
                                            if raw_data.get("port") is not None:
                                                terminal_output.append(f"[DEBUG]   port: {raw_data.get('port')} (type: {type(raw_data.get('port')).__name__})")
                                            terminal_output.append(f"[DEBUG]   Selected: {ssh_host}:{ssh_port} ({connection_type} connection)")
                                            st.session_state[terminal_output_key] = terminal_output
                                        
                                        if ssh_host:
                                            active_job["ssh_host"] = ssh_host
                                            active_job["ssh_port"] = ssh_port
                                            training_manager._save_job(active_job)
                                            terminal_output.append(f"[VALIDATION] ✓ Retrieved SSH info from API: {ssh_host}:{ssh_port}")
                                            st.session_state[terminal_output_key] = terminal_output
                                        else:
                                            terminal_output.append(f"[WARNING] SSH info not yet available from API. Instance may still be initializing.")
                                            st.session_state[terminal_output_key] = terminal_output
                                    except Exception as e:
                                        error_msg = str(e)
                                        terminal_output.append(f"[ERROR] Could not fetch SSH info from API: {error_msg}")
                                        terminal_output.append(f"[ERROR] Please ensure the instance is running and try again.")
                                        st.session_state[terminal_output_key] = terminal_output
                                
                                # Show spinner while validation runs
                                validation_result = None
                                with st.spinner("Validating instance... This may take 30-60 seconds."):
                                    try:
                                        terminal_output.append(f"[VALIDATION] Checking instance status...")
                                        st.session_state[terminal_output_key] = terminal_output
                                        
                                        # Pass SSH info to validation (use None if not available, let validation try to get it)
                                        validation_result = training_manager.validate_instance(instance_id, ssh_host_override=ssh_host, ssh_port_override=ssh_port if ssh_host else None)
                                        
                                        terminal_output.append(f"[VALIDATION] Validation complete.")
                                        st.session_state[terminal_output_key] = terminal_output
                                    except Exception as e:
                                        error_msg = str(e)
                                        terminal_output.append(f"[ERROR] Validation failed with exception: {error_msg}")
                                        terminal_output.append(f"[ERROR] This may indicate a network issue or API timeout.")
                                        st.session_state[terminal_output_key] = terminal_output
                                        st.error(f"Validation error: {error_msg}")
                                        st.rerun()
                                        return
                                
                                # If validation failed with exception, we already returned above
                                if validation_result is None:
                                    terminal_output.append(f"[ERROR] Validation returned no result.")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.error("Validation failed - no result returned.")
                                    st.rerun()
                                    return
                                
                                # Display validation results
                                terminal_output.append(f"[VALIDATION] ========================================")
                                
                                if validation_result.get("valid"):
                                    terminal_output.append(f"[VALIDATION] ✅ Instance validation PASSED")
                                else:
                                    terminal_output.append(f"[VALIDATION] ❌ Instance validation FAILED")
                                
                                # Show errors
                                errors = validation_result.get("errors", [])
                                if errors:
                                    terminal_output.append(f"[VALIDATION] Errors found ({len(errors)}):")
                                    for error in errors:
                                        terminal_output.append(f"[ERROR]   • {error}")
                                
                                # Show warnings
                                warnings = validation_result.get("warnings", [])
                                if warnings:
                                    terminal_output.append(f"[VALIDATION] Warnings ({len(warnings)}):")
                                    for warning in warnings:
                                        terminal_output.append(f"[WARNING]   • {warning}")
                                
                                # Show details
                                details = validation_result.get("details", {})
                                if details:
                                    terminal_output.append(f"[VALIDATION] Details:")
                                    if "status" in details:
                                        terminal_output.append(f"[INFO]   Status: {details['status']}")
                                    if "ssh_host" in details:
                                        terminal_output.append(f"[INFO]   SSH Host: {details['ssh_host']}:{details.get('ssh_port', 22)}")
                                    if "jupyter_url" in details:
                                        terminal_output.append(f"[INFO]   Jupyter URL: {details['jupyter_url']}")
                                        terminal_output.append(f"[INFO]   (Note: SSH is still required for file uploads)")
                                    if "disk_available_gb" in details:
                                        terminal_output.append(f"[INFO]   Disk Available: {details['disk_available_gb']:.1f} GB")
                                    if "gpu_info" in details:
                                        terminal_output.append(f"[INFO]   GPU Info:")
                                        for gpu_line in details["gpu_info"]:
                                            terminal_output.append(f"[INFO]     {gpu_line}")
                                    if "python_version" in details:
                                        terminal_output.append(f"[INFO]   Python: {details['python_version']}")
                                    if details.get("directories_ok"):
                                        terminal_output.append(f"[INFO]   Directories: OK")
                                
                                terminal_output.append(f"[VALIDATION] ========================================")
                                
                                # If validation passed, advance to Phase 2
                                if validation_result.get("valid"):
                                    terminal_output.append(f"[SUCCESS] All validation checks passed! Instance is ready for training.")
                                    terminal_output.append(f"[SUCCESS] Proceeding to Phase 2...")
                                    
                                    # Save SSH info from validation to job (so it persists across phases and redo operations)
                                    validation_details = validation_result.get("details", {})
                                    if "ssh_host" in validation_details:
                                        active_job["ssh_host"] = validation_details["ssh_host"]
                                        active_job["ssh_port"] = validation_details.get("ssh_port", 22)
                                    
                                    # Update job status
                                    active_job["status"] = "validated"
                                    training_manager._save_job(active_job)
                                    
                                    # Advance to Phase 2
                                    st.session_state[phase_key] = 2
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.success("✅ Instance validation passed! Proceeding to Phase 2.")
                                    st.rerun()
                                else:
                                    # Check if we have errors or just warnings
                                    errors = validation_result.get("errors", [])
                                    warnings = validation_result.get("warnings", [])
                                    if errors:
                                        terminal_output.append(f"[ERROR] Validation failed. Please fix the issues above before proceeding.")
                                        st.error(f"❌ Validation failed with {len(errors)} error(s). See details above.")
                                    else:
                                        # Only warnings - allow proceeding
                                        terminal_output.append(f"[WARNING] Validation completed with warnings. You can proceed if you've verified SSH works manually.")
                                        st.warning(f"⚠️ Validation completed with {len(warnings)} warning(s). See details above.")
                                        st.info("💡 If you've verified SSH works manually, you can proceed to Phase 2.")
                                        
                                        # Add a button to proceed anyway if only warnings
                                        if st.button("✅ Proceed to Phase 2 Anyway", key="proceed_with_warnings", type="primary"):
                                            terminal_output.append(f"[INFO] Proceeding to Phase 2 despite warnings (user confirmed SSH works manually).")
                                            
                                            # Save SSH info from validation to job (so it persists across phases and redo operations)
                                            validation_details = validation_result.get("details", {})
                                            if "ssh_host" in validation_details:
                                                active_job["ssh_host"] = validation_details["ssh_host"]
                                                active_job["ssh_port"] = validation_details.get("ssh_port", 22)
                                            
                                            # Update job status
                                            active_job["status"] = "validated"
                                            training_manager._save_job(active_job)
                                            
                                            # Advance to Phase 2
                                            st.session_state[phase_key] = 2
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.success("✅ Proceeding to Phase 2.")
                                            st.rerun()
                                    
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.rerun()
                                
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] Validation error: {error_msg}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error during validation: {error_msg}")
                    
                    with col2:
                        if st.button("🔄 Redo Phase", key="retry_phase_1"):
                            try:
                                # Clear terminal before redoing phase
                                st.session_state[terminal_output_key] = []
                                terminal_output = []
                                
                                instance_id = active_job.get("instance_id")
                                if instance_id:
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Resetting Phase 1 validation...")
                                    terminal_output.append(f"[INFO] You can now click 'Check Instance' again to re-validate the instance.")
                                    
                                    # Reset job status
                                    active_job["status"] = "launching"
                                    training_manager._save_job(active_job)
                                    
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.info("ℹ️ Phase 1 reset. Click 'Check Instance' to validate the instance again.")
                                    st.rerun()
                                else:
                                    terminal_output.append(f"[ERROR] No instance ID found in job")
                                    st.error("No instance ID found")
                                st.session_state[terminal_output_key] = terminal_output
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] Failed to redo phase: {error_msg}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                
                # Phase 2: Upload File
                elif current_phase == 2:
                    # Ensure datetime is available in this scope
                    from datetime import datetime
                    
                    st.markdown(f"### {phases[2]['icon']} Phase 2: {phases[2]['name']}")
                    st.caption(phases[2]['description'])
                    
                    # Initialize Phase 2: Check SSH and create directories (show in terminal, don't auto-upload)
                    phase2_init_key = f"phase2_init_{active_job.get('instance_id')}"
                    
                    # For jobs with a queue, verify files are actually uploaded
                    # Reset files_uploaded flag if all_package_infos doesn't match job queue
                    job_queue = active_job.get("job_queue")
                    if job_queue and len(job_queue) > 0:
                        all_package_infos = active_job.get("all_package_infos", [])
                        # If we have a job queue but all_package_infos doesn't match, files haven't been uploaded
                        # Also check if all_package_infos is empty or None
                        if not all_package_infos or len(all_package_infos) != len(job_queue):
                            if active_job.get("files_uploaded", False):
                                active_job["files_uploaded"] = False
                                training_manager._save_job(active_job)
                    
                    if not active_job.get("files_uploaded", False) and phase2_init_key not in st.session_state:
                        try:
                            instance_id = active_job.get("instance_id")
                            if instance_id:
                                # Get SSH info - prefer saved SSH details from job over API
                                ssh_host = active_job.get("ssh_host")
                                ssh_port = active_job.get("ssh_port", 22)
                                
                                # If not in job, get from API
                                if not ssh_host:
                                    job_status = training_manager.get_job_status(instance_id)
                                    ssh_host = job_status.get("ssh_host")
                                    ssh_port = job_status.get("ssh_port", 22)
                                    # Save to job for future use
                                    if ssh_host:
                                        active_job["ssh_host"] = ssh_host
                                        active_job["ssh_port"] = ssh_port
                                        training_manager._save_job(active_job)
                                
                                if ssh_host:
                                    # Append to existing terminal output (don't clear it)
                                    if not terminal_output:
                                        terminal_output = []
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Phase 2...")
                                    terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                    
                                    # Test SSH connection
                                    import subprocess
                                    test_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "echo 'SSH connection test'"
                                    ]
                                    test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
                                    if test_result.returncode == 0:
                                        terminal_output.append(f"[SSH] Connection test successful")
                                    else:
                                        stderr_filtered = filter_malloc_warnings(test_result.stderr)
                                        terminal_output.append(f"[SSH] Connection test failed: {stderr_filtered[:200]}")
                                    
                                    # Create directories with retry logic
                                    terminal_output.append(f"[SSH] Creating directories on remote instance...")
                                    import time
                                    mkdir_success = False
                                    for retry in range(3):
                                        if retry > 0:
                                            wait_time = 2 ** retry  # Exponential backoff: 2, 4 seconds
                                            terminal_output.append(f"[SSH] Retry {retry}/3 after {wait_time}s wait...")
                                            time.sleep(wait_time)
                                        
                                        mkdir_cmd = [
                                            "ssh", "-p", str(ssh_port), 
                                            "-o", "StrictHostKeyChecking=no", 
                                            "-o", "ConnectTimeout=30",
                                            "-o", "UserKnownHostsFile=/dev/null",
                                            f"root@{ssh_host}",
                                            "mkdir -p /workspace/data && mkdir -p /workspace/output/training && echo 'Directories ready'"
                                        ]
                                        mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=30)
                                        if mkdir_result.returncode == 0:
                                            terminal_output.append(f"[SSH] Directories created successfully")
                                            
                                            # Check for onstart errors in Phase 2
                                            try:
                                                check_onstart_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    "tail -30 /var/log/onstart.log 2>/dev/null || tail -30 /tmp/onstart.log 2>/dev/null || echo 'no_onstart_log'"
                                                ]
                                                onstart_check = subprocess.run(check_onstart_cmd, capture_output=True, text=True, timeout=15)
                                                if "no_onstart_log" not in onstart_check.stdout and onstart_check.stdout.strip():
                                                    onstart_content = onstart_check.stdout.lower()
                                                    # Check for critical errors
                                                    if any(err in onstart_content for err in ["syntax error", "error:", "failed", "exception", "traceback", "fatal", "no such file", "command not found"]):
                                                        terminal_output.append(f"[WARNING] Errors detected in onstart script:")
                                                        for line in onstart_check.stdout.strip().split("\n")[-5:]:
                                                            if line.strip() and any(err in line.lower() for err in ["error", "failed", "exception", "syntax"]):
                                                                terminal_output.append(f"[ONSTART ERROR] {line[:200]}")
                                                        st.warning("⚠️ Errors detected in onstart script. Installation may have failed.")
                                            except:
                                                pass  # Don't block on error checking
                                            
                                            terminal_output.append(f"[SUCCESS] File structure ready. Click 'Upload Files' button to proceed.")
                                            mkdir_success = True
                                            break
                                        else:
                                            stderr_filtered = filter_malloc_warnings(mkdir_result.stderr)
                                            if retry < 2:
                                                terminal_output.append(f"[SSH] Directory creation failed (attempt {retry + 1}/3): {stderr_filtered[:200]}")
                                            else:
                                                terminal_output.append(f"[SSH] Directory creation failed after 3 attempts: {stderr_filtered[:200]}")
                                    if not mkdir_success:
                                        terminal_output.append(f"[WARNING] Directory creation failed. SSH may need more time to initialize. You can retry the upload.")
                                    
                                    # Save terminal output and mark as initialized (but don't auto-upload)
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.session_state[phase2_init_key] = True
                                    st.rerun()
                                else:
                                    # SSH not available yet, mark as initialized so we don't keep trying
                                    terminal_output.append(f"[WARNING] SSH host not available yet. Waiting for instance to be ready...")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.session_state[phase2_init_key] = True
                        except Exception as e:
                            # If initialization fails, mark as initialized so we don't keep trying
                            terminal_output.append(f"[ERROR] Initialization error: {str(e)}")
                            st.session_state[terminal_output_key] = terminal_output
                            st.session_state[phase2_init_key] = True
                            # Continue - user can click Upload Files
                            pass
                    
                    # Terminal output area (scrollable)
                    st.markdown("#### Terminal Output")
                    terminal_container = st.container()
                    with terminal_container:
                        if terminal_output:
                            # Keep only last 200 lines
                            display_output = terminal_output[-200:] if len(terminal_output) > 200 else terminal_output
                            output_text = "\n".join(display_output)
                            st.code(output_text, language="text")
                            if len(terminal_output) > 200:
                                st.caption(f"Showing last 200 of {len(terminal_output)} lines")
                        else:
                            if active_job.get("files_uploaded", False):
                                st.info("Files already uploaded. Check terminal output above or click 'Next Phase' to continue.")
                            else:
                                phase2_init_key = f"phase2_init_{active_job.get('instance_id')}"
                                if phase2_init_key in st.session_state:
                                    st.info("File structure initialized. Review terminal output above, then click 'Upload Files' to proceed.")
                                else:
                                    st.info("Initializing file structure... Check terminal output above.")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        # Only show upload button if:
                        # 1. Files haven't been uploaded yet
                        # 2. Phase 2 initialization was successful (directories created)
                        files_already_uploaded = active_job.get("files_uploaded", False)
                        phase2_init_key = f"phase2_init_{active_job.get('instance_id')}"
                        phase2_initialized = phase2_init_key in st.session_state
                        
                        if not files_already_uploaded and phase2_initialized:
                            if st.button("📤 Upload Files", key="upload_files"):
                                try:
                                    instance_id = active_job.get("instance_id")
                                    package_info = active_job.get("package_info")
                                    
                                    if not package_info:
                                        st.error("Package info not found. Please restart training.")
                                    else:
                                        terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] ===== Starting File Upload =====")
                                        
                                        # Get job queue information
                                        job_queue = active_job.get("job_queue")
                                        
                                        if not job_queue or len(job_queue) == 0:
                                            st.error("Job queue not found. Please restart training.")
                                            return
                                        
                                        terminal_output.append(f"[INFO] Uploading {len(job_queue)} job(s) to instance...")
                                        
                                        # Prepare package_info for all jobs if not already prepared
                                        from utils.vast_training_manager import VastTrainingManager
                                        all_package_infos = active_job.get("all_package_infos", [])
                                        
                                        if len(all_package_infos) != len(job_queue):
                                            terminal_output.append(f"[INFO] Preparing training packages for all {len(job_queue)} job(s)...")
                                            all_package_infos = []
                                            
                                            for job_idx, job_item in enumerate(job_queue):
                                                yaml_path = job_item.get("yaml_path")
                                                file_group = job_item.get("file_group", [])
                                                
                                                # Prepare package for this job
                                                job_package = training_manager.prepare_training_package(
                                                    epochs=active_job.get("epochs", 10),
                                                    learning_rate=active_job.get("learning_rate", 2e-4),
                                                    hf_model_override=active_job.get("hf_model_override"),
                                                    yaml_config_path=yaml_path,
                                                    file_group=file_group
                                                )
                                                all_package_infos.append(job_package)
                                            
                                            # Save all package infos to job
                                            active_job["all_package_infos"] = all_package_infos
                                            training_manager._save_job(active_job)
                                            terminal_output.append(f"[INFO] Prepared {len(all_package_infos)} training package(s)")
                                        
                                        # Log what will be uploaded for all jobs
                                        terminal_output.append(f"[PRE-UPLOAD] Files to be uploaded for {len(job_queue)} job(s):")
                                        
                                        for job_idx, job_item in enumerate(job_queue):
                                            job_yaml = job_item.get("yaml_filename")
                                            file_group = job_item.get("file_group", [])
                                            
                                            terminal_output.append(f"[PRE-UPLOAD] --- Job {job_idx + 1}/{len(job_queue)} ---")
                                            if job_yaml:
                                                terminal_output.append(f"[PRE-UPLOAD] YAML Config: {job_yaml}")
                                            else:
                                                terminal_output.append(f"[PRE-UPLOAD] YAML Config: None (using stock/default Axolotl config)")
                                            
                                            terminal_output.append(f"[PRE-UPLOAD] Queue files ({len(file_group)} file(s)):")
                                            for file_meta in file_group:
                                                filename = file_meta.get("filename", "unknown")
                                                file_type = file_meta.get("file_type", "unknown").upper()
                                                file_size = file_meta.get("size", 0)
                                                attached_yaml = file_meta.get("attached_yaml")
                                                # Files without attached_yaml should always show as using stock/default config
                                                # Files with attached_yaml show their specific YAML
                                                if attached_yaml:
                                                    yaml_info = f" (YAML: {attached_yaml})"
                                                else:
                                                    yaml_info = " (using stock/default config)"
                                                terminal_output.append(f"[PRE-UPLOAD]   • {filename} ({file_type}, {file_size:,} bytes){yaml_info}")
                                            
                                            # Show what will be uploaded for this job
                                            job_package = all_package_infos[job_idx]
                                            job_config_path = job_package.get('config_path')
                                            job_dataset_path = job_package.get('dataset_path')
                                            
                                            terminal_output.append(f"[PRE-UPLOAD] Processed files:")
                                            if job_config_path:
                                                job_config_file = Path(job_config_path)
                                                if job_config_file.exists():
                                                    config_size = job_config_file.stat().st_size
                                                    terminal_output.append(f"[PRE-UPLOAD]   • Config: {job_config_file.name} ({config_size:,} bytes)")
                                                    terminal_output.append(f"[PRE-UPLOAD]     → /workspace/data/axolotl_config_{job_idx}.yaml")
                                            
                                            if job_dataset_path:
                                                job_dataset_file = Path(job_dataset_path)
                                                if job_dataset_file.exists():
                                                    dataset_size = job_dataset_file.stat().st_size
                                                    terminal_output.append(f"[PRE-UPLOAD]   • Training Data: {job_dataset_file.name} ({dataset_size:,} bytes)")
                                                    terminal_output.append(f"[PRE-UPLOAD]     → /workspace/data/training_data_{job_idx}.jsonl")
                                        
                                        terminal_output.append(f"[PRE-UPLOAD] ========================================")
                                        
                                        # Get SSH info - prefer saved SSH details from job over API
                                        ssh_host = active_job.get("ssh_host")
                                        ssh_port = active_job.get("ssh_port", 22)
                                        
                                        # If not in job, get from API
                                        if not ssh_host:
                                            job_status = training_manager.get_job_status(instance_id)
                                            ssh_host = job_status.get("ssh_host")
                                            ssh_port = job_status.get("ssh_port", 22)
                                            # Save to job for future use
                                            if ssh_host:
                                                active_job["ssh_host"] = ssh_host
                                                active_job["ssh_port"] = ssh_port
                                                training_manager._save_job(active_job)
                                        
                                        if not ssh_host:
                                            terminal_output.append(f"[ERROR] SSH host not available. Instance may not be ready.")
                                            st.error("SSH host not available. Please wait for instance to be ready.")
                                        else:
                                            # Check if directories already exist (from auto-init)
                                            # If not, create them
                                            check_dirs_cmd = [
                                                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                f"root@{ssh_host}",
                                                "test -d /workspace/data && test -d /workspace/output/training && echo 'exists' || echo 'missing'"
                                            ]
                                            import subprocess
                                            check_result = subprocess.run(check_dirs_cmd, capture_output=True, text=True, timeout=15)
                                            
                                            if "exists" not in check_result.stdout:
                                                # Directories don't exist, create them with retry logic
                                                terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                                terminal_output.append(f"[SSH] Creating directories on remote instance...")
                                                import time
                                                mkdir_success = False
                                                for retry in range(3):
                                                    if retry > 0:
                                                        wait_time = 2 ** retry  # Exponential backoff: 2, 4 seconds
                                                        terminal_output.append(f"[SSH] Retry {retry}/3 after {wait_time}s wait...")
                                                        time.sleep(wait_time)
                                                    
                                                    mkdir_cmd = [
                                                        "ssh", "-p", str(ssh_port), 
                                                        "-o", "StrictHostKeyChecking=no", 
                                                        "-o", "ConnectTimeout=30",
                                                        "-o", "UserKnownHostsFile=/dev/null",
                                                        f"root@{ssh_host}",
                                                        "mkdir -p /workspace/data && mkdir -p /workspace/output/training && echo 'Directories ready'"
                                                    ]
                                                    mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=30)
                                                    if mkdir_result.returncode == 0:
                                                        terminal_output.append(f"[SSH] Directories created successfully")
                                                        mkdir_success = True
                                                        break
                                                    else:
                                                        stderr_filtered = filter_malloc_warnings(mkdir_result.stderr)
                                                        if retry < 2:
                                                            terminal_output.append(f"[SSH] Directory creation failed (attempt {retry + 1}/3): {stderr_filtered[:200]}")
                                                        else:
                                                            terminal_output.append(f"[SSH] Directory creation failed after 3 attempts: {stderr_filtered[:200]}")
                                                if not mkdir_success:
                                                    terminal_output.append(f"[WARNING] Directory creation failed. Uploads may fail.")
                                            else:
                                                # Directories already exist
                                                terminal_output.append(f"[SSH] Directories already exist, proceeding with upload...")
                                            
                                            # Upload all jobs with numbered filenames
                                            all_uploads_successful = True
                                            
                                            for job_idx, job_item in enumerate(job_queue):
                                                job_package = all_package_infos[job_idx]
                                                job_yaml = job_item.get("yaml_filename")
                                                file_group = job_item.get("file_group", [])
                                                
                                                terminal_output.append(f"[UPLOAD] --- Uploading Job {job_idx + 1}/{len(job_queue)} ---")
                                                
                                                # Upload config file for this job
                                                job_config_path = job_package.get('config_path')
                                                job_config_uploaded = False
                                                
                                                if job_config_path:
                                                    job_config_file = Path(job_config_path)
                                                    if job_config_file.exists():
                                                        # Safety check: Fix YAML adapter issue if present (redundant but safe)
                                                        import yaml
                                                        try:
                                                            with open(job_config_file, 'r') as f:
                                                                config = yaml.safe_load(f) or {}
                                                            
                                                            # Fix adapter issue: if adapter is set to "lora" as a string (not a path), remove it
                                                            if config.get("adapter") == "lora" and not Path(str(config.get("adapter", ""))).exists():
                                                                del config["adapter"]
                                                                terminal_output.append(f"[FIX] Removed 'adapter: lora' from config before upload")
                                                            
                                                            # Auto-adjust for small datasets to prevent empty batch errors
                                                            # Count training examples from the dataset file
                                                            dataset_path = job_package.get('dataset_path')
                                                            if dataset_path and Path(dataset_path).exists():
                                                                try:
                                                                    with open(dataset_path, 'r') as f:
                                                                        total_examples = sum(1 for line in f if line.strip())
                                                                    
                                                                    if total_examples > 0:
                                                                        min_eval_examples = 2
                                                                        val_set_size = config.get("val_set_size", 0.1)
                                                                        
                                                                        # If validation set would be too small, adjust it
                                                                        if total_examples * val_set_size < min_eval_examples:
                                                                            if total_examples < 50:
                                                                                # Very small dataset: disable validation entirely
                                                                                config["val_set_size"] = 0.0
                                                                                terminal_output.append(f"[FIX] Dataset has only {total_examples} examples. Disabled validation set.")
                                                                            elif total_examples < 200:
                                                                                # Small dataset: disable sample packing for stability
                                                                                if config.get("sample_packing", False):
                                                                                    config["sample_packing"] = False
                                                                                    terminal_output.append(f"[FIX] Dataset has {total_examples} examples. Disabled sample_packing.")
                                                                                # Ensure val_set_size results in at least min_eval_examples
                                                                                min_val_size = min_eval_examples / total_examples
                                                                                if val_set_size < min_val_size:
                                                                                    config["val_set_size"] = min_val_size
                                                                                    terminal_output.append(f"[FIX] Adjusted val_set_size to {min_val_size:.3f} to ensure at least {min_eval_examples} eval examples.")
                                                                            else:
                                                                                # Medium dataset: just ensure val_set_size is reasonable
                                                                                min_val_size = min_eval_examples / total_examples
                                                                                if val_set_size < min_val_size:
                                                                                    config["val_set_size"] = min_val_size
                                                                                    terminal_output.append(f"[FIX] Adjusted val_set_size to {min_val_size:.3f} to ensure at least {min_eval_examples} eval examples.")
                                                                except Exception as e:
                                                                    # If we can't count examples, continue without adjustment
                                                                    pass
                                                            
                                                            # Write fixed config back if any changes were made
                                                            with open(job_config_file, 'w') as f:
                                                                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                                                        except Exception as e:
                                                            # If YAML parsing fails, log but continue (config might already be fixed)
                                                            terminal_output.append(f"[WARNING] Could not verify YAML config: {str(e)[:100]}")
                                                        
                                                        config_size = job_config_file.stat().st_size
                                                        remote_config_name = f"axolotl_config_{job_idx}.yaml"
                                                        terminal_output.append(f"[SCP] Uploading config: {job_config_file.name} ({config_size:,} bytes)")
                                                        terminal_output.append(f"[SCP]   → /workspace/data/{remote_config_name}")
                                                        
                                                        scp_config_cmd = [
                                                            "scp", "-P", str(ssh_port), 
                                                            "-o", "StrictHostKeyChecking=no", 
                                                            "-o", "ConnectTimeout=30",
                                                            "-o", "UserKnownHostsFile=/dev/null",
                                                            str(job_config_path),
                                                            f"root@{ssh_host}:/workspace/data/{remote_config_name}"
                                                        ]
                                                        
                                                        # Retry logic for SCP uploads (Vast.ai connections can be flaky)
                                                        import time
                                                        config_upload_success = False
                                                        for retry in range(3):
                                                            if retry > 0:
                                                                wait_time = 2 ** retry  # Exponential backoff: 2, 4 seconds
                                                                terminal_output.append(f"[SCP] Retry {retry}/3 after {wait_time}s wait...")
                                                                time.sleep(wait_time)
                                                            
                                                            scp_config_result = subprocess.run(scp_config_cmd, capture_output=True, text=True, timeout=300)
                                                            if scp_config_result.returncode == 0:
                                                                job_config_uploaded = True
                                                                config_upload_success = True
                                                                terminal_output.append(f"[SUCCESS] Config uploaded: {remote_config_name}")
                                                                break
                                                            else:
                                                                error_output = scp_config_result.stderr or scp_config_result.stdout
                                                                error_output = filter_malloc_warnings(error_output)
                                                                if "Welcome to vast.ai" in error_output:
                                                                    lines = error_output.split('\n')
                                                                    actual_errors = [line for line in lines if line and 'Welcome to vast.ai' not in line and 'Have fun!' not in line]
                                                                    error_output = '\n'.join(actual_errors) if actual_errors else error_output
                                                                if retry < 2:
                                                                    terminal_output.append(f"[SCP] Config upload failed (attempt {retry + 1}/3): {error_output[:200]}")
                                                                else:
                                                                    terminal_output.append(f"[ERROR] Config upload failed after 3 attempts: {error_output[:300]}")
                                                        
                                                        if not config_upload_success:
                                                            all_uploads_successful = False
                                                    else:
                                                        terminal_output.append(f"[ERROR] Config file not found: {job_config_path}")
                                                        all_uploads_successful = False
                                                else:
                                                    terminal_output.append(f"[WARNING] No config path specified for job {job_idx + 1}")
                                                    all_uploads_successful = False
                                                
                                                # Upload dataset file for this job
                                                job_dataset_path = job_package.get('dataset_path')
                                                job_dataset_uploaded = False
                                                
                                                if job_dataset_path:
                                                    job_dataset_file = Path(job_dataset_path)
                                                    if job_dataset_file.exists():
                                                        dataset_size = job_dataset_file.stat().st_size
                                                        remote_dataset_name = f"training_data_{job_idx}.jsonl"
                                                        terminal_output.append(f"[SCP] Uploading training data: {job_dataset_file.name} ({dataset_size:,} bytes)")
                                                        terminal_output.append(f"[SCP]   → /workspace/data/{remote_dataset_name}")
                                                        
                                                        scp_cmd = [
                                                            "scp", "-P", str(ssh_port), 
                                                            "-o", "StrictHostKeyChecking=no", 
                                                            "-o", "ConnectTimeout=30",
                                                            "-o", "UserKnownHostsFile=/dev/null",
                                                            str(job_dataset_path),
                                                            f"root@{ssh_host}:/workspace/data/{remote_dataset_name}"
                                                        ]
                                                        
                                                        # Retry logic for SCP uploads (Vast.ai connections can be flaky)
                                                        dataset_upload_success = False
                                                        for retry in range(3):
                                                            if retry > 0:
                                                                wait_time = 2 ** retry  # Exponential backoff: 2, 4 seconds
                                                                terminal_output.append(f"[SCP] Retry {retry}/3 after {wait_time}s wait...")
                                                                time.sleep(wait_time)
                                                            
                                                            scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
                                                            if scp_result.returncode == 0:
                                                                job_dataset_uploaded = True
                                                                dataset_upload_success = True
                                                                terminal_output.append(f"[SUCCESS] Training data uploaded: {remote_dataset_name}")
                                                                break
                                                            else:
                                                                error_output = scp_result.stderr or scp_result.stdout
                                                                error_output = filter_malloc_warnings(error_output)
                                                                if "Welcome to vast.ai" in error_output:
                                                                    lines = error_output.split('\n')
                                                                    actual_errors = [line for line in lines if line and 'Welcome to vast.ai' not in line and 'Have fun!' not in line]
                                                                    error_output = '\n'.join(actual_errors) if actual_errors else error_output
                                                                if retry < 2:
                                                                    terminal_output.append(f"[SCP] Training data upload failed (attempt {retry + 1}/3): {error_output[:200]}")
                                                                else:
                                                                    terminal_output.append(f"[ERROR] Training data upload failed after 3 attempts: {error_output[:300]}")
                                                        
                                                        if not dataset_upload_success:
                                                            all_uploads_successful = False
                                                    else:
                                                        terminal_output.append(f"[ERROR] Training data file not found: {job_dataset_path}")
                                                        all_uploads_successful = False
                                                else:
                                                    terminal_output.append(f"[WARNING] No dataset path specified for job {job_idx + 1}")
                                                    all_uploads_successful = False
                                                
                                                if not (job_config_uploaded and job_dataset_uploaded):
                                                    all_uploads_successful = False
                                            
                                            # Check if all uploads succeeded
                                            if all_uploads_successful:
                                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] ===== Upload Summary =====")
                                                terminal_output.append(f"[SUCCESS] All {len(job_queue)} job(s) uploaded successfully!")
                                                
                                                for job_idx, job_item in enumerate(job_queue):
                                                    job_yaml = job_item.get("yaml_filename")
                                                    file_group = job_item.get("file_group", [])
                                                    yaml_desc = job_yaml if job_yaml else "stock/default config"
                                                    terminal_output.append(f"[SUCCESS] Job {job_idx + 1}: {len(file_group)} file(s), YAML: {yaml_desc}")
                                                    terminal_output.append(f"[SUCCESS]   Config: axolotl_config_{job_idx}.yaml")
                                                    terminal_output.append(f"[SUCCESS]   Data: training_data_{job_idx}.jsonl")
                                                
                                                terminal_output.append(f"[SUCCESS] ========================================")
                                                terminal_output.append(f"[INFO] Phase 2 complete! Ready to proceed to training.")
                                                active_job["files_uploaded"] = True
                                                training_manager._save_job(active_job)
                                                
                                                # Save terminal output before rerun
                                                st.session_state[terminal_output_key] = terminal_output
                                                
                                                # Auto-advance to Phase 3 when upload is successful
                                                st.session_state[phase_key] = 3
                                                st.success("✅ Files uploaded successfully! Advancing to Phase 3...")
                                                st.rerun()
                                            else:
                                                terminal_output.append(f"[ERROR] File upload failed - check errors above")
                                                terminal_output.append(f"[INFO] Please check the errors above and retry if needed.")
                                    
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.rerun()
                                except Exception as e:
                                    error_msg = str(e)
                                    terminal_output.append(f"[ERROR] {error_msg}")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.error(f"Error: {error_msg}")
                        else:
                            # Files uploaded - show status
                            st.info("✅ Files uploaded")
                    
                    with col2:
                        if st.button("🔄 Redo Phase", key="retry_phase_2"):
                            try:
                                # Clear terminal before redoing phase
                                st.session_state[terminal_output_key] = []
                                terminal_output = []
                                
                                instance_id = active_job.get("instance_id")
                                if instance_id:
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Redoing Phase 2 - cleaning up uploaded files...")
                                    
                                    # Get SSH info - prefer saved SSH details from job over API
                                    ssh_host = active_job.get("ssh_host")
                                    ssh_port = active_job.get("ssh_port", 22)
                                    
                                    # If not in job, get from API
                                    if not ssh_host:
                                        job_status = training_manager.get_job_status(instance_id)
                                        ssh_host = job_status.get("ssh_host")
                                        ssh_port = job_status.get("ssh_port", 22)
                                        # Save to job for future use
                                        if ssh_host:
                                            active_job["ssh_host"] = ssh_host
                                            active_job["ssh_port"] = ssh_port
                                            training_manager._save_job(active_job)
                                    
                                    if ssh_host:
                                        terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                        import subprocess
                                        import time
                                        
                                        # Delete uploaded files and directories with retry logic
                                        cleanup_success = False
                                        for retry in range(3):
                                            if retry > 0:
                                                wait_time = 2 ** retry  # Exponential backoff: 2, 4 seconds
                                                terminal_output.append(f"[SSH] Retry {retry}/3 after {wait_time}s wait...")
                                                time.sleep(wait_time)
                                            
                                            cleanup_cmd = [
                                                "ssh", "-p", str(ssh_port), 
                                                "-o", "StrictHostKeyChecking=no", 
                                                "-o", "ConnectTimeout=30",
                                                "-o", "UserKnownHostsFile=/dev/null",
                                                f"root@{ssh_host}",
                                                "rm -rf /workspace/data/* /workspace/output/training/* && echo 'Cleanup complete'"
                                            ]
                                            cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30)
                                            if cleanup_result.returncode == 0:
                                                terminal_output.append(f"[SSH] Files and directories deleted successfully")
                                                cleanup_success = True
                                                break
                                            else:
                                                stderr_filtered = filter_malloc_warnings(cleanup_result.stderr)
                                                if retry < 2:
                                                    terminal_output.append(f"[SSH] Cleanup failed (attempt {retry + 1}/3): {stderr_filtered[:200]}")
                                                else:
                                                    terminal_output.append(f"[SSH] Cleanup failed after 3 attempts: {stderr_filtered[:200]}")
                                        
                                        # Recreate directories with retry logic
                                        mkdir_success = False
                                        for retry in range(3):
                                            if retry > 0:
                                                wait_time = 2 ** retry  # Exponential backoff: 2, 4 seconds
                                                terminal_output.append(f"[SSH] Retry {retry}/3 after {wait_time}s wait...")
                                                time.sleep(wait_time)
                                            
                                            mkdir_cmd = [
                                                "ssh", "-p", str(ssh_port), 
                                                "-o", "StrictHostKeyChecking=no", 
                                                "-o", "ConnectTimeout=30",
                                                "-o", "UserKnownHostsFile=/dev/null",
                                                f"root@{ssh_host}",
                                                "mkdir -p /workspace/data && mkdir -p /workspace/output/training && echo 'Directories recreated'"
                                            ]
                                            mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=30)
                                            if mkdir_result.returncode == 0:
                                                terminal_output.append(f"[SSH] Directories recreated successfully")
                                                mkdir_success = True
                                                break
                                            else:
                                                stderr_filtered = filter_malloc_warnings(mkdir_result.stderr)
                                                if retry < 2:
                                                    terminal_output.append(f"[SSH] Directory recreation failed (attempt {retry + 1}/3): {stderr_filtered[:200]}")
                                                else:
                                                    terminal_output.append(f"[SSH] Directory recreation failed after 3 attempts: {stderr_filtered[:200]}")
                                        
                                        if cleanup_success and mkdir_success:
                                            terminal_output.append(f"[SUCCESS] Phase 2 cleanup complete. Ready to upload files.")
                                        else:
                                            terminal_output.append(f"[WARNING] Some cleanup operations failed. You may need to retry.")
                                    else:
                                        terminal_output.append(f"[WARNING] SSH host not available - will clean up on next upload")
                                    
                                    # Reset files_uploaded flag
                                    if active_job.get("files_uploaded"):
                                        active_job["files_uploaded"] = False
                                        training_manager._save_job(active_job)
                                    
                                    terminal_output.append(f"[SUCCESS] Phase 2 reset - ready to upload files again")
                                    
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.success("✅ Phase 2 reset. You can now upload files again.")
                                    st.rerun()
                                else:
                                    terminal_output.append(f"[ERROR] No instance ID found")
                                    st.error("No instance ID found")
                                    st.session_state[terminal_output_key] = terminal_output
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] Failed to redo phase: {error_msg}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                    
                    with col3:
                        if active_job.get("files_uploaded"):
                            st.success("✅ Files uploaded! Click 'Next Phase' to continue.")
                            if st.button("➡️ Next Phase", key="next_phase_2", type="primary"):
                                st.session_state[phase_key] = 3
                                # Clear terminal for next phase
                                st.session_state[terminal_output_key] = []
                                st.rerun()
                        else:
                            st.info("Upload files to proceed to training phase")
                
                # Phase 3: Do Training
                elif current_phase == 3:
                    # Ensure datetime is available in this scope
                    from datetime import datetime
                    
                    st.markdown(f"### {phases[3]['icon']} Phase 3: {phases[3]['name']}")
                    st.caption(phases[3]['description'])
                    
                    # Get job queue info for initialization
                    job_queue = active_job.get("job_queue")
                    current_job_index = active_job.get("current_job_index")
                    
                    # Initialize Phase 3: Ensure current job's files are active on instance
                    phase3_init_key = f"phase3_init_{active_job.get('instance_id')}_{current_job_index}"
                    if phase3_init_key not in st.session_state and job_queue and current_job_index is not None:
                        try:
                            import subprocess
                            instance_id = active_job.get("instance_id")
                            # Get SSH info - prefer saved SSH details from job over API
                            ssh_host = active_job.get("ssh_host")
                            ssh_port = active_job.get("ssh_port", 22)
                            
                            # If not in job, get from API
                            if not ssh_host:
                                job_status = training_manager.get_job_status(instance_id)
                                ssh_host = job_status.get("ssh_host")
                                ssh_port = job_status.get("ssh_port", 22)
                                # Save to job for future use
                                if ssh_host:
                                    active_job["ssh_host"] = ssh_host
                                    active_job["ssh_port"] = ssh_port
                                    training_manager._save_job(active_job)
                            
                            if ssh_host:
                                if not terminal_output:
                                    terminal_output = []
                                
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Phase 3 for job {current_job_index + 1}...")
                                terminal_output.append(f"[SSH] Ensuring job {current_job_index + 1} files are active on instance...")
                                
                                # Clean up any old active files and rename current job's files to active names
                                rename_cmd = [
                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                    f"root@{ssh_host}",
                                    f"cd /workspace/data && "
                                    f"rm -f axolotl_config.yaml training_data.jsonl && "
                                    f"mv axolotl_config_{current_job_index}.yaml axolotl_config.yaml 2>/dev/null || true && "
                                    f"mv training_data_{current_job_index}.jsonl training_data.jsonl 2>/dev/null || true && "
                                    f"echo 'Files activated for job {current_job_index + 1}'"
                                ]
                                rename_result = subprocess.run(rename_cmd, capture_output=True, text=True, timeout=30)
                                
                                if rename_result.returncode == 0:
                                    terminal_output.append(f"[SUCCESS] Job {current_job_index + 1} files are now active")
                                    
                                    # Verify and fix config file (dataset_preparation_path, tokenizer_type, etc.)
                                    terminal_output.append(f"[SSH] Verifying and fixing config file...")
                                    fix_config_remote_cmd = (
                                        f"cd /workspace/data && "
                                        f"python3 << 'PYTHON_EOF'\n"
                                        f"import yaml\n"
                                        f"import os\n"
                                        f"import sys\n"
                                        f"try:\n"
                                        f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                        f"        config = yaml.safe_load(f) or {{}}\n"
                                        f"    \n"
                                        f"    fixed = False\n"
                                        f"    \n"
                                        f"    # Always set dataset_preparation_path to absolute path (force fix)\n"
                                        f"    dataset_prep_path = config.get('dataset_preparation_path', './prepared_data')\n"
                                        f"    target_path = '/workspace/axolotl/prepared_data'\n"
                                        f"    if dataset_prep_path != target_path:\n"
                                        f"        config['dataset_preparation_path'] = target_path\n"
                                        f"        fixed = True\n"
                                        f"        print(f'Fixed dataset_preparation_path: {{dataset_prep_path}} -> {{target_path}}')\n"
                                        f"    else:\n"
                                        f"        print(f'Dataset preparation path already correct: {{target_path}}')\n"
                                        f"    \n"
                                                                f"    # Ensure the directory exists and create subdirectories\n"
                                                                f"    os.makedirs(target_path, exist_ok=True)\n"
                                                                f"    # Also create the last_run_prepared subdirectory that Axolotl uses for lock files\n"
                                                                f"    last_run_dir = os.path.join(target_path, 'last_run_prepared')\n"
                                                                f"    os.makedirs(last_run_dir, exist_ok=True)\n"
                                                                f"    # Also create it in the working directory as a fallback (Axolotl runs from /workspace/axolotl)\n"
                                                                f"    working_dir_fallback = '/workspace/axolotl/last_run_prepared'\n"
                                                                f"    os.makedirs(working_dir_fallback, exist_ok=True)\n"
                                                                f"    print(f'Ensured directory exists: {{target_path}}')\n"
                                                                f"    print(f'Created last_run_prepared subdirectory: {{last_run_dir}}')\n"
                                                                f"    print(f'Created fallback last_run_prepared in working dir: {{working_dir_fallback}}')\n"
                                        f"    \n"
                                        f"    # Fix tokenizer_type if needed\n"
                                        f"    tokenizer_type = config.get('tokenizer_type', '')\n"
                                        f"    base_model = config.get('base_model', '').lower()\n"
                                        f"    \n"
                                        f"    if 'gemma' in base_model and tokenizer_type != 'GemmaTokenizer':\n"
                                        f"        config['tokenizer_type'] = 'GemmaTokenizer'\n"
                                        f"        fixed = True\n"
                                        f"        print('Fixed tokenizer_type to GemmaTokenizer')\n"
                                        f"    elif 'mistral' in base_model and tokenizer_type not in ['MistralTokenizer', 'AutoTokenizer']:\n"
                                        f"        config['tokenizer_type'] = 'MistralTokenizer'\n"
                                        f"        fixed = True\n"
                                        f"        print('Fixed tokenizer_type to MistralTokenizer')\n"
                                        f"    elif 'phi' in base_model and tokenizer_type != 'PhiTokenizer':\n"
                                        f"        config['tokenizer_type'] = 'PhiTokenizer'\n"
                                        f"        fixed = True\n"
                                        f"        print('Fixed tokenizer_type to PhiTokenizer')\n"
                                        f"    elif 'qwen' in base_model and tokenizer_type != 'Qwen2Tokenizer':\n"
                                        f"        config['tokenizer_type'] = 'Qwen2Tokenizer'\n"
                                        f"        fixed = True\n"
                                        f"        print('Fixed tokenizer_type to Qwen2Tokenizer')\n"
                                        f"    \n"
                                        f"    # Fix adapter issue: if adapter is set to 'lora' as a string (not a path), remove it\n"
                                        f"    # Axolotl will infer LoRA from lora_* parameters, and 'lora' as a string causes it to look for adapter files\n"
                                        f"    if config.get('adapter') == 'lora':\n"
                                        f"        import os\n"
                                        f"        adapter_path = str(config.get('adapter', ''))\n"
                                        f"        if not os.path.exists(adapter_path):\n"
                                        f"            del config['adapter']\n"
                                        f"            fixed = True\n"
                                        f"            print('Removed adapter: lora (LoRA will be inferred from lora_* parameters)')\n"
                                        f"    \n"
                                        f"    # Fix quantization/adapter conflict: if quantization is enabled but no valid adapter, disable quantization\n"
                                        f"    # Axolotl requires an adapter field when quantization is enabled, but setting adapter: 'lora' causes path issues\n"
                                        f"    has_lora = config.get('lora_r') is not None\n"
                                        f"    has_quantization = config.get('load_in_4bit') or config.get('load_in_8bit')\n"
                                        f"    has_adapter = 'adapter' in config and config.get('adapter') != 'lora'\n"
                                        f"    \n"
                                        f"    if has_lora and has_quantization and not has_adapter:\n"
                                        f"        # Disable quantization to avoid adapter requirement - LoRA works fine without quantization\n"
                                        f"        if config.get('load_in_8bit'):\n"
                                        f"            config['load_in_8bit'] = False\n"
                                        f"            fixed = True\n"
                                        f"            print('Disabled load_in_8bit (required adapter field would cause issues)')\n"
                                        f"        if config.get('load_in_4bit'):\n"
                                        f"            config['load_in_4bit'] = False\n"
                                        f"            fixed = True\n"
                                        f"            print('Disabled load_in_4bit (required adapter field would cause issues)')\n"
                                        f"    \n"
                                        f"    # Fix model_type if needed\n"
                                        f"    model_type = config.get('model_type', '')\n"
                                        f"    if 'gemma' in base_model:\n"
                                        f"        # Gemma 3 uses different architecture - remove model_type to let auto-detect\n"
                                        f"        if 'gemma-3' in base_model or 'gemma3' in base_model:\n"
                                        f"            if 'model_type' in config:\n"
                                        f"                del config['model_type']\n"
                                        f"                fixed = True\n"
                                        f"                print('Removed model_type for Gemma 3 (auto-detect)')\n"
                                        f"        elif 'GemmaForCausalLM' not in model_type:\n"
                                        f"            config['model_type'] = 'GemmaForCausalLM'\n"
                                        f"            fixed = True\n"
                                        f"            print('Fixed model_type to GemmaForCausalLM')\n"
                                        f"    elif 'mistral' in base_model and 'MistralForCausalLM' not in model_type:\n"
                                        f"        config['model_type'] = 'MistralForCausalLM'\n"
                                        f"        fixed = True\n"
                                        f"        print('Fixed model_type to MistralForCausalLM')\n"
                                        f"    \n"
                                        f"    # Auto-adjust for small datasets to prevent empty batch errors\n"
                                        f"    # Count training examples from the dataset file\n"
                                        f"    import json\n"
                                        f"    try:\n"
                                        f"        datasets = config.get('datasets', [])\n"
                                        f"        dataset_path = datasets[0].get('path', '/workspace/data/training_data.jsonl') if datasets else '/workspace/data/training_data.jsonl'\n"
                                        f"        if os.path.exists(dataset_path):\n"
                                        f"            with open(dataset_path, 'r') as f:\n"
                                        f"                total_examples = sum(1 for line in f if line.strip())\n"
                                        f"            \n"
                                        f"            if total_examples > 0:\n"
                                        f"                min_eval_examples = 2\n"
                                        f"                val_set_size = config.get('val_set_size', 0.1)\n"
                                        f"                \n"
                                        f"                # If validation set would be too small, adjust it\n"
                                        f"                if total_examples * val_set_size < min_eval_examples:\n"
                                        f"                    if total_examples < 50:\n"
                                        f"                        # Very small dataset: disable validation entirely\n"
                                        f"                        config['val_set_size'] = 0.0\n"
                                        f"                        fixed = True\n"
                                        f"                        print(f'Dataset has only {{total_examples}} examples. Disabled validation set.')\n"
                                        f"                    elif total_examples < 200:\n"
                                        f"                        # Small dataset: disable sample packing for stability\n"
                                        f"                        if config.get('sample_packing', False):\n"
                                        f"                            config['sample_packing'] = False\n"
                                        f"                            fixed = True\n"
                                        f"                            print(f'Dataset has {{total_examples}} examples. Disabled sample_packing.')\n"
                                        f"                        # Ensure val_set_size results in at least min_eval_examples\n"
                                        f"                        min_val_size = min_eval_examples / total_examples\n"
                                        f"                        if val_set_size < min_val_size:\n"
                                        f"                            config['val_set_size'] = min_val_size\n"
                                        f"                            fixed = True\n"
                                        f"                            print(f'Adjusted val_set_size to {{min_val_size:.3f}} to ensure at least {{min_eval_examples}} eval examples.')\n"
                                        f"                    else:\n"
                                        f"                        # Medium dataset: just ensure val_set_size is reasonable\n"
                                        f"                        min_val_size = min_eval_examples / total_examples\n"
                                        f"                        if val_set_size < min_val_size:\n"
                                        f"                            config['val_set_size'] = min_val_size\n"
                                        f"                            fixed = True\n"
                                        f"                            print(f'Adjusted val_set_size to {{min_val_size:.3f}} to ensure at least {{min_eval_examples}} eval examples.')\n"
                                        f"    except Exception as e:\n"
                                        f"        # If we can't count examples, continue without adjustment\n"
                                        f"        pass\n"
                                        f"    \n"
                                        f"    if fixed:\n"
                                        f"        with open('axolotl_config.yaml', 'w') as f:\n"
                                        f"            yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                        f"        print('Config file updated successfully')\n"
                                        f"    else:\n"
                                        f"        print('Config file already correct')\n"
                                        f"except Exception as e:\n"
                                        f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                        f"    sys.exit(1)\n"
                                        f"PYTHON_EOF"
                                    )
                                    fix_config_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        fix_config_remote_cmd
                                    ]
                                    fix_config_result = subprocess.run(fix_config_cmd, capture_output=True, text=True, timeout=30)
                                    
                                    if fix_config_result.returncode == 0:
                                        stdout_filtered = filter_malloc_warnings(fix_config_result.stdout)
                                        if "updated successfully" in fix_config_result.stdout:
                                            terminal_output.append(f"[SUCCESS] Config file verified and fixed: {stdout_filtered.strip()}")
                                        elif "already correct" in fix_config_result.stdout:
                                            terminal_output.append(f"[INFO] Config file verified: {stdout_filtered.strip()}")
                                        else:
                                            terminal_output.append(f"[INFO] Config check completed: {stdout_filtered.strip()}")
                                    else:
                                        error_msg = filter_malloc_warnings(fix_config_result.stderr or fix_config_result.stdout)
                                        terminal_output.append(f"[WARNING] Config verification failed: {error_msg[:200]}")
                                    
                                    # Update package_info to point to current job's package
                                    all_package_infos = active_job.get("all_package_infos", [])
                                    if all_package_infos and current_job_index < len(all_package_infos):
                                        active_job["package_info"] = all_package_infos[current_job_index]
                                        # Update YAML config in package_info for display
                                        current_job = job_queue[current_job_index]
                                        if current_job.get("yaml_path"):
                                            from pathlib import Path
                                            yaml_filename = Path(current_job.get("yaml_path")).name
                                            active_job["package_info"]["yaml_config"] = yaml_filename
                                        else:
                                            active_job["package_info"]["yaml_config"] = None
                                        training_manager._save_job(active_job)
                                else:
                                    error_output = filter_malloc_warnings(rename_result.stderr or rename_result.stdout)
                                    terminal_output.append(f"[WARNING] File rename result: {error_output[:200]}")
                                
                                st.session_state[terminal_output_key] = terminal_output
                                st.session_state[phase3_init_key] = True
                                st.rerun()
                        except Exception as e:
                            if not terminal_output:
                                terminal_output = []
                            terminal_output.append(f"[WARNING] Phase 3 initialization error: {str(e)}")
                            st.session_state[terminal_output_key] = terminal_output
                            st.session_state[phase3_init_key] = True
                    
                    # YAML Debugging - Check at beginning of Phase 3
                    phase3_yaml_debug_key = f"phase3_yaml_debug_{active_job.get('instance_id')}_{current_job_index}"
                    if phase3_yaml_debug_key not in st.session_state:
                        try:
                            from datetime import datetime
                            import subprocess
                            
                            # Initialize terminal output if needed
                            if not terminal_output:
                                terminal_output = []
                            
                            terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] === YAML Configuration Debugging ===")
                            
                            # 1. Check if YAML was supposed to be included
                            package_info = active_job.get("package_info", {})
                            expected_yaml = package_info.get("yaml_config")
                            expected_yaml_from_queue = None
                            expected_yaml_path = None
                            
                            if job_queue and current_job_index is not None:
                                current_job = job_queue[current_job_index]
                                expected_yaml_from_queue = current_job.get("yaml_filename")
                                expected_yaml_path = current_job.get("yaml_path")
                                terminal_output.append(f"[YAML DEBUG] Job queue index: {current_job_index}")
                                terminal_output.append(f"[YAML DEBUG] Expected YAML from queue: {expected_yaml_from_queue if expected_yaml_from_queue else 'None'}")
                                terminal_output.append(f"[YAML DEBUG] Expected YAML path: {expected_yaml_path if expected_yaml_path else 'None'}")
                            else:
                                terminal_output.append(f"[YAML DEBUG] No job queue or index - checking package_info")
                                terminal_output.append(f"[YAML DEBUG] Expected YAML from package_info: {expected_yaml if expected_yaml else 'None'}")
                            
                            yaml_should_be_included = bool(expected_yaml or expected_yaml_from_queue)
                            terminal_output.append(f"[YAML DEBUG] ========================================")
                            terminal_output.append(f"[YAML DEBUG] YAML supposed to be included: {'YES ✓' if yaml_should_be_included else 'NO ✗'}")
                            if yaml_should_be_included:
                                terminal_output.append(f"[YAML DEBUG] Expected YAML filename: {expected_yaml_from_queue or expected_yaml or 'Unknown'}")
                            
                            # 2. Check if YAML file exists on remote instance
                            instance_id = active_job.get("instance_id")
                            # Get SSH info - prefer saved SSH details from job over API
                            ssh_host = active_job.get("ssh_host")
                            ssh_port = active_job.get("ssh_port", 22)
                            
                            # If not in job, get from API
                            if not ssh_host:
                                job_status = training_manager.get_job_status(instance_id)
                                ssh_host = job_status.get("ssh_host")
                                ssh_port = job_status.get("ssh_port", 22)
                                # Save to job for future use
                                if ssh_host:
                                    active_job["ssh_host"] = ssh_host
                                    active_job["ssh_port"] = ssh_port
                                    training_manager._save_job(active_job)
                            
                            yaml_found_on_remote = False
                            yaml_file_path = None
                            
                            if ssh_host:
                                # Check if YAML file exists on remote (could be in /workspace/data/)
                                yaml_to_check = expected_yaml_from_queue or expected_yaml
                                if yaml_to_check:
                                    terminal_output.append(f"[YAML DEBUG] Checking for YAML file on remote: {yaml_to_check}")
                                    check_yaml_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        f"test -f /workspace/data/{yaml_to_check} && echo 'found' || echo 'not_found'"
                                    ]
                                    yaml_check_result = subprocess.run(check_yaml_cmd, capture_output=True, text=True, timeout=15)
                                    if "found" in yaml_check_result.stdout:
                                        yaml_found_on_remote = True
                                        yaml_file_path = f"/workspace/data/{yaml_to_check}"
                                        terminal_output.append(f"[YAML DEBUG] YAML file found on remote: ✓ {yaml_file_path}")
                                    else:
                                        terminal_output.append(f"[YAML DEBUG] YAML file NOT found on remote: ✗ /workspace/data/{yaml_to_check}")
                                else:
                                    terminal_output.append(f"[YAML DEBUG] No YAML expected - skipping remote file check")
                                
                                terminal_output.append(f"[YAML DEBUG] ========================================")
                                terminal_output.append(f"[YAML DEBUG] YAML file uploaded: {'YES ✓' if yaml_found_on_remote else 'NO ✗' if yaml_should_be_included else 'N/A (not expected)'}")
                                
                                # Also check config file to see if it references YAML settings
                                check_config_cmd = [
                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                    f"root@{ssh_host}",
                                    "test -f /workspace/data/axolotl_config.yaml && cat /workspace/data/axolotl_config.yaml | head -50 || echo 'config_not_found'"
                                ]
                                config_check_result = subprocess.run(check_config_cmd, capture_output=True, text=True, timeout=15)
                                
                                if "config_not_found" not in config_check_result.stdout:
                                    config_content = config_check_result.stdout
                                    terminal_output.append(f"[YAML DEBUG] Config file exists on remote")
                                    
                                    # Check if config has YAML-specific settings (indicating YAML was applied)
                                    import yaml as yaml_lib
                                    try:
                                        config_data = yaml_lib.safe_load(config_content)
                                        if config_data:
                                            # Check for common YAML-injected settings
                                            has_datasets = "datasets" in config_data
                                            has_output_dir = "output_dir" in config_data
                                            has_base_model = "base_model" in config_data
                                            
                                            terminal_output.append(f"[YAML DEBUG] Config contains datasets: {has_datasets}")
                                            terminal_output.append(f"[YAML DEBUG] Config contains output_dir: {has_output_dir}")
                                            terminal_output.append(f"[YAML DEBUG] Config contains base_model: {has_base_model}")
                                            
                                            # If YAML was supposed to be included, check if custom settings are present
                                            if yaml_should_be_included:
                                                # YAML configs typically have more settings than basic configs
                                                # Check for any non-standard settings that would indicate YAML was applied
                                                standard_keys = {"datasets", "output_dir", "base_model", "base_model_config", "dataset_preparation_path"}
                                                custom_keys = set(config_data.keys()) - standard_keys
                                                if custom_keys:
                                                    terminal_output.append(f"[YAML DEBUG] ========================================")
                                                    terminal_output.append(f"[YAML DEBUG] YAML applied successfully: ✓")
                                                    terminal_output.append(f"[YAML DEBUG] Found custom settings: {', '.join(list(custom_keys)[:5])}")
                                                    if len(custom_keys) > 5:
                                                        terminal_output.append(f"[YAML DEBUG] ... and {len(custom_keys) - 5} more custom settings")
                                                else:
                                                    terminal_output.append(f"[YAML DEBUG] ========================================")
                                                    terminal_output.append(f"[YAML DEBUG] YAML may not have been applied: ✗")
                                                    terminal_output.append(f"[YAML DEBUG] Only standard settings found (no custom YAML settings detected)")
                                            else:
                                                terminal_output.append(f"[YAML DEBUG] ========================================")
                                                terminal_output.append(f"[YAML DEBUG] No YAML expected, using standard config: N/A")
                                    except Exception as e:
                                        terminal_output.append(f"[YAML DEBUG] Error parsing config: {str(e)}")
                                else:
                                    terminal_output.append(f"[YAML DEBUG] Config file NOT found on remote")
                            else:
                                terminal_output.append(f"[YAML DEBUG] SSH not available - cannot check remote files")
                            
                            terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] === End YAML Debugging ===")
                            
                            st.session_state[terminal_output_key] = terminal_output
                            st.session_state[phase3_yaml_debug_key] = True
                            st.rerun()
                        except Exception as e:
                            # If debugging fails, just mark as done and continue
                            if not terminal_output:
                                terminal_output = []
                            terminal_output.append(f"[YAML DEBUG] Error during YAML debugging: {str(e)}")
                            st.session_state[terminal_output_key] = terminal_output
                            st.session_state[phase3_yaml_debug_key] = True
                    
                    # Terminal output area (scrollable)
                    st.markdown("#### Terminal Output")
                    terminal_container = st.container()
                    with terminal_container:
                        if terminal_output:
                            # Keep only last 200 lines
                            display_output = terminal_output[-200:] if len(terminal_output) > 200 else terminal_output
                            output_text = "\n".join(display_output)
                            st.code(output_text, language="text")
                            if len(terminal_output) > 200:
                                st.caption(f"Showing last 200 of {len(terminal_output)} lines")
                        else:
                            st.info("Terminal is empty. Click 'Check Training Status.'")
                    
                    # Action buttons
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                    with col1:
                        # Check if training is already running
                        training_status = active_job.get("training_status", {})
                        training_is_running = training_status.get("status") in ["training", "preprocessing", "completed"]
                        
                        # Check if we're in queue mode and should wait for automatic transition
                        job_queue = active_job.get("job_queue")
                        current_job_index = active_job.get("current_job_index")
                        in_queue_mode = job_queue and current_job_index is not None and len(job_queue) > 1
                        
                        # Don't show manual start button if in queue mode with multiple jobs
                        # (queue jobs should transition automatically when previous completes)
                        if in_queue_mode and current_job_index + 1 < len(job_queue):
                            # In queue mode - jobs transition automatically
                            if not training_is_running:
                                st.info(f"⏳ Waiting for job {current_job_index + 1} to complete before starting job {current_job_index + 2}...")
                            else:
                                st.info(f"🔄 Job {current_job_index + 1}/{len(job_queue)} in progress. Next job will start automatically when this completes.")
                        elif not training_is_running:
                            # Show "Start Training" button if training hasn't started
                            if st.button("▶️ Start Training", key="start_training", type="primary"):
                                try:
                                    instance_id = active_job.get("instance_id")
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
                                    
                                    # Get Hugging Face token for gated models
                                    hf_token = get_hf_token()
                                    if hf_token:
                                        terminal_output.append(f"[INFO] Using Hugging Face token for gated model access")
                                    else:
                                        terminal_output.append(f"[WARNING] No Hugging Face token found. Gated models (e.g., Gemma) may fail to load.")
                                    
                                    # Get SSH info
                                    ssh_host = active_job.get("ssh_host")
                                    ssh_port = active_job.get("ssh_port", 22)
                                    
                                    if not ssh_host:
                                        job_status = training_manager.get_job_status(instance_id)
                                        ssh_host = job_status.get("ssh_host")
                                        ssh_port = job_status.get("ssh_port", 22)
                                        if ssh_host:
                                            active_job["ssh_host"] = ssh_host
                                            active_job["ssh_port"] = ssh_port
                                            training_manager._save_job(active_job)
                                    
                                    if ssh_host:
                                        terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                        
                                        # Check if files exist
                                        import subprocess
                                        check_files_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "test -f /workspace/data/axolotl_config.yaml && test -f /workspace/data/training_data.jsonl && echo 'files_exist' || echo 'files_missing'"
                                        ]
                                        files_check = subprocess.run(check_files_cmd, capture_output=True, text=True, timeout=15)
                                        
                                        if "files_exist" in files_check.stdout:
                                            terminal_output.append(f"[SSH] Training files found - setting up environment...")
                                            
                                            # Check if there's already a training process running for this job
                                            # Only kill if we're in a queue and this is a different job, or if manually restarting
                                            job_queue = active_job.get("job_queue")
                                            current_job_index = active_job.get("current_job_index")
                                            
                                            # Check for existing processes
                                            check_processes_cmd = [
                                                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                f"root@{ssh_host}",
                                                "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | head -5 || echo 'no_processes'"
                                            ]
                                            check_processes_result = subprocess.run(check_processes_cmd, capture_output=True, text=True, timeout=15)
                                            
                                            has_existing_process = "no_processes" not in check_processes_result.stdout and check_processes_result.stdout.strip()
                                            
                                            if has_existing_process:
                                                # There's a process running - check if it's for a different job in queue
                                                if job_queue and current_job_index is not None:
                                                    # In queue mode - it's OK to kill previous job's process
                                                    terminal_output.append(f"[SSH] Found existing training process - stopping (transitioning to job {current_job_index + 1})...")
                                                else:
                                                    # Single job mode - warn but allow restart
                                                    terminal_output.append(f"[WARNING] Found existing training process. This will be stopped to start fresh training.")
                                                
                                                # Kill existing processes
                                                kill_existing_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    "pkill -9 -f accelerate || true; "
                                                    "pkill -9 -f axolotl || true; "
                                                    "ps aux | grep -E 'python.*train|python.*axolotl|python.*accelerate' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true; "
                                                    "sleep 2; "
                                                    "remaining=$(ps aux | grep -E 'accelerate|axolotl|train' | grep -v grep | wc -l); "
                                                    "echo 'Processes stopped. Remaining: '$remaining"
                                                ]
                                                kill_existing_result = subprocess.run(kill_existing_cmd, capture_output=True, text=True, timeout=30)
                                                if kill_existing_result.returncode == 0:
                                                    stdout_filtered = filter_malloc_warnings(kill_existing_result.stdout)
                                                    if stdout_filtered.strip():
                                                        terminal_output.append(f"[SSH] {stdout_filtered.strip()}")
                                            else:
                                                terminal_output.append(f"[SSH] No existing training processes found - starting fresh")
                                            
                                            # Clean output directory
                                            terminal_output.append(f"[SSH] Cleaning output directory...")
                                            cleanup_cmd = [
                                                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                f"root@{ssh_host}",
                                                "rm -rf /workspace/output/training/* && echo 'Output directory cleaned'"
                                            ]
                                            cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30)
                                            if cleanup_result.returncode == 0:
                                                terminal_output.append(f"[SSH] Output directory cleaned")
                                            
                                            # Setup script that installs dependencies and starts training
                                            # This mirrors the onstart script but runs directly
                                            # Use a here-doc to avoid quote escaping issues
                                            
                                            # Build HF token export section
                                            if hf_token:
                                                hf_token_escaped = hf_token.replace("'", "'\"'\"'")
                                                hf_token_export = f"export HF_TOKEN='{hf_token_escaped}'\nexport HUGGING_FACE_HUB_TOKEN='{hf_token_escaped}'\n"
                                            else:
                                                hf_token_export = ""
                                            
                                            # Build the script by concatenating parts
                                            script_parts = [
                                                "bash << 'TRAIN_SCRIPT'\n",
                                                "set -e\n",
                                                "cd /workspace\n",
                                                "\n",
                                                "# Check if axolotl is already installed\n",
                                                "AXOLOTL_INSTALLED=0\n",
                                                "if [ -d axolotl ]; then\n",
                                                " /opt/conda/bin/python -c 'import axolotl' 2>/dev/null && AXOLOTL_INSTALLED=1 || python3 -c 'import axolotl' 2>/dev/null && AXOLOTL_INSTALLED=1 || python -c 'import axolotl' 2>/dev/null && AXOLOTL_INSTALLED=1 || true\n",
                                                "fi\n",
                                                "\n",
                                                "if [ $AXOLOTL_INSTALLED -eq 0 ]; then\n",
                                                " echo 'Installing axolotl...'\n",
                                                " if [ ! -d axolotl ]; then\n",
                                                "  git clone https://github.com/OpenAccess-AI-Collective/axolotl.git || true\n",
                                                " fi\n",
                                                " cd axolotl\n",
                                                " /opt/conda/bin/pip install -e . || pip3 install -e . || pip install -e . || true\n",
                                                "else\n",
                                                " echo 'Axolotl already installed, skipping...'\n",
                                                " cd axolotl || (git clone https://github.com/OpenAccess-AI-Collective/axolotl.git && cd axolotl)\n",
                                                "fi\n",
                                                "\n",
                                                "# Check and install dependencies\n",
                                                "echo 'Checking dependencies...'\n",
                                                "PYTHON_CMD='/opt/conda/bin/python'\n",
                                                "DEPS_INSTALLED=0\n",
                                                "$PYTHON_CMD -c 'import accelerate' 2>/dev/null && DEPS_INSTALLED=1 || python3 -c 'import accelerate' 2>/dev/null && DEPS_INSTALLED=1 || python -c 'import accelerate' 2>/dev/null && DEPS_INSTALLED=1 || true\n",
                                                "\n",
                                                "if [ $DEPS_INSTALLED -eq 0 ]; then\n",
                                                " echo 'Installing accelerate, huggingface_hub, peft, bitsandbytes...'\n",
                                                " INSTALL_SUCCESS=0\n",
                                                " $PYTHON_CMD -m pip install accelerate huggingface_hub peft bitsandbytes && INSTALL_SUCCESS=1 || \\\n",
                                                " python3 -m pip install accelerate huggingface_hub peft bitsandbytes && INSTALL_SUCCESS=1 || \\\n",
                                                " python -m pip install accelerate huggingface_hub peft bitsandbytes && INSTALL_SUCCESS=1 || true\n",
                                                " \n",
                                                " # Verify installation worked\n",
                                                " if [ $INSTALL_SUCCESS -eq 1 ]; then\n",
                                                "  $PYTHON_CMD -c 'import accelerate' 2>/dev/null && DEPS_INSTALLED=1 || python3 -c 'import accelerate' 2>/dev/null && DEPS_INSTALLED=1 || python -c 'import accelerate' 2>/dev/null && DEPS_INSTALLED=1 || true\n",
                                                " fi\n",
                                                " \n",
                                                " if [ $DEPS_INSTALLED -eq 0 ]; then\n",
                                                "  echo 'ERROR: Failed to install accelerate'\n",
                                                "  exit 1\n",
                                                " fi\n",
                                                " echo 'Dependencies installed successfully'\n",
                                                "else\n",
                                                " echo 'Dependencies already installed, skipping...'\n",
                                                "fi\n",
                                                "\n",
                                                "# Set Hugging Face token for gated models\n",
                                                hf_token_export,
                                                "\n",
                                                "# Start training\n",
                                                "echo 'Starting training...'\n",
                                                "cd /workspace/data\n",
                                                "CF='/workspace/data/axolotl_config.yaml'\n",
                                                "cd /workspace/axolotl\n",
                                                "# Try each Python command in sequence\n",
                                                "# Check which Python has accelerate available first\n",
                                                "TRAIN_PYTHON=''\n",
                                                "if $PYTHON_CMD -c 'import accelerate' 2>/dev/null; then\n",
                                                " TRAIN_PYTHON='$PYTHON_CMD'\n",
                                                "elif python3 -c 'import accelerate' 2>/dev/null; then\n",
                                                " TRAIN_PYTHON='python3'\n",
                                                "elif python -c 'import accelerate' 2>/dev/null; then\n",
                                                " TRAIN_PYTHON='python'\n",
                                                "fi\n",
                                                "\n",
                                                "if [ -z \"$TRAIN_PYTHON\" ]; then\n",
                                                " echo 'ERROR: accelerate not found in any Python'\n",
                                                " exit 1\n",
                                                "fi\n",
                                                "\n",
                                                "# Find accelerate executable from the Python that has accelerate\n",
                                                "echo \"Starting training with $TRAIN_PYTHON\"\n",
                                                "# Get the directory containing the Python executable\n",
                                                "if [[ \"$TRAIN_PYTHON\" == /* ]]; then\n",
                                                " PYTHON_DIR=$(dirname \"$TRAIN_PYTHON\")\n",
                                                " ACCELERATE_CMD=\"$PYTHON_DIR/accelerate\"\n",
                                                "else\n",
                                                " ACCELERATE_CMD=$(command -v accelerate 2>/dev/null || echo \"accelerate\")\n",
                                                "fi\n",
                                                "# Try accelerate command, fallback to python -m accelerate.commands.launch\n",
                                                "# Note: HF_TOKEN and HUGGING_FACE_HUB_TOKEN are already exported above\n",
                                                "if [ -f \"$ACCELERATE_CMD\" ] || command -v accelerate >/dev/null 2>&1; then\n",
                                                " nohup $ACCELERATE_CMD launch -m axolotl.cli.train \"$CF\" > /workspace/output/training/training.log 2>&1 &\n",
                                                "else\n",
                                                " nohup $TRAIN_PYTHON -m accelerate.commands.launch -m axolotl.cli.train \"$CF\" > /workspace/output/training/training.log 2>&1 &\n",
                                                "fi\n",
                                                "TRAIN_PID=$!\n",
                                                "sleep 3\n",
                                                "# Verify the process is actually running\n",
                                                "if ps -p $TRAIN_PID > /dev/null 2>&1 || pgrep -f 'axolotl.cli.train' > /dev/null 2>&1; then\n",
                                                " echo \"Training started successfully (PID: $TRAIN_PID)\"\n",
                                                "else\n",
                                                " echo 'ERROR: Training process did not start'\n",
                                                " echo 'Last 20 lines of log:'\n",
                                                " tail -20 /workspace/output/training/training.log 2>/dev/null || echo 'No log file'\n",
                                                " echo 'Checking if accelerate is available:'\n",
                                                " $TRAIN_PYTHON -c 'import accelerate; print(accelerate.__version__)' 2>&1 || echo 'accelerate import failed'\n",
                                                " exit 1\n",
                                                "fi\n",
                                                "echo 'Training started'\n",
                                                "TRAIN_SCRIPT"
                                            ]
                                            setup_and_train_cmd = "".join(script_parts)
                                            
                                            # Start training in background
                                            terminal_output.append(f"[SSH] Installing dependencies and starting training...")
                                            # Write script to remote file first to avoid quote escaping issues
                                            script_content = setup_and_train_cmd
                                            
                                            # Write the script to a file on the remote server
                                            write_cmd = [
                                                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                f"root@{ssh_host}",
                                                "cat > /tmp/start_training.sh"
                                            ]
                                            write_result = subprocess.run(write_cmd, capture_output=True, text=True, timeout=30, input=script_content)
                                            
                                            if write_result.returncode == 0:
                                                # Make executable and run
                                                chmod_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=3",
                                                    f"root@{ssh_host}",
                                                    "chmod +x /tmp/start_training.sh && bash /tmp/start_training.sh 2>&1"
                                                ]
                                                training_result = subprocess.run(chmod_cmd, capture_output=True, text=True, timeout=600)
                                                
                                                # Show output from script execution
                                                if training_result.stdout:
                                                    stdout_filtered = filter_malloc_warnings(training_result.stdout)
                                                    for line in stdout_filtered.strip().split("\n")[-10:]:
                                                        if line.strip():
                                                            terminal_output.append(f"[SCRIPT] {line[:200]}")
                                                if training_result.stderr:
                                                    stderr_filtered = filter_malloc_warnings(training_result.stderr)
                                                    for line in stderr_filtered.strip().split("\n")[-10:]:
                                                        if line.strip():
                                                            terminal_output.append(f"[SCRIPT ERROR] {line[:200]}")
                                            else:
                                                error_msg = filter_malloc_warnings(write_result.stderr or write_result.stdout)
                                                terminal_output.append(f"[ERROR] Failed to write script: {error_msg[:200]}")
                                                training_result = write_result
                                            
                                            # Verify training actually started by checking for the process
                                            if training_result.returncode == 0:
                                                # Wait a moment for process to start
                                                import time
                                                time.sleep(2)
                                                
                                                # Check if training process is actually running
                                                check_process_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    "ps aux | grep -E '(axolotl|accelerate|train)' | grep -v grep | head -1 || echo 'no_training'"
                                                ]
                                                process_check = subprocess.run(check_process_cmd, capture_output=True, text=True, timeout=15)
                                                
                                                if "no_training" not in process_check.stdout and process_check.stdout.strip():
                                                    terminal_output.append(f"[SUCCESS] Training started successfully!")
                                                    terminal_output.append(f"[INFO] Training is running in the background.")
                                                    terminal_output.append(f"[INFO] Click 'Check Training Status' to monitor progress.")
                                                    
                                                    # Update job status
                                                    active_job["training_status"] = {"status": "training"}
                                                    training_manager._save_job(active_job)
                                                else:
                                                    # Script ran but training didn't start - check logs
                                                    check_log_cmd = [
                                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                        f"root@{ssh_host}",
                                                        "tail -20 /workspace/output/training/training.log 2>/dev/null || echo 'no_log'"
                                                    ]
                                                    log_check = subprocess.run(check_log_cmd, capture_output=True, text=True, timeout=15)
                                                    if "no_log" not in log_check.stdout:
                                                        log_output = filter_malloc_warnings(log_check.stdout)
                                                        terminal_output.append(f"[WARNING] Script completed but training process not found.")
                                                        terminal_output.append(f"[WARNING] Last log output:")
                                                        for line in log_output.strip().split("\n")[-5:]:
                                                            if line.strip():
                                                                terminal_output.append(f"[LOG] {line[:200]}")
                                                    terminal_output.append(f"[ERROR] Training script completed but training process did not start.")
                                                    terminal_output.append(f"[ACTION] Check the script output above and training.log for errors.")
                                                    st.error("Training script ran but training process did not start. Check terminal output above.")
                                            else:
                                                error_msg = filter_malloc_warnings(training_result.stderr or training_result.stdout)
                                                terminal_output.append(f"[ERROR] Failed to start training: {error_msg[:300]}")
                                                st.error(f"Failed to start training: {error_msg[:200]}")
                                        else:
                                            terminal_output.append(f"[ERROR] Training files not found on instance!")
                                            terminal_output.append(f"[ACTION] Please go back to Phase 2 and upload files first.")
                                            st.error("Training files not found. Please upload files in Phase 2 first.")
                                    else:
                                        terminal_output.append(f"[ERROR] SSH host not available.")
                                        st.error("SSH host not available.")
                                    
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.rerun()
                                except Exception as e:
                                    error_msg = str(e)
                                    terminal_output.append(f"[ERROR] Failed to start training: {error_msg}")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.error(f"Error: {error_msg}")
                        else:
                            # Training is running - show check button
                            status_msg = training_status.get("status", "unknown")
                            if status_msg == "training":
                                st.info("🔄 Training is currently running. Use 'Check Training Status' to monitor progress.")
                            elif status_msg == "completed":
                                st.success("✅ Training completed! Check status for details.")
                            
                            if st.button("🔄 Check Training Status", key="check_training_status"):
                                try:
                                    instance_id = active_job.get("instance_id")
                                    
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Checking training status...")
                                    
                                    # Initialize SSH check results (will be updated if SSH is available)
                                    ssh_files_exist = None
                                    ssh_training_running = None
                                    training_logs_content = None
                                    debug_log_content = None
                                    training_error = None
                                    
                                    # Get SSH info - prefer saved SSH details from job over API
                                    ssh_host = active_job.get("ssh_host")
                                    ssh_port = active_job.get("ssh_port", 22)
                                    
                                    # If not in job, get from API
                                    if not ssh_host:
                                        job_status = training_manager.get_job_status(instance_id)
                                        ssh_host = job_status.get("ssh_host")
                                        ssh_port = job_status.get("ssh_port", 22)
                                        # Save to job for future use
                                        if ssh_host:
                                            active_job["ssh_host"] = ssh_host
                                            active_job["ssh_port"] = ssh_port
                                            training_manager._save_job(active_job)
                                    
                                    if ssh_host:
                                        terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                        
                                        # Check if training process is running
                                        terminal_output.append(f"[SSH] Checking for training process...")
                                        import subprocess
                                        check_process_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "ps aux | grep -E '(axolotl|accelerate|train)' | grep -v grep || echo 'no_training'"
                                        ]
                                        process_result = subprocess.run(check_process_cmd, capture_output=True, text=True, timeout=15)
                                        if "no_training" not in process_result.stdout:
                                            stdout_filtered = filter_malloc_warnings(process_result.stdout)
                                            terminal_output.append(f"[SSH] Training process found: {stdout_filtered[:200]}")
                                        else:
                                            terminal_output.append(f"[SSH] No training process found")
                                        
                                        # Check for output directory
                                        terminal_output.append(f"[SSH] Checking output directory...")
                                        check_output_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "ls -la /workspace/output/training 2>/dev/null | tail -5 || echo 'no_output'"
                                        ]
                                        output_result = subprocess.run(check_output_cmd, capture_output=True, text=True, timeout=15)
                                        if "no_output" not in output_result.stdout:
                                            terminal_output.append(f"[SSH] Output directory exists")
                                            stdout_filtered = filter_malloc_warnings(output_result.stdout)
                                            terminal_output.append(f"[SSH] {stdout_filtered[:300]}")
                                        else:
                                            terminal_output.append(f"[SSH] Output directory not found")
                                        
                                        # Check if training files exist
                                        terminal_output.append(f"[SSH] Checking if training files are present...")
                                        check_files_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "ls -la /workspace/data/ 2>/dev/null | grep -E '(training_data|axolotl_config)' || echo 'files_missing'"
                                        ]
                                        files_result = subprocess.run(check_files_cmd, capture_output=True, text=True, timeout=15)
                                        # Capture whether files exist for diagnostics
                                        ssh_files_exist = "files_missing" not in files_result.stdout
                                        if not ssh_files_exist:
                                            terminal_output.append(f"[WARNING] Training files not found in /workspace/data/")
                                        else:
                                            stdout_filtered = filter_malloc_warnings(files_result.stdout)
                                            terminal_output.append(f"[SSH] Training files found:")
                                            for line in stdout_filtered.strip().split("\n")[:5]:
                                                if line.strip():
                                                    terminal_output.append(f"[SSH]   {line}")
                                        
                                        # Check for training processes (ignore onstart script for existing instances)
                                        terminal_output.append(f"[SSH] Checking for training processes...")
                                        check_training_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | head -5 || echo 'no_training'"
                                        ]
                                        training_process_result = subprocess.run(check_training_cmd, capture_output=True, text=True, timeout=15)
                                        ssh_training_running = "no_training" not in training_process_result.stdout
                                        if ssh_training_running:
                                            stdout_filtered = filter_malloc_warnings(training_process_result.stdout)
                                            terminal_output.append(f"[SSH] ✓ Training process detected:")
                                            for line in stdout_filtered.strip().split("\n")[:5]:
                                                if line.strip():
                                                    terminal_output.append(f"[SSH]   {line[:150]}")
                                            
                                            # Get more details about the process - show full command line
                                            check_process_cmd_full = [
                                                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                f"root@{ssh_host}",
                                                "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | head -1"
                                            ]
                                            process_cmd_full_result = subprocess.run(check_process_cmd_full, capture_output=True, text=True, timeout=15)
                                            if process_cmd_full_result.stdout and process_cmd_full_result.stdout.strip():
                                                cmd_filtered = filter_malloc_warnings(process_cmd_full_result.stdout)
                                                terminal_output.append(f"[SSH] Process details:")
                                                # Parse the ps output to show PID, status, and command
                                                parts = cmd_filtered.strip().split(None, 10)
                                                if len(parts) >= 11:
                                                    pid = parts[1]
                                                    stat = parts[7]
                                                    cmd = " ".join(parts[10:])
                                                    terminal_output.append(f"[SSH]   PID: {pid}, Status: {stat}")
                                                    terminal_output.append(f"[SSH]   Command: {cmd[:300]}")
                                                    
                                                    # Check if it's actually the training process
                                                    if "axolotl.cli.train" in cmd or ("accelerate" in cmd and "train" in cmd):
                                                        terminal_output.append(f"[SSH]   ✓ This is the training process")
                                                    else:
                                                        terminal_output.append(f"[WARNING] This may not be the training process - it might be a shell script or other process")
                                                        terminal_output.append(f"[ACTION] The training process may have failed to start. Check if training actually started.")
                                                    
                                                    # Check process state - if it's in 'Z' (zombie) or 'T' (stopped), it's not running properly
                                                    if 'Z' in stat:
                                                        terminal_output.append(f"[WARNING] Process is a zombie (exited but not reaped) - training may have failed")
                                                    elif 'T' in stat:
                                                        terminal_output.append(f"[WARNING] Process is stopped - training is not running")
                                        else:
                                            terminal_output.append(f"[SSH] No training process found")
                                        
                                        # Check training log (not onstart log - we're using existing instances)
                                        terminal_output.append(f"[SSH] Checking training logs...")
                                        
                                        # First check if log file exists and its size
                                        check_log_exists_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "ls -lh /workspace/output/training/training.log 2>/dev/null | awk '{print $5, $9}' || echo 'log_not_found'"
                                        ]
                                        log_exists_result = subprocess.run(check_log_exists_cmd, capture_output=True, text=True, timeout=15)
                                        if "log_not_found" not in log_exists_result.stdout and log_exists_result.stdout.strip():
                                            terminal_output.append(f"[SSH] Training log file: {log_exists_result.stdout.strip()}")
                                        
                                        check_training_log_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "tail -100 /workspace/output/training/training.log 2>/dev/null || echo 'no_training_log'"
                                        ]
                                        training_log_result = subprocess.run(check_training_log_cmd, capture_output=True, text=True, timeout=15)
                                        training_logs_content = None
                                        if "no_training_log" not in training_log_result.stdout and training_log_result.stdout.strip():
                                            stdout_filtered = filter_malloc_warnings(training_log_result.stdout)
                                            training_logs_content = stdout_filtered
                                            
                                            # Also try to get the full log to extract stats more accurately
                                            full_log_cmd = [
                                                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                f"root@{ssh_host}",
                                                "cat /workspace/output/training/training.log 2>/dev/null | head -500 || echo 'no_full_log'"
                                            ]
                                            full_log_result = subprocess.run(full_log_cmd, capture_output=True, text=True, timeout=15)
                                            if "no_full_log" not in full_log_result.stdout and full_log_result.stdout.strip():
                                                full_log_content = filter_malloc_warnings(full_log_result.stdout)
                                                # Use full log for stats extraction (more accurate)
                                                training_logs_content = full_log_content
                                            
                                            # Extract dataset statistics from training logs
                                            training_stats = extract_dataset_stats(training_logs_content)
                                            
                                            # If we still don't have final_count, try to check the prepared dataset directly
                                            if training_stats.get('original_count') and not training_stats.get('final_count'):
                                                terminal_output.append(f"[DATASET STATS] Attempting to get final count from prepared dataset...")
                                                check_dataset_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    "python3 -c \"import json; f=open('/workspace/axolotl/prepared_data/last_run_prepared/*/train.jsonl' if __import__('glob').glob('/workspace/axolotl/prepared_data/last_run_prepared/*/train.jsonl') else '/workspace/axolotl/prepared_data/*/train.jsonl', 'r'); count=sum(1 for _ in f); print(f'final_count:{count}')\" 2>/dev/null || "
                                                    "find /workspace/axolotl/prepared_data -name 'train.jsonl' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print \"final_count:\" $1}' || echo 'cannot_count'"
                                                ]
                                                dataset_count_result = subprocess.run(check_dataset_cmd, capture_output=True, text=True, timeout=15)
                                                if "final_count:" in dataset_count_result.stdout:
                                                    try:
                                                        count_line = [l for l in dataset_count_result.stdout.split('\n') if 'final_count:' in l][0]
                                                        final_count = int(count_line.split(':')[1].strip())
                                                        training_stats['final_count'] = final_count
                                                        terminal_output.append(f"[DATASET STATS] Found final count from prepared dataset: {final_count}")
                                                    except:
                                                        pass
                                            if training_stats:
                                                terminal_output.append(f"[DATASET STATS] ========================================")
                                                if training_stats.get("original_count"):
                                                    terminal_output.append(f"[DATASET STATS] Original samples: {training_stats['original_count']}")
                                                if training_stats.get("final_count"):
                                                    terminal_output.append(f"[DATASET STATS] Final training samples: {training_stats['final_count']}")
                                                if training_stats.get("dropped_long"):
                                                    terminal_output.append(f"[DATASET STATS] Dropped (too long): {training_stats['dropped_long']}")
                                                if training_stats.get("dropped_zero_tokens"):
                                                    terminal_output.append(f"[DATASET STATS] Dropped (zero tokens): {training_stats['dropped_zero_tokens']}")
                                                if training_stats.get("total_dropped"):
                                                    dropped_pct = (training_stats['total_dropped'] / training_stats['original_count'] * 100) if training_stats.get('original_count') else 0
                                                    terminal_output.append(f"[DATASET STATS] Total dropped: {training_stats['total_dropped']} ({dropped_pct:.1f}%)")
                                                terminal_output.append(f"[DATASET STATS] ========================================")
                                            
                                            terminal_output.append(f"[TRAINING LOG] Last 20 lines:")
                                            for line in stdout_filtered.strip().split("\n")[-20:]:
                                                if line.strip():
                                                    terminal_output.append(f"[TRAINING] {line[:200]}")
                                        else:
                                            terminal_output.append(f"[TRAINING LOG] No training log found yet - training may not have started")
                                            
                                            # If process is running but no logs, check for errors in other locations
                                            if ssh_training_running:
                                                terminal_output.append(f"[DIAGNOSTICS] Process detected but no logs - checking for errors...")
                                                
                                                # Check if the log file is being created but is empty (process might be starting)
                                                check_log_size_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    "test -f /workspace/output/training/training.log && wc -l /workspace/output/training/training.log || echo 'log_not_exists'"
                                                ]
                                                log_size_result = subprocess.run(check_log_size_cmd, capture_output=True, text=True, timeout=15)
                                                if "log_not_exists" not in log_size_result.stdout and log_size_result.stdout.strip():
                                                    log_info = log_size_result.stdout.strip()
                                                    terminal_output.append(f"[DIAGNOSTICS] Log file status: {log_info}")
                                                    
                                                    # If log exists but has 0 lines, process might have just started
                                                    if "0 " in log_info or " 0 " in log_info:
                                                        terminal_output.append(f"[INFO] Log file exists but is empty - training may be initializing")
                                                
                                                # Check if process might have failed immediately - look for any output in /tmp or other common locations
                                                check_tmp_output_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    "ls -lht /tmp/*.log /tmp/*.out /tmp/*.err 2>/dev/null | head -3 || echo 'no_tmp_output'"
                                                ]
                                                tmp_output_result = subprocess.run(check_tmp_output_cmd, capture_output=True, text=True, timeout=15)
                                                if "no_tmp_output" not in tmp_output_result.stdout and tmp_output_result.stdout.strip():
                                                    tmp_filtered = filter_malloc_warnings(tmp_output_result.stdout)
                                                    terminal_output.append(f"[DIAGNOSTICS] Found output files in /tmp: {tmp_filtered.strip()}")
                                                
                                                # Check if we can see the process's file descriptors to see if it's writing
                                                check_process_fds_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | awk '{print $2}' | head -1 | xargs -I {} sh -c 'if [ -n \"{}\" ]; then ls -l /proc/{}/fd/ 2>/dev/null | grep -E \"(training.log|stdout|stderr)\" || echo \"no_fds\"; else echo \"no_pid\"; fi'"
                                                ]
                                                process_fds_result = subprocess.run(check_process_fds_cmd, capture_output=True, text=True, timeout=15)
                                                if process_fds_result.stdout and "no_fds" not in process_fds_result.stdout and "no_pid" not in process_fds_result.stdout:
                                                    fds_filtered = filter_malloc_warnings(process_fds_result.stdout)
                                                    terminal_output.append(f"[DIAGNOSTICS] Process file descriptors: {fds_filtered.strip()}")
                                        
                                        # Check debug.log if it exists
                                        terminal_output.append(f"[SSH] Checking debug.log for errors...")
                                        debug_log_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "if [ -f /workspace/output/training/debug.log ]; then tail -30 /workspace/output/training/debug.log; else echo 'no_debug_log'; fi"
                                        ]
                                        debug_result = subprocess.run(debug_log_cmd, capture_output=True, text=True, timeout=15)
                                        debug_log_content = None
                                        if "no_debug_log" not in debug_result.stdout and debug_result.stdout.strip():
                                            stdout_filtered = filter_malloc_warnings(debug_result.stdout)
                                            debug_log_content = stdout_filtered
                                            terminal_output.append(f"[DEBUG LOG] Last 30 lines:")
                                            for line in stdout_filtered.strip().split("\n")[-10:]:
                                                if line.strip():
                                                    terminal_output.append(f"[DEBUG] {line}")
                                        
                                        # Get training logs
                                        terminal_output.append(f"[SSH] Retrieving training logs...")
                                        log_command = (
                                            "if [ -f /workspace/output/training/training.log ]; then "
                                            "tail -50 /workspace/output/training/training.log; "
                                            "elif [ -f /workspace/axolotl/training.log ]; then "
                                            "tail -50 /workspace/axolotl/training.log; "
                                            "elif [ -f /workspace/output/training/debug.log ]; then "
                                            "tail -50 /workspace/output/training/debug.log; "
                                            "else echo 'no_logs'; fi"
                                        )
                                        log_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            log_command
                                        ]
                                        log_result = subprocess.run(log_cmd, capture_output=True, text=True, timeout=15)
                                        if "no_logs" not in log_result.stdout and log_result.stdout.strip():
                                            stdout_filtered = filter_malloc_warnings(log_result.stdout)
                                            # Get more lines to find the actual error (not just the wrapper)
                                            log_lines = stdout_filtered.strip().split("\n")
                                            training_logs_content = "\n".join(log_lines)
                                            
                                            # Extract dataset statistics from logs
                                            dataset_stats = extract_dataset_stats(training_logs_content)
                                            if dataset_stats:
                                                terminal_output.append(f"[DATASET STATS] ========================================")
                                                if dataset_stats.get("original_count"):
                                                    terminal_output.append(f"[DATASET STATS] Original samples: {dataset_stats['original_count']}")
                                                if dataset_stats.get("final_count"):
                                                    terminal_output.append(f"[DATASET STATS] Final training samples: {dataset_stats['final_count']}")
                                                if dataset_stats.get("dropped_long"):
                                                    terminal_output.append(f"[DATASET STATS] Dropped (too long): {dataset_stats['dropped_long']}")
                                                if dataset_stats.get("dropped_zero_tokens"):
                                                    terminal_output.append(f"[DATASET STATS] Dropped (zero tokens): {dataset_stats['dropped_zero_tokens']}")
                                                if dataset_stats.get("total_dropped"):
                                                    dropped_pct = (dataset_stats['total_dropped'] / dataset_stats['original_count'] * 100) if dataset_stats.get('original_count') else 0
                                                    terminal_output.append(f"[DATASET STATS] Total dropped: {dataset_stats['total_dropped']} ({dropped_pct:.1f}%)")
                                                terminal_output.append(f"[DATASET STATS] ========================================")
                                            
                                            # Show last 20 lines in terminal
                                            display_lines = log_lines[-20:] if len(log_lines) > 20 else log_lines
                                            terminal_output.append(f"[TRAINING LOGS] Last {len(display_lines)} lines:")
                                            for line in display_lines:
                                                if line.strip():
                                                    terminal_output.append(f"[TRAINING] {line}")
                                
                                    # Check for errors in training logs and debug logs before checking status
                                    training_error = None
                                    # Combine onstart logs, training logs, and debug logs for error detection and stats
                                    all_logs_content = ""
                                    if training_logs_content:
                                        all_logs_content = training_logs_content
                                    if debug_log_content:
                                        if all_logs_content:
                                            all_logs_content = all_logs_content + "\n" + debug_log_content
                                        else:
                                            all_logs_content = debug_log_content
                                    
                                    if all_logs_content:
                                        # Look for common error patterns - prioritize actual errors over wrapper errors
                                        # Prioritize specific errors that we can fix
                                        priority_error_patterns = [
                                            "FileNotFoundError",
                                            "AttributeError",
                                            "ModuleNotFoundError",
                                            "ImportError",
                                        ]
                                        other_error_patterns = [
                                            "PermissionError",
                                            "RuntimeError",
                                            "ValueError",
                                            "KeyError",
                                            "TypeError",
                                            "OSError",
                                            "Traceback (most recent call last)",
                                            "Error:",
                                            "Exception:",
                                        ]
                                        
                                        lines = all_logs_content.split("\n")
                                        
                                        # Find the last actual error (not the accelerate wrapper)
                                        # Look backwards through the log to find the root cause
                                        # First look for priority errors, then other errors
                                        last_error_idx = -1
                                        priority_error_found = False
                                        
                                        # First pass: look for priority errors
                                        for i in range(len(lines) - 1, -1, -1):
                                            line = lines[i]
                                            # Skip accelerate wrapper errors
                                            if "accelerate" in line.lower() and ("CalledProcessError" in line or "subprocess" in line):
                                                continue
                                            # Look for priority errors first
                                            for pattern in priority_error_patterns:
                                                if pattern in line:
                                                    last_error_idx = i
                                                    priority_error_found = True
                                                    break
                                            if priority_error_found:
                                                break
                                        
                                        # Second pass: if no priority error, look for other errors
                                        if not priority_error_found:
                                            for i in range(len(lines) - 1, -1, -1):
                                                line = lines[i]
                                                # Skip accelerate wrapper errors
                                                if "accelerate" in line.lower() and ("CalledProcessError" in line or "subprocess" in line):
                                                    continue
                                                # Look for other errors
                                                for pattern in other_error_patterns:
                                                    if pattern in line:
                                                        last_error_idx = i
                                                        break
                                                if last_error_idx >= 0:
                                                    break
                                        
                                        if last_error_idx >= 0:
                                            # Extract error context - get more lines for better context
                                            start = max(0, last_error_idx - 5)
                                            # Go forward to find the end of the error (usually 10-15 lines)
                                            end = min(len(lines), last_error_idx + 15)
                                            training_error = "\n".join(lines[start:end])
                                    
                                    training_status = training_manager.check_training_status(instance_id)
                                    
                                    # Ensure training_status is a dict
                                    if training_status is None:
                                        training_status = {}
                                    
                                    # Override status if we found errors in logs
                                    if training_error and training_status.get("status") != "completed":
                                        training_status["status"] = "failed"
                                        # Extract a concise error message
                                        error_lines = training_error.split("\n")
                                        for line in reversed(error_lines):
                                            if any(keyword in line for keyword in ["Error", "Exception", "AttributeError", "ModuleNotFoundError", "ImportError"]):
                                                training_status["failure_reason"] = line.strip()
                                                break
                                        if "failure_reason" not in training_status:
                                            training_status["failure_reason"] = "Training error detected in logs (see training logs above)"
                                    
                                    # If SSH check shows training is running but status is unknown, set it to training
                                    if ssh_training_running is not None and ssh_training_running:
                                        if training_status.get("status") == "unknown" or not training_status.get("status"):
                                            training_status["status"] = "training"
                                            terminal_output.append(f"[INFO] Training process detected via SSH - setting status to 'training'")
                                    
                                    active_job["training_status"] = training_status
                                    training_manager._save_job(active_job)
                                    
                                    status_val = training_status.get("status", "unknown")
                                    
                                    # Check for completion indicators in logs BEFORE checking for preprocessing
                                    # This ensures completion takes priority over preprocessing detection
                                    if all_logs_content:
                                        completion_indicators = [
                                            "training completed", "model successfully saved", "saving trained model",
                                            "checkpoint-", "train_loss", "eval_loss", "train_runtime", "epoch"
                                        ]
                                        is_completed = any(indicator in all_logs_content.lower() for indicator in completion_indicators)
                                        
                                        if is_completed and status_val != "completed":
                                            # Override status to completed if we see completion indicators
                                            status_val = "completed"
                                            training_status["status"] = "completed"
                                            active_job["training_status"] = training_status
                                            training_manager._save_job(active_job)
                                    
                                    terminal_output.append(f"[INFO] Training status: {status_val}")
                                    
                                    # Add detailed diagnostics - use SSH check results if available, otherwise fall back to training_status
                                    
                                    if ssh_files_exist is not None and ssh_training_running is not None:
                                        # Use the SSH check results we just performed
                                        training_files_exist = ssh_files_exist
                                        training_running = ssh_training_running
                                    else:
                                        # No SSH checks performed, use training_status results
                                        training_files_exist = training_status.get("training_files_exist", False)
                                        training_running = training_status.get("training_running", False)
                                        
                                        terminal_output.append(f"[DIAGNOSTICS] Training files exist: {training_files_exist}")
                                        terminal_output.append(f"[DIAGNOSTICS] Training process running: {training_running}")
                                        
                                    # Show diagnostics for unknown status
                                    if status_val == "unknown":
                                        if not training_files_exist:
                                            terminal_output.append(f"[WARNING] Training files not found on instance!")
                                            terminal_output.append(f"[WARNING] Files may have been uploaded correctly in Phase 2.")
                                            terminal_output.append(f"[ACTION] Go back to Phase 2 and re-upload files.")
                                        elif not training_running and training_files_exist:
                                            # For existing instances, we should start training directly
                                            terminal_output.append(f"[INFO] Files are present but training is not running.")
                                            terminal_output.append(f"[INFO] For existing instances, training should be started directly.")
                                            terminal_output.append(f"[ACTION] Click 'Start Training' button below to begin training.")
                                        elif training_running and training_files_exist:
                                            # If process is running but status is unknown, set it to training
                                            if status_val == "unknown":
                                                status_val = "training"
                                                training_status["status"] = "training"
                                                active_job["training_status"] = training_status
                                                training_manager._save_job(active_job)
                                                terminal_output.append(f"[INFO] Training process detected - setting status to 'training'")
                                            
                                            # Check for completion indicators first (these take priority)
                                            completion_indicators = [
                                                "training completed", "model successfully saved", "saving trained model",
                                                "checkpoint-", "epoch", "train_loss", "eval_loss", "train_runtime"
                                            ]
                                            is_completed = any(indicator in all_logs_content.lower() if all_logs_content else False for indicator in completion_indicators)
                                            
                                            # Check if preprocessing/training is actually happening (only if not completed)
                                            if not is_completed:
                                                preprocessing_indicators = ["tokenizing", "preprocessing", "dropping", "saving the dataset", "sample packing"]
                                                is_preprocessing = any(indicator in all_logs_content.lower() if all_logs_content else False for indicator in preprocessing_indicators)
                                                
                                                if is_preprocessing:
                                                    terminal_output.append(f"[INFO] ✓ Training is active - preprocessing data (tokenizing, preparing dataset).")
                                                    terminal_output.append(f"[INFO] This is normal and can take several minutes depending on dataset size.")
                                                    terminal_output.append(f"[INFO] Check the logs above to see preprocessing progress.")
                                                    # Mark that we detected preprocessing so we don't show "unclear" message later
                                                    status_val = "training"  # Override status to training when preprocessing is detected
                                                    # Update training_status to reflect this
                                                    training_status["status"] = "training"
                                                    active_job["training_status"] = training_status
                                                    training_manager._save_job(active_job)
                                                else:
                                                    # Training is running and files exist - might be installing or about to start
                                                    terminal_output.append(f"[INFO] Training process is running and files are present.")
                                                    terminal_output.append(f"[INFO] Training may be starting soon (installing dependencies or initializing).")
                                                    terminal_output.append(f"[INFO] Check the training logs above to see current progress.")
                                                    terminal_output.append(f"[INFO] If training doesn't start within a few minutes, check the logs manually via SSH.")
                                        # If completed, the status check above should have already set it to "completed"
                                    
                                    if status_val == "completed":
                                        terminal_output.append(f"[SUCCESS] Training completed!")
                                        
                                        # Check if there are more jobs in the queue
                                        job_queue = active_job.get("job_queue")
                                        current_job_index = active_job.get("current_job_index")
                                        
                                        if job_queue and current_job_index is not None and current_job_index + 1 < len(job_queue):
                                            # More jobs in queue - merge adapter from completed job into next job
                                            next_job_index = current_job_index + 1
                                            next_job = job_queue[next_job_index]
                                            
                                            terminal_output.append(f"[QUEUE] Job {current_job_index + 1} complete. Starting job {next_job_index + 1}/{len(job_queue)}...")
                                            terminal_output.append(f"[MERGE] Merging adapter from job {current_job_index + 1} into job {next_job_index + 1}...")
                                            
                                            # Rename files on instance to bring next job's files to the front
                                            try:
                                                import subprocess
                                                # Get SSH info - prefer saved SSH details from job over API
                                                ssh_host = active_job.get("ssh_host")
                                                ssh_port = active_job.get("ssh_port", 22)
                                                
                                                # If not in job, get from API
                                                if not ssh_host:
                                                    job_status = training_manager.get_job_status(instance_id)
                                                    ssh_host = job_status.get("ssh_host")
                                                    ssh_port = job_status.get("ssh_port", 22)
                                                    # Save to job for future use
                                                    if ssh_host:
                                                        active_job["ssh_host"] = ssh_host
                                                        active_job["ssh_port"] = ssh_port
                                                        training_manager._save_job(active_job)
                                                
                                                if ssh_host:
                                                    # Step 0: Verify training process has actually stopped/completed
                                                    terminal_output.append(f"[QUEUE] ========================================")
                                                    terminal_output.append(f"[QUEUE] Verifying job {current_job_index + 1} has completed...")
                                                    verify_complete_cmd = [
                                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                        f"root@{ssh_host}",
                                                        "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | wc -l"
                                                    ]
                                                    verify_complete_result = subprocess.run(verify_complete_cmd, capture_output=True, text=True, timeout=15)
                                                    remaining_processes = 0
                                                    if verify_complete_result.returncode == 0 and verify_complete_result.stdout.strip().isdigit():
                                                        remaining_processes = int(verify_complete_result.stdout.strip())
                                                    
                                                    if remaining_processes > 0:
                                                        terminal_output.append(f"[WARNING] Found {remaining_processes} training process(es) still running!")
                                                        terminal_output.append(f"[WARNING] Waiting for job {current_job_index + 1} to complete before transitioning...")
                                                        terminal_output.append(f"[INFO] Please wait for training to finish, then check status again.")
                                                        terminal_output.append(f"[INFO] Queue transition will happen automatically when job {current_job_index + 1} completes.")
                                                        # Don't proceed with transition - training is still running
                                                        st.session_state[terminal_output_key] = terminal_output
                                                        st.rerun()
                                                        return
                                                    else:
                                                        terminal_output.append(f"[QUEUE] ✓ Job {current_job_index + 1} process has stopped - safe to transition")
                                                    
                                                    # Step 1: Save completed job's results to a safe location before cleanup
                                                    terminal_output.append(f"[CLEANUP] ========================================")
                                                    terminal_output.append(f"[CLEANUP] Step 1: Saving results from job {current_job_index + 1}...")
                                                    save_results_cmd = [
                                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                        f"root@{ssh_host}",
                                                        f"mkdir -p /workspace/data/job_{current_job_index}_results && "
                                                        f"cp -r /workspace/output/training/* /workspace/data/job_{current_job_index}_results/ 2>/dev/null && "
                                                        f"ls -la /workspace/data/job_{current_job_index}_results/ | head -10 && "
                                                        f"echo 'Results saved' || echo 'No results to save'"
                                                    ]
                                                    save_result = subprocess.run(save_results_cmd, capture_output=True, text=True, timeout=60)
                                                    
                                                    if "Results saved" in save_result.stdout:
                                                        terminal_output.append(f"[CLEANUP] ✓ Job {current_job_index + 1} results saved to /workspace/data/job_{current_job_index}_results/")
                                                        # Show what was saved
                                                        if save_result.stdout:
                                                            saved_files = [line for line in save_result.stdout.split('\n') if line.strip() and 'Results saved' not in line and 'total' not in line]
                                                            if saved_files:
                                                                terminal_output.append(f"[CLEANUP]   Saved files: {len(saved_files)} items (checkpoints, adapters, etc.)")
                                                    else:
                                                        terminal_output.append(f"[CLEANUP] ⚠ No results found to save from job {current_job_index + 1}")
                                                    
                                                    # Step 2: Copy completed job's adapter to a location for next job (for merging)
                                                    terminal_output.append(f"[MERGE] ========================================")
                                                    terminal_output.append(f"[MERGE] Step 2: Extracting adapter from job {current_job_index + 1} for merging into job {next_job_index + 1}...")
                                                    # Check multiple possible adapter locations
                                                    copy_adapter_cmd = [
                                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                        f"root@{ssh_host}",
                                                        f"mkdir -p /workspace/data/previous_adapter_{current_job_index} && "
                                                        f"(test -d /workspace/data/job_{current_job_index}_results/adapter && cp -r /workspace/data/job_{current_job_index}_results/adapter/* /workspace/data/previous_adapter_{current_job_index}/ 2>/dev/null && echo 'adapter_from_subdir' || "
                                                        f"test -f /workspace/data/job_{current_job_index}_results/adapter_config.json && cp -r /workspace/data/job_{current_job_index}_results/* /workspace/data/previous_adapter_{current_job_index}/ 2>/dev/null && echo 'adapter_from_root' || "
                                                        f"test -d /workspace/output/training/adapter && cp -r /workspace/output/training/adapter/* /workspace/data/previous_adapter_{current_job_index}/ 2>/dev/null && echo 'adapter_from_output' || "
                                                        f"echo 'no_adapter_found') && "
                                                        f"ls -la /workspace/data/previous_adapter_{current_job_index}/ 2>/dev/null | head -5 && "
                                                        f"test -f /workspace/data/previous_adapter_{current_job_index}/adapter_config.json && echo 'adapter_valid' || echo 'adapter_invalid'"
                                                    ]
                                                    copy_result = subprocess.run(copy_adapter_cmd, capture_output=True, text=True, timeout=60)
                                                    
                                                    adapter_found = "adapter_valid" in copy_result.stdout or ("adapter_from" in copy_result.stdout and "no_adapter_found" not in copy_result.stdout)
                                                    
                                                    if adapter_found:
                                                        terminal_output.append(f"[MERGE] ✓ Adapter extracted from job {current_job_index + 1}")
                                                        terminal_output.append(f"[MERGE]   Location: /workspace/data/previous_adapter_{current_job_index}/")
                                                        if "adapter_valid" in copy_result.stdout:
                                                            terminal_output.append(f"[MERGE]   Adapter verified: adapter_config.json found")
                                                        terminal_output.append(f"[MERGE]   This adapter will be merged into job {next_job_index + 1} (cumulative training)")
                                                    else:
                                                        terminal_output.append(f"[MERGE] ⚠ No adapter found from job {current_job_index + 1}")
                                                        terminal_output.append(f"[MERGE]   Checked locations:")
                                                        terminal_output.append(f"[MERGE]     - /workspace/data/job_{current_job_index}_results/adapter/")
                                                        terminal_output.append(f"[MERGE]     - /workspace/data/job_{current_job_index}_results/")
                                                        terminal_output.append(f"[MERGE]     - /workspace/output/training/adapter/")
                                                        terminal_output.append(f"[MERGE]   Job {next_job_index + 1} will train from base model (not cumulative)")
                                                    
                                                    # Step 3: Clean output directory to prevent conflicts with next job
                                                    terminal_output.append(f"[CLEANUP] ========================================")
                                                    terminal_output.append(f"[CLEANUP] Step 3: Cleaning output directory to prevent conflicts...")
                                                    terminal_output.append(f"[CLEANUP]   Removing old checkpoints and files from /workspace/output/training/...")
                                                    cleanup_cmd = [
                                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                        f"root@{ssh_host}",
                                                        f"ls -la /workspace/output/training/ 2>/dev/null | head -5 && "
                                                        f"rm -rf /workspace/output/training/* && "
                                                        f"mkdir -p /workspace/output/training && "
                                                        f"ls -la /workspace/output/training/ && "
                                                        f"echo 'Output directory cleaned'"
                                                    ]
                                                    cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30)
                                                    
                                                    if cleanup_result.returncode == 0 and "Output directory cleaned" in cleanup_result.stdout:
                                                        terminal_output.append(f"[CLEANUP] ✓ Output directory cleaned successfully")
                                                        terminal_output.append(f"[CLEANUP]   Old files removed, directory ready for job {next_job_index + 1}")
                                                        terminal_output.append(f"[CLEANUP]   This prevents file conflicts and write errors")
                                                    else:
                                                        terminal_output.append(f"[CLEANUP] ⚠ Cleanup may have failed, but continuing...")
                                                        if cleanup_result.stdout:
                                                            terminal_output.append(f"[CLEANUP]   Output: {cleanup_result.stdout[:200]}")
                                                    
                                                    # Step 4: Update next job's config to use the previous adapter (merge)
                                                    # Only attempt merge if adapter was actually found
                                                    # Initialize adapter_merged flag (will be set to True if merge succeeds)
                                                    adapter_merged = False
                                                    if adapter_found:
                                                        terminal_output.append(f"[MERGE] ========================================")
                                                        terminal_output.append(f"[MERGE] Step 4: Merging adapter into job {next_job_index + 1} config...")
                                                        terminal_output.append(f"[MERGE]   Setting adapter path: /workspace/data/previous_adapter_{current_job_index}")
                                                        # Use Python to update YAML config (Python is available in the PyTorch image)
                                                        update_config_remote_cmd = (
                                                            f"cd /workspace/data && "
                                                            f"python3 << 'PYTHON_EOF'\n"
                                                            f"import yaml\n"
                                                            f"import sys\n"
                                                            f"try:\n"
                                                            f"    with open('axolotl_config_{next_job_index}.yaml', 'r') as f:\n"
                                                            f"        config = yaml.safe_load(f) or {{}}\n"
                                                            f"    old_adapter = config.get('adapter', 'none')\n"
                                                            f"    config['adapter'] = '/workspace/data/previous_adapter_{current_job_index}'\n"
                                                            f"    with open('axolotl_config_{next_job_index}.yaml', 'w') as f:\n"
                                                            f"        yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                            f"    print(f'Config updated: adapter set to /workspace/data/previous_adapter_{current_job_index}')\n"
                                                            f"    print(f'Previous adapter value: {{old_adapter}}')\n"
                                                            f"except Exception as e:\n"
                                                            f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                                            f"    sys.exit(1)\n"
                                                            f"PYTHON_EOF"
                                                        )
                                                        update_config_cmd = [
                                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                            f"root@{ssh_host}",
                                                            update_config_remote_cmd
                                                        ]
                                                        update_result = subprocess.run(update_config_cmd, capture_output=True, text=True, timeout=30)
                                                        
                                                        if update_result.returncode == 0 and "Config updated" in update_result.stdout:
                                                            adapter_merged = True
                                                            terminal_output.append(f"[MERGE] ✓ Config updated successfully")
                                                            terminal_output.append(f"[MERGE]   Job {next_job_index + 1} will continue training from job {current_job_index + 1}'s adapter")
                                                            terminal_output.append(f"[MERGE]   This enables cumulative/incremental training")
                                                            if update_result.stdout:
                                                                for line in update_result.stdout.split('\n'):
                                                                    if line.strip() and 'Config updated' in line:
                                                                        terminal_output.append(f"[MERGE]   {line.strip()}")
                                                        else:
                                                            error_msg = update_result.stderr or update_result.stdout
                                                            terminal_output.append(f"[MERGE] ⚠ Failed to update config: {error_msg[:200]}")
                                                            terminal_output.append(f"[MERGE]   Job {next_job_index + 1} will train from base model (adapter merge skipped)")
                                                    else:
                                                        terminal_output.append(f"[MERGE] ========================================")
                                                        terminal_output.append(f"[MERGE] Step 4: Skipping adapter merge (no adapter found from job {current_job_index + 1})")
                                                        terminal_output.append(f"[MERGE]   Job {next_job_index + 1} will train from base model")
                                                    
                                                    # Step 5: Clean up old active files and rename next job's files to active names
                                                    terminal_output.append(f"[SSH] Cleaning up old active files and activating job {next_job_index + 1} files...")
                                                    rename_cmd = [
                                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                        f"root@{ssh_host}",
                                                        f"cd /workspace/data && "
                                                        f"rm -f axolotl_config.yaml training_data.jsonl && "
                                                        f"mv axolotl_config_{next_job_index}.yaml axolotl_config.yaml && "
                                                        f"mv training_data_{next_job_index}.jsonl training_data.jsonl && "
                                                        f"echo 'Files renamed successfully'"
                                                    ]
                                                    rename_result = subprocess.run(rename_cmd, capture_output=True, text=True, timeout=30)
                                                    
                                                    if rename_result.returncode == 0:
                                                        terminal_output.append(f"[SUCCESS] Files renamed: Job {next_job_index + 1} files are now active")
                                                        
                                                        # Update job index
                                                        active_job["current_job_index"] = next_job_index
                                                        
                                                        # Update package_info to point to next job's package
                                                        all_package_infos = active_job.get("all_package_infos", [])
                                                        if all_package_infos and next_job_index < len(all_package_infos):
                                                            active_job["package_info"] = all_package_infos[next_job_index]
                                                            # Update YAML config in package_info for display
                                                            if next_job.get("yaml_path"):
                                                                from pathlib import Path
                                                                yaml_filename = Path(next_job.get("yaml_path")).name
                                                                active_job["package_info"]["yaml_config"] = yaml_filename
                                                            else:
                                                                active_job["package_info"]["yaml_config"] = None
                                                        
                                                        training_manager._save_job(active_job)
                                                        
                                                        terminal_output.append(f"[QUEUE] ========================================")
                                                        terminal_output.append(f"[QUEUE] Job {next_job_index + 1}/{len(job_queue)} ready: {next_job.get('file_count', 0)} file(s)")
                                                        terminal_output.append(f"[QUEUE] Summary:")
                                                        terminal_output.append(f"[QUEUE]   ✓ Job {current_job_index + 1} results saved")
                                                        terminal_output.append(f"[QUEUE]   ✓ Output directory cleaned (no conflicts)")
                                                        if adapter_merged:
                                                            terminal_output.append(f"[QUEUE]   ✓ Adapter from job {current_job_index + 1} merged into job {next_job_index + 1}")
                                                            terminal_output.append(f"[QUEUE]   ✓ Job {next_job_index + 1} will continue training from job {current_job_index + 1}'s adapter (cumulative training)")
                                                        else:
                                                            terminal_output.append(f"[QUEUE]   ⚠ Adapter merge skipped (no adapter found or config update failed)")
                                                            terminal_output.append(f"[QUEUE]   ⚠ Job {next_job_index + 1} will train from base model (not cumulative)")
                                                        terminal_output.append(f"[QUEUE] ========================================")
                                                        
                                                        # Step 6: Verify no processes are running (should already be stopped, but double-check)
                                                        terminal_output.append(f"[SSH] Verifying no training processes are running before starting job {next_job_index + 1}...")
                                                        verify_no_processes_cmd = [
                                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                            f"root@{ssh_host}",
                                                            "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | wc -l"
                                                        ]
                                                        verify_no_processes_result = subprocess.run(verify_no_processes_cmd, capture_output=True, text=True, timeout=15)
                                                        remaining_before_start = 0
                                                        if verify_no_processes_result.returncode == 0 and verify_no_processes_result.stdout.strip().isdigit():
                                                            remaining_before_start = int(verify_no_processes_result.stdout.strip())
                                                        
                                                        if remaining_before_start > 0:
                                                            # There are still processes - kill them (shouldn't happen if previous job completed, but safety check)
                                                            terminal_output.append(f"[WARNING] Found {remaining_before_start} process(es) still running - stopping them...")
                                                            kill_training_cmd = [
                                                                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                                f"root@{ssh_host}",
                                                                "pkill -9 -f accelerate || true; "
                                                                "pkill -9 -f axolotl || true; "
                                                                "ps aux | grep -E 'python.*train|python.*axolotl|python.*accelerate' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true; "
                                                                "sleep 2; "
                                                                "remaining=$(ps aux | grep -E 'accelerate|axolotl|train' | grep -v grep | wc -l); "
                                                                "echo 'Processes stopped. Remaining: '$remaining"
                                                            ]
                                                            kill_result = subprocess.run(kill_training_cmd, capture_output=True, text=True, timeout=30)
                                                            if kill_result.returncode == 0:
                                                                stdout_filtered = filter_malloc_warnings(kill_result.stdout)
                                                                if stdout_filtered.strip():
                                                                    terminal_output.append(f"[SSH] {stdout_filtered.strip()}")
                                                        else:
                                                            terminal_output.append(f"[SSH] ✓ No training processes running - safe to start job {next_job_index + 1}")
                                                        
                                                        # Note: Output directory was already cleaned in Step 3 above
                                                        # Step 7: Start training for the next job
                                                        terminal_output.append(f"[SSH] Starting training for job {next_job_index + 1}...")
                                                        
                                                        # Get Hugging Face token for gated models
                                                        hf_token = get_hf_token()
                                                        
                                                        # Build environment variable string for token
                                                        if hf_token:
                                                            hf_token_escaped = hf_token.replace("'", "'\"'\"'")
                                                            env_vars_set = f"ENV_VARS=\"HF_TOKEN='{hf_token_escaped}' HUGGING_FACE_HUB_TOKEN='{hf_token_escaped}'\" && "
                                                            env_prefix = "env $ENV_VARS "
                                                        else:
                                                            env_vars_set = "ENV_VARS=\"\" && "
                                                            env_prefix = ""
                                                        
                                                        # Unified training start: find Python with accelerate and axolotl, then use python -m accelerate.commands.launch
                                                        restart_training_remote_cmd = (
                                                            "cd /workspace/axolotl && "
                                                            "TRAIN_PYTHON='' && "
                                                            "if /opt/conda/bin/python -c 'import accelerate; import axolotl' 2>/dev/null; then "
                                                            " TRAIN_PYTHON='/opt/conda/bin/python'; "
                                                            "elif python3 -c 'import accelerate; import axolotl' 2>/dev/null; then "
                                                            " TRAIN_PYTHON='python3'; "
                                                            "elif python -c 'import accelerate; import axolotl' 2>/dev/null; then "
                                                            " TRAIN_PYTHON='python'; "
                                                            "fi && "
                                                            "if [ -z \"$TRAIN_PYTHON\" ]; then "
                                                            " echo 'ERROR: accelerate or axolotl not found'; exit 1; "
                                                            "fi && "
                                                            "if [[ \"$TRAIN_PYTHON\" == /* ]]; then "
                                                            " PYTHON_DIR=$(dirname \"$TRAIN_PYTHON\"); "
                                                            " ACCELERATE_CMD=\"$PYTHON_DIR/accelerate\"; "
                                                            "else "
                                                            " ACCELERATE_CMD=$(command -v accelerate 2>/dev/null || echo \"accelerate\"); "
                                                            "fi && "
                                                            f"{env_vars_set}"
                                                            "if [ -f \"$ACCELERATE_CMD\" ] || command -v accelerate >/dev/null 2>&1; then "
                                                            f" {env_prefix}nohup $ACCELERATE_CMD launch -m axolotl.cli.train /workspace/data/axolotl_config.yaml "
                                                            "> /workspace/output/training/training.log 2>&1 < /dev/null &; "
                                                            "else "
                                                            f" {env_prefix}nohup $TRAIN_PYTHON -m accelerate.commands.launch -m axolotl.cli.train /workspace/data/axolotl_config.yaml "
                                                            "> /workspace/output/training/training.log 2>&1 < /dev/null &; "
                                                            "fi"
                                                        )
                                                        restart_training_cmd = [
                                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                                            f"root@{ssh_host}",
                                                            restart_training_remote_cmd
                                                        ]
                                                        try:
                                                            restart_result = subprocess.run(restart_training_cmd, capture_output=True, text=True, timeout=5)
                                                            if restart_result.returncode == 0:
                                                                terminal_output.append(f"[SUCCESS] Training command sent for job {next_job_index + 1}")
                                                            else:
                                                                terminal_output.append(f"[WARNING] Training command sent (checking process status...)")
                                                        except subprocess.TimeoutExpired:
                                                            # Timeout is expected for background process - command was sent
                                                            terminal_output.append(f"[SUCCESS] Training command sent for job {next_job_index + 1} (timeout expected for background process)")
                                                        
                                                        # Verify training process started
                                                        terminal_output.append(f"[SSH] Verifying training process started...")
                                                        verify_cmd = [
                                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                            f"root@{ssh_host}",
                                                            "sleep 3 && ps aux | grep -E '(accelerate|axolotl)' | grep -v grep | head -2 || echo 'no_process'"
                                                        ]
                                                        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=15)
                                                        if "no_process" not in verify_result.stdout:
                                                            stdout_filtered = filter_malloc_warnings(verify_result.stdout)
                                                            terminal_output.append(f"[SSH] Training process found:")
                                                            for line in stdout_filtered.strip().split("\n")[:2]:
                                                                if line.strip():
                                                                    terminal_output.append(f"[SSH]   {line[:150]}")
                                                            terminal_output.append(f"[SUCCESS] Training started successfully for job {next_job_index + 1}")
                                                        else:
                                                            terminal_output.append(f"[WARNING] Training command sent but process not found yet")
                                                            terminal_output.append(f"[INFO] Training may still be starting. Use 'Check Training Status' to verify.")
                                                        
                                                        # Reset training status for the new job
                                                        active_job["training_status"] = {"status": "training", "restarted": True}
                                                        training_manager._save_job(active_job)
                                                        
                                                        terminal_output.append(f"[INFO] Training in progress for job {next_job_index + 1}...")
                                                        
                                                        # Stay in Phase 3 - training is now running
                                                        st.session_state[terminal_output_key] = terminal_output
                                                        st.rerun()
                                                    else:
                                                        error_output = rename_result.stderr or rename_result.stdout
                                                        error_output = filter_malloc_warnings(error_output)
                                                        terminal_output.append(f"[ERROR] Failed to rename files: {error_output[:300]}")
                                                        terminal_output.append(f"[INFO] You may need to manually rename files on the instance")
                                                else:
                                                    terminal_output.append(f"[ERROR] SSH host not available. Cannot rename files.")
                                            except Exception as e:
                                                terminal_output.append(f"[ERROR] Failed to activate next job: {str(e)}")
                                                terminal_output.append(f"[INFO] All jobs complete. Proceeding to Phase 4...")
                                                # Fall through to Phase 4
                                                st.session_state[phase_key] = 4
                                                st.session_state[terminal_output_key] = terminal_output
                                                st.rerun()
                                            else:
                                                # All jobs complete - move to Phase 4
                                                terminal_output.append(f"[INFO] All jobs in queue complete!")
                                                terminal_output.append(f"[INFO] Phase 3 complete!")
                                                
                                                # Auto-advance to phase 4
                                                st.session_state[phase_key] = 4
                                                # Clear terminal for next phase
                                                st.session_state[terminal_output_key] = []
                                                st.rerun()
                                    elif status_val == "training":
                                        terminal_output.append(f"[INFO] Training is actively running...")
                                    elif status_val == "failed":
                                        terminal_output.append(f"[ERROR] Training failed!")
                                        failure_reason = training_status.get("failure_reason", "Unknown error")
                                        terminal_output.append(f"[ERROR] {failure_reason}")
                                        # If we detected an error in logs, show it prominently
                                        if training_error:
                                            terminal_output.append(f"[ERROR] Error details from training logs:")
                                            for line in training_error.split("\n"):
                                                if line.strip():
                                                    terminal_output.append(f"[ERROR] {line}")
                                        
                                        # Check for common fixable errors and attempt to fix them
                                        if training_error and ssh_host:
                                            # Check if it's an adapter config error (trying to load non-existent adapter)
                                            if "Can't find 'adapter_config.json'" in training_error or ("adapter" in training_error.lower() and "not found" in training_error.lower()):
                                                terminal_output.append(f"[ACTION] Detected adapter loading error - removing invalid adapter path from config...")
                                                try:
                                                    fix_adapter_remote_cmd = (
                                                    f"cd /workspace/data && "
                                                    f"python3 << 'PYTHON_EOF'\n"
                                                    f"import yaml\n"
                                                    f"import sys\n"
                                                    f"try:\n"
                                                    f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                                    f"        config = yaml.safe_load(f) or {{}}\n"
                                                    f"    \n"
                                                    f"    # Remove adapter field if it's set to a string (not a valid path)\n"
                                                    f"    # For new LoRA training, adapter should not be set - lora_* parameters are enough\n"
                                                    f"    if 'adapter' in config:\n"
                                                    f"        adapter_val = config['adapter']\n"
                                                    f"        # If adapter is set to a string (not a path), remove it\n"
                                                    f"        if isinstance(adapter_val, str) and adapter_val != 'lora' and not adapter_val.startswith('/'):\n"
                                                    f"            del config['adapter']\n"
                                                    f"            print('Removed invalid adapter field')\n"
                                                    f"        elif adapter_val == 'lora':\n"
                                                    f"            # 'lora' as string is interpreted as path - remove it\n"
                                                    f"            del config['adapter']\n"
                                                    f"            print('Removed adapter: lora (lora_* parameters are sufficient)')\n"
                                                    f"    \n"
                                                    f"    with open('axolotl_config.yaml', 'w') as f:\n"
                                                    f"        yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                    f"    print('Config fixed successfully')\n"
                                                    f"except Exception as e:\n"
                                                    f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                                    f"    sys.exit(1)\n"
                                                    f"PYTHON_EOF"
                                                )
                                                    fix_adapter_cmd = [
                                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                        f"root@{ssh_host}",
                                                        fix_adapter_remote_cmd
                                                    ]
                                                    fix_adapter_result = subprocess.run(fix_adapter_cmd, capture_output=True, text=True, timeout=30)
                                                    if fix_adapter_result.returncode == 0 and "fixed successfully" in fix_adapter_result.stdout:
                                                        terminal_output.append(f"[SUCCESS] Config file fixed! Removed invalid adapter field.")
                                                        terminal_output.append(f"[INFO] LoRA training will use lora_* parameters (no adapter field needed for new training).")
                                                        terminal_output.append(f"[INFO] You can now click 'Redo Phase' to restart training with the corrected config.")
                                                    elif fix_adapter_result.returncode == 0:
                                                        terminal_output.append(f"[INFO] Config check completed: {fix_adapter_result.stdout.strip()}")
                                                    else:
                                                        error_msg = fix_adapter_result.stderr or fix_adapter_result.stdout
                                                        terminal_output.append(f"[WARNING] Could not fix config: {error_msg[:200]}")
                                                except Exception as e:
                                                    terminal_output.append(f"[WARNING] Error attempting to fix config: {str(e)}")
                                        
                                        # Check if it's a tokenizer type error
                                        elif ("AttributeError" in training_error or "has no attribute" in training_error) and ("Gemma" in training_error or "tokenizer_type" in training_error.lower()):
                                            terminal_output.append(f"[ACTION] Detected tokenizer type error - attempting to fix config...")
                                            try:
                                                fix_tokenizer_remote_cmd = (
                                                    f"cd /workspace/data && "
                                                    f"python3 << 'PYTHON_EOF'\n"
                                                    f"import yaml\n"
                                                    f"import sys\n"
                                                    f"try:\n"
                                                    f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                                    f"        config = yaml.safe_load(f) or {{}}\n"
                                                    f"    \n"
                                                    f"    # Fix tokenizer type if it's incorrect\n"
                                                    f"    tokenizer_type = config.get('tokenizer_type', '')\n"
                                                    f"    base_model = config.get('base_model', '').lower()\n"
                                                    f"    fixed = False\n"
                                                    f"    \n"
                                                    f"    if 'gemma' in base_model:\n"
                                                    f"        if tokenizer_type == 'Gemma':\n"
                                                    f"            config['tokenizer_type'] = 'GemmaTokenizer'\n"
                                                    f"            fixed = True\n"
                                                    f"        elif tokenizer_type != 'GemmaTokenizer':\n"
                                                    f"            config['tokenizer_type'] = 'GemmaTokenizer'\n"
                                                    f"            fixed = True\n"
                                                    f"        # Gemma 3 uses different architecture - remove model_type to let auto-detect\n"
                                                    f"        if 'gemma-3' in base_model or 'gemma3' in base_model:\n"
                                                    f"            if 'model_type' in config:\n"
                                                    f"                del config['model_type']\n"
                                                    f"                fixed = True\n"
                                                    f"        elif 'GemmaForCausalLM' not in config.get('model_type', ''):\n"
                                                    f"            config['model_type'] = 'GemmaForCausalLM'\n"
                                                    f"            fixed = True\n"
                                                    f"    elif 'mistral' in base_model:\n"
                                                    f"        if tokenizer_type == 'Mistral':\n"
                                                    f"            config['tokenizer_type'] = 'MistralTokenizer'\n"
                                                    f"            fixed = True\n"
                                                    f"        if 'MistralForCausalLM' not in config.get('model_type', ''):\n"
                                                    f"            config['model_type'] = 'MistralForCausalLM'\n"
                                                    f"            fixed = True\n"
                                                    f"    elif 'phi' in base_model:\n"
                                                    f"        if tokenizer_type == 'Phi':\n"
                                                    f"            config['tokenizer_type'] = 'PhiTokenizer'\n"
                                                    f"            fixed = True\n"
                                                    f"        if 'PhiForCausalLM' not in config.get('model_type', ''):\n"
                                                    f"            config['model_type'] = 'PhiForCausalLM'\n"
                                                    f"            fixed = True\n"
                                                    f"    elif 'qwen' in base_model:\n"
                                                    f"        if tokenizer_type == 'Qwen':\n"
                                                    f"            config['tokenizer_type'] = 'Qwen2Tokenizer'\n"
                                                    f"            fixed = True\n"
                                                    f"        if 'Qwen2ForCausalLM' not in config.get('model_type', ''):\n"
                                                    f"            config['model_type'] = 'Qwen2ForCausalLM'\n"
                                                    f"            fixed = True\n"
                                                    f"    \n"
                                                    f"    if fixed:\n"
                                                    f"        with open('axolotl_config.yaml', 'w') as f:\n"
                                                    f"            yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                    f"        print('Config fixed successfully')\n"
                                                    f"    else:\n"
                                                    f"        print('Config already correct')\n"
                                                    f"except Exception as e:\n"
                                                    f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                                    f"    sys.exit(1)\n"
                                                    f"PYTHON_EOF"
                                                )
                                                fix_tokenizer_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    fix_tokenizer_remote_cmd
                                                ]
                                                fix_result = subprocess.run(fix_tokenizer_cmd, capture_output=True, text=True, timeout=30)
                                                if fix_result.returncode == 0 and "fixed successfully" in fix_result.stdout:
                                                    terminal_output.append(f"[SUCCESS] Config file fixed! Tokenizer type corrected.")
                                                    terminal_output.append(f"[INFO] You can now click 'Redo Phase' to restart training with the corrected config.")
                                                elif fix_result.returncode == 0:
                                                    terminal_output.append(f"[INFO] Config check completed: {fix_result.stdout.strip()}")
                                                else:
                                                    error_msg = fix_result.stderr or fix_result.stdout
                                                    terminal_output.append(f"[WARNING] Could not fix config: {error_msg[:200]}")
                                            except Exception as e:
                                                terminal_output.append(f"[WARNING] Error attempting to fix config: {str(e)}")
                                        
                                        # Check if it's a FileNotFoundError for dataset preparation path
                                        elif "FileNotFoundError" in training_error and ("datasets_prep.lock" in training_error or "dataset_preparation_path" in training_error.lower() or "last_run_prepared" in training_error):
                                            terminal_output.append(f"[ACTION] Detected dataset preparation path error - attempting to fix...")
                                            try:
                                                fix_dataset_path_remote_cmd = (
                                                    f"cd /workspace/data && "
                                                    f"python3 << 'PYTHON_EOF'\n"
                                                    f"import yaml\n"
                                                    f"import os\n"
                                                    f"import sys\n"
                                                    f"try:\n"
                                                    f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                                    f"        config = yaml.safe_load(f) or {{}}\n"
                                                    f"    \n"
                                                    f"    # Fix dataset_preparation_path to use absolute path\n"
                                                    f"    dataset_prep_path = config.get('dataset_preparation_path', './prepared_data')\n"
                                                    f"    if not os.path.isabs(dataset_prep_path):\n"
                                                    f"        # Convert to absolute path in /workspace/axolotl\n"
                                                    f"        config['dataset_preparation_path'] = '/workspace/axolotl/prepared_data'\n"
                                                    f"        print('Fixed dataset_preparation_path to absolute path')\n"
                                                    f"    \n"
                                                    f"    # Ensure the directory exists\n"
                                                    f"    os.makedirs(config['dataset_preparation_path'], exist_ok=True)\n"
                                                    f"    print('Created dataset preparation directory')\n"
                                                    f"    \n"
                                                    f"    with open('axolotl_config.yaml', 'w') as f:\n"
                                                    f"        yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                    f"    print('Config fixed successfully')\n"
                                                    f"except Exception as e:\n"
                                                    f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                                    f"    sys.exit(1)\n"
                                                    f"PYTHON_EOF"
                                                )
                                                fix_dataset_path_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    fix_dataset_path_remote_cmd
                                                ]
                                                fix_result = subprocess.run(fix_dataset_path_cmd, capture_output=True, text=True, timeout=30)
                                                if fix_result.returncode == 0 and "fixed successfully" in fix_result.stdout:
                                                    terminal_output.append(f"[SUCCESS] Config file fixed! Dataset preparation path corrected.")
                                                    terminal_output.append(f"[INFO] You can now click 'Redo Phase' to restart training with the corrected config.")
                                                elif fix_result.returncode == 0:
                                                    terminal_output.append(f"[INFO] Config check completed: {fix_result.stdout.strip()}")
                                                else:
                                                    error_msg = fix_result.stderr or fix_result.stdout
                                                    terminal_output.append(f"[WARNING] Could not fix config: {error_msg[:200]}")
                                            except Exception as e:
                                                terminal_output.append(f"[WARNING] Error attempting to fix config: {str(e)}")
                                        
                                        # Check if it's a CUDA out of memory error
                                        elif "OutOfMemoryError" in training_error or "out of memory" in training_error.lower() or "CUDA out of memory" in training_error:
                                            terminal_output.append(f"[ACTION] Detected CUDA out of memory error - attempting to fix config...")
                                            terminal_output.append(f"[INFO] Enabling 4-bit quantization and reducing batch size to save memory...")
                                            try:
                                                fix_oom_remote_cmd = (
                                                    f"cd /workspace/data && "
                                                    f"python3 << 'PYTHON_EOF'\n"
                                                    f"import yaml\n"
                                                    f"import sys\n"
                                                    f"try:\n"
                                                    f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                                    f"        config = yaml.safe_load(f) or {{}}\n"
                                                    f"    \n"
                                                    f"    fixed = False\n"
                                                    f"    \n"
                                                    f"    # Enable 4-bit quantization\n"
                                                    f"    if not config.get('load_in_4bit', False):\n"
                                                    f"        config['load_in_4bit'] = True\n"
                                                    f"        config['load_in_8bit'] = False\n"
                                                    f"        fixed = True\n"
                                                    f"        print('Enabled 4-bit quantization')\n"
                                                    f"    \n"
                                                    f"    # Reduce batch size if it's too large\n"
                                                    f"    current_batch = config.get('micro_batch_size', 4)\n"
                                                    f"    if current_batch > 2:\n"
                                                    f"        config['micro_batch_size'] = 2\n"
                                                    f"        # Increase gradient accumulation to maintain effective batch size\n"
                                                    f"        current_grad_accum = config.get('gradient_accumulation_steps', 4)\n"
                                                    f"        config['gradient_accumulation_steps'] = current_grad_accum * 2\n"
                                                    f"        fixed = True\n"
                                                    f"        print(f'Reduced batch size from {{current_batch}} to 2, increased gradient accumulation to {{config[\"gradient_accumulation_steps\"]}}')\n"
                                                    f"    \n"
                                                    f"    # Ensure gradient checkpointing is enabled\n"
                                                    f"    if not config.get('gradient_checkpointing', False):\n"
                                                    f"        config['gradient_checkpointing'] = True\n"
                                                    f"        fixed = True\n"
                                                    f"        print('Enabled gradient checkpointing')\n"
                                                    f"    \n"
                                                    f"    if fixed:\n"
                                                    f"        with open('axolotl_config.yaml', 'w') as f:\n"
                                                    f"            yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                    f"        print('Config fixed successfully for OOM')\n"
                                                    f"    else:\n"
                                                    f"        print('Config already optimized for memory')\n"
                                                    f"except Exception as e:\n"
                                                    f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                                    f"    sys.exit(1)\n"
                                                    f"PYTHON_EOF"
                                                )
                                                fix_oom_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    fix_oom_remote_cmd
                                                ]
                                                fix_oom_result = subprocess.run(fix_oom_cmd, capture_output=True, text=True, timeout=30)
                                                if fix_oom_result.returncode == 0 and "fixed successfully" in fix_oom_result.stdout:
                                                    terminal_output.append(f"[SUCCESS] Config file fixed for memory optimization!")
                                                    terminal_output.append(f"[INFO] Changes: Enabled 4-bit quantization, reduced batch size, enabled gradient checkpointing")
                                                    terminal_output.append(f"[INFO] You can now click 'Redo Phase' to restart training with the optimized config.")
                                                elif fix_oom_result.returncode == 0:
                                                    terminal_output.append(f"[INFO] Config check completed: {fix_oom_result.stdout.strip()}")
                                                else:
                                                    error_msg = fix_oom_result.stderr or fix_oom_result.stdout
                                                    terminal_output.append(f"[WARNING] Could not fix config: {error_msg[:200]}")
                                            except Exception as e:
                                                terminal_output.append(f"[WARNING] Error attempting to fix config: {str(e)}")
                                        else:
                                            # Final else for the training_error and ssh_host check (no fixable errors or no SSH)
                                            if training_error:
                                                terminal_output.append(f"[ERROR] Training error detected in logs!")
                                                terminal_output.append(f"[ERROR] Error details:")
                                                for line in training_error.split("\n"):
                                                    if line.strip():
                                                        terminal_output.append(f"[ERROR] {line}")
                                                terminal_output.append(f"[ACTION] Training appears to have failed. Check the error above and fix the configuration.")
                                    elif status_val == "unknown":
                                        # Only show "unclear" if status is actually unknown and we haven't already shown a clear message
                                        # (preprocessing detection above would have set status to "training")
                                        terminal_output.append(f"[INFO] Training status unclear.")
                                        terminal_output.append(f"[INFO] Press 'Check Training Status' button again to refresh.")
                                    
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.rerun()
                                except Exception as e:
                                    error_msg = str(e)
                                    terminal_output.append(f"[ERROR] {error_msg}")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.error(f"Error: {error_msg}")
                    
                    with col2:
                        if st.button("🗑️ Clear Terminal", key="clear_terminal_phase3", help="Clear terminal output"):
                            st.session_state[terminal_output_key] = []
                            st.rerun()
                    
                    with col3:
                        if st.button("🔄 Redo Phase", key="redo_phase_3", type="secondary"):
                            try:
                                instance_id = active_job.get("instance_id")
                                
                                # Clear terminal output
                                terminal_output = []
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Redoing Phase 3 - killing processes and cleaning up...")
                                
                                # Get SSH info - prefer saved SSH details from job over API
                                ssh_host = active_job.get("ssh_host")
                                ssh_port = active_job.get("ssh_port", 22)
                                
                                # If not in job, get from API
                                if not ssh_host:
                                    job_status = training_manager.get_job_status(instance_id)
                                    ssh_host = job_status.get("ssh_host")
                                    ssh_port = job_status.get("ssh_port", 22)
                                    # Save to job for future use
                                    if ssh_host:
                                        active_job["ssh_host"] = ssh_host
                                        active_job["ssh_port"] = ssh_port
                                        training_manager._save_job(active_job)
                                
                                if ssh_host:
                                    terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                    import subprocess
                                    
                                    # Step 1: Kill all training-related processes
                                    terminal_output.append(f"[SSH] Killing training processes...")
                                    # Build kill command as a single string for SSH
                                    kill_command = (
                                        "pkill -9 -f accelerate || true; "
                                        "pkill -9 -f axolotl || true; "
                                        "ps aux | grep -E 'python.*train|python.*axolotl|python.*accelerate' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true; "
                                        "ps aux | grep '/workspace/axolotl' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true; "
                                        "sleep 2; "
                                        "remaining=$(ps aux | grep -E 'accelerate|axolotl|train' | grep -v grep | wc -l); "
                                        "echo 'Processes killed. Remaining training processes: '$remaining"
                                    )
                                    kill_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        kill_command
                                    ]
                                    kill_result = subprocess.run(kill_cmd, capture_output=True, text=True, timeout=30)
                                    stdout_filtered = filter_malloc_warnings(kill_result.stdout)
                                    stderr_filtered = filter_malloc_warnings(kill_result.stderr)
                                    if kill_result.returncode == 0:
                                        terminal_output.append(f"[SSH] Training processes killed successfully")
                                        if stdout_filtered.strip():
                                            terminal_output.append(f"[SSH] {stdout_filtered}")
                                    else:
                                        terminal_output.append(f"[WARNING] Some processes may still be running")
                                        if stderr_filtered.strip():
                                            terminal_output.append(f"[SSH] {stderr_filtered[:200]}")
                                    
                                    # Step 2: Delete artifacts in output directory
                                    terminal_output.append(f"[SSH] Deleting artifacts in output directory...")
                                    cleanup_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "rm -rf /workspace/output/training/* && echo 'Artifacts deleted'"
                                    ]
                                    cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30)
                                    if cleanup_result.returncode == 0:
                                        terminal_output.append(f"[SSH] Output directory cleaned successfully")
                                        stdout_filtered = filter_malloc_warnings(cleanup_result.stdout)
                                        if stdout_filtered.strip():
                                            terminal_output.append(f"[SSH] {stdout_filtered}")
                                    else:
                                        stderr_filtered = filter_malloc_warnings(cleanup_result.stderr)
                                        terminal_output.append(f"[WARNING] Cleanup may have failed: {stderr_filtered[:200]}")
                                    
                                    # Step 2.5: Fix tokenizer type in config if it's incorrect (e.g., "Gemma" instead of "GemmaTokenizer")
                                    terminal_output.append(f"[SSH] Checking and fixing tokenizer type in config...")
                                    fix_tokenizer_remote_cmd_redo = (
                                        f"cd /workspace/data && "
                                        f"python3 << 'PYTHON_EOF'\n"
                                        f"import yaml\n"
                                        f"import sys\n"
                                        f"try:\n"
                                        f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                        f"        config = yaml.safe_load(f) or {{}}\n"
                                        f"    \n"
                                        f"    # Fix tokenizer type if it's incorrect\n"
                                        f"    tokenizer_type = config.get('tokenizer_type', '')\n"
                                        f"    base_model = config.get('base_model', '').lower()\n"
                                        f"    \n"
                                        f"    if 'gemma' in base_model and tokenizer_type == 'Gemma':\n"
                                        f"        config['tokenizer_type'] = 'GemmaTokenizer'\n"
                                        f"        print('Fixed: Gemma -> GemmaTokenizer')\n"
                                        f"    elif 'gemma' in base_model and tokenizer_type != 'GemmaTokenizer':\n"
                                        f"        config['tokenizer_type'] = 'GemmaTokenizer'\n"
                                        f"        print(f'Fixed: {{tokenizer_type}} -> GemmaTokenizer')\n"
                                        f"    elif 'mistral' in base_model and tokenizer_type == 'Mistral':\n"
                                        f"        config['tokenizer_type'] = 'MistralTokenizer'\n"
                                        f"        print('Fixed: Mistral -> MistralTokenizer')\n"
                                        f"    elif 'phi' in base_model and tokenizer_type == 'Phi':\n"
                                        f"        config['tokenizer_type'] = 'PhiTokenizer'\n"
                                        f"        print('Fixed: Phi -> PhiTokenizer')\n"
                                        f"    elif 'qwen' in base_model and tokenizer_type == 'Qwen':\n"
                                        f"        config['tokenizer_type'] = 'Qwen2Tokenizer'\n"
                                        f"        print('Fixed: Qwen -> Qwen2Tokenizer')\n"
                                        f"    \n"
                                        f"    # Also fix model_type if needed\n"
                                        f"    model_type = config.get('model_type', '')\n"
                                        f"    if 'gemma' in base_model:\n"
                                        f"        # Gemma 3 uses different architecture - remove model_type to let auto-detect\n"
                                        f"        if 'gemma-3' in base_model or 'gemma3' in base_model:\n"
                                        f"            if 'model_type' in config:\n"
                                        f"                del config['model_type']\n"
                                        f"                print('Removed model_type for Gemma 3 (auto-detect)')\n"
                                        f"        elif 'GemmaForCausalLM' not in model_type:\n"
                                        f"            config['model_type'] = 'GemmaForCausalLM'\n"
                                        f"            print('Fixed model_type: -> GemmaForCausalLM')\n"
                                        f"    \n"
                                        f"    with open('axolotl_config.yaml', 'w') as f:\n"
                                        f"        yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                        f"    print('Config file updated successfully')\n"
                                        f"except Exception as e:\n"
                                        f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                        f"    sys.exit(1)\n"
                                        f"PYTHON_EOF"
                                    )
                                    fix_tokenizer_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        fix_tokenizer_remote_cmd_redo
                                    ]
                                    fix_result = subprocess.run(fix_tokenizer_cmd, capture_output=True, text=True, timeout=30)
                                    if fix_result.returncode == 0:
                                        stdout_filtered = filter_malloc_warnings(fix_result.stdout)
                                        if "Fixed" in fix_result.stdout or "updated successfully" in fix_result.stdout:
                                            terminal_output.append(f"[SUCCESS] Tokenizer type fixed: {stdout_filtered}")
                                        else:
                                            terminal_output.append(f"[INFO] Tokenizer type already correct")
                                    else:
                                        error_msg = fix_result.stderr or fix_result.stdout
                                        terminal_output.append(f"[WARNING] Could not fix tokenizer type: {error_msg[:200]}")
                                    
                                    # Step 2.6: Regenerate and re-upload config file with correct model name
                                    terminal_output.append(f"[INFO] Regenerating config file with correct model mapping...")
                                    package_info = active_job.get("package_info")
                                    if package_info:
                                        try:
                                            # Get model metadata to get base_model
                                            from utils.model_manager import ModelManager
                                            model_manager = ModelManager()
                                            model_name = active_job.get("model_name")
                                            metadata = model_manager.get_model_metadata(model_name)
                                            base_model = metadata.get("base_model", "llama2") if metadata else "llama2"
                                            
                                            terminal_output.append(f"[INFO] Model name: {model_name}, Base model: {base_model}")
                                            
                                            # Get HF model name using current mapping (which is now fixed)
                                            from utils.axolotl_prep import AxolotlDataPrep
                                            axolotl_prep = AxolotlDataPrep(model_name)
                                            hf_model = axolotl_prep.get_hf_model_name(base_model)
                                            
                                            terminal_output.append(f"[INFO] Mapped to HF model: {hf_model}")
                                            
                                            if hf_model:
                                                # Regenerate config with correct model name
                                                from pathlib import Path
                                                import tempfile
                                                import yaml
                                                
                                                # Use a reliable temp directory
                                                temp_dir = Path(tempfile.mkdtemp(prefix="anvil_redo_"))
                                                config_path = temp_dir / "axolotl_config.yaml"
                                                
                                                old_config = package_info.get("config", {})
                                                
                                                terminal_output.append(f"[INFO] Creating config at: {config_path}")
                                                
                                                # Create new config with updated model name
                                                new_config = axolotl_prep.create_axolotl_config(
                                                    base_model=hf_model,
                                                    dataset_path="/workspace/data/training_data.jsonl",
                                                    output_dir="/workspace/output/training",
                                                    output_path=config_path,
                                                    num_epochs=old_config.get("num_epochs", 10),
                                                    learning_rate=old_config.get("learning_rate", 2e-4),
                                                    lora_r=old_config.get("lora_r", 8),
                                                    lora_alpha=old_config.get("lora_alpha", 16),
                                                    lora_dropout=old_config.get("lora_dropout", 0.05),
                                                    batch_size=old_config.get("micro_batch_size", 4),
                                                    gradient_accumulation_steps=old_config.get("gradient_accumulation_steps", 4)
                                                )
                                                
                                                # Handle quantization settings carefully
                                                # Axolotl requires an adapter field when quantization is enabled,
                                                # but setting adapter: "lora" causes it to look for adapter files (path issue)
                                                # Solution: Only enable quantization if we can properly set adapter,
                                                # otherwise disable quantization (LoRA will still work without quantization)
                                                
                                                has_lora = new_config.get("lora_r") or old_config.get("lora_r")
                                                old_quantization_8bit = old_config.get("load_in_8bit", False)
                                                old_quantization_4bit = old_config.get("load_in_4bit", False)
                                                
                                                # If YAML had quantization enabled, but we can't set adapter properly,
                                                # disable quantization to avoid validation errors
                                                # LoRA training works fine without quantization (just uses more memory)
                                                if (old_quantization_8bit or old_quantization_4bit) and has_lora:
                                                    # Disable quantization - LoRA will still work
                                                    new_config["load_in_8bit"] = False
                                                    new_config["load_in_4bit"] = False
                                                    terminal_output.append(f"[INFO] Disabled quantization (was enabled in YAML) to avoid adapter field requirement")
                                                    terminal_output.append(f"[INFO] LoRA training will proceed without quantization (may use more memory)")
                                                elif old_quantization_8bit or old_quantization_4bit:
                                                    # Preserve quantization settings if no LoRA (unlikely but handle it)
                                                    if "load_in_8bit" in old_config:
                                                        new_config["load_in_8bit"] = old_config["load_in_8bit"]
                                                    if "load_in_4bit" in old_config:
                                                        new_config["load_in_4bit"] = old_config["load_in_4bit"]
                                                
                                                # Write updated config back to file
                                                with open(config_path, 'w') as f:
                                                    yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
                                                
                                                # Verify config file was created
                                                if not config_path.exists():
                                                    terminal_output.append(f"[ERROR] Config file was not created at {config_path}")
                                                else:
                                                    # Verify the model name in the config
                                                    with open(config_path, 'r') as f:
                                                        config_content = yaml.safe_load(f)
                                                        config_model = config_content.get("base_model", "NOT_FOUND")
                                                        terminal_output.append(f"[INFO] Config file created. Model in config: {config_model}")
                                                    
                                                    # Update package_info with new config path
                                                    package_info["config_path"] = str(config_path)
                                                    package_info["hf_model"] = hf_model
                                                    active_job["package_info"] = package_info
                                                    training_manager._save_job(active_job)
                                                    
                                                    terminal_output.append(f"[INFO] Config regenerated with model: {hf_model}")
                                                    
                                                    # Re-upload config file with retry logic (Vast.ai sometimes has auth issues)
                                                    terminal_output.append(f"[SCP] Re-uploading config file with corrected model name...")
                                                    scp_config_cmd = [
                                                        "scp", "-P", str(ssh_port), 
                                                        "-o", "StrictHostKeyChecking=no", 
                                                        "-o", "ConnectTimeout=30",
                                                        "-o", "UserKnownHostsFile=/dev/null",
                                                        "-o", "BatchMode=yes",
                                                        str(config_path),
                                                        f"root@{ssh_host}:/workspace/data/axolotl_config.yaml"
                                                    ]
                                                    
                                                    # Try up to 3 times with delays (Vast.ai auth can be flaky)
                                                    scp_success = False
                                                    for attempt in range(1, 4):
                                                        if attempt > 1:
                                                            terminal_output.append(f"[SCP] Retry attempt {attempt}/3...")
                                                            import time
                                                            time.sleep(2)  # Wait 2 seconds between retries
                                                        
                                                        scp_config_result = subprocess.run(scp_config_cmd, capture_output=True, text=True, timeout=300)
                                                        stdout_filtered = filter_malloc_warnings(scp_config_result.stdout)
                                                        stderr_filtered = filter_malloc_warnings(scp_config_result.stderr)
                                                        
                                                        if scp_config_result.returncode == 0:
                                                            scp_success = True
                                                            terminal_output.append(f"[SCP] Config file re-uploaded successfully")
                                                            if stdout_filtered.strip():
                                                                terminal_output.append(f"[SCP] {stdout_filtered}")
                                                            break
                                                        else:
                                                            if attempt < 3:
                                                                terminal_output.append(f"[WARNING] Upload attempt {attempt} failed, will retry...")
                                                            error_output = stderr_filtered or stdout_filtered
                                                            if "Permission denied" in error_output or "publickey" in error_output:
                                                                terminal_output.append(f"[WARNING] Authentication issue detected, retrying...")
                                                        
                                                    if scp_success:
                                                        # Verify the uploaded file on remote
                                                        terminal_output.append(f"[SSH] Verifying uploaded config file...")
                                                        verify_cmd = [
                                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                            f"root@{ssh_host}",
                                                            "grep -E 'base_model|base_model_config' /workspace/data/axolotl_config.yaml | head -2"
                                                        ]
                                                        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=15)
                                                        if verify_result.returncode == 0:
                                                            verify_output = filter_malloc_warnings(verify_result.stdout)
                                                            terminal_output.append(f"[SSH] Remote config contains: {verify_output.strip()}")
                                                            
                                                            # Fix config file (dataset_preparation_path, tokenizer_type, etc.) before restarting
                                                            terminal_output.append(f"[SSH] Verifying and fixing config file settings...")
                                                            fix_config_remote_cmd_redo = (
                                                                f"cd /workspace/data && "
                                                                f"python3 << 'PYTHON_EOF'\n"
                                                                f"import yaml\n"
                                                                f"import os\n"
                                                                f"import sys\n"
                                                                f"try:\n"
                                                                f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                                                f"        config = yaml.safe_load(f) or {{}}\n"
                                                                f"    \n"
                                                                f"    fixed = False\n"
                                                                f"    \n"
                                                                f"    # Always set dataset_preparation_path to absolute path (force fix)\n"
                                                                f"    dataset_prep_path = config.get('dataset_preparation_path', './prepared_data')\n"
                                                                f"    target_path = '/workspace/axolotl/prepared_data'\n"
                                                                f"    if dataset_prep_path != target_path:\n"
                                                                f"        config['dataset_preparation_path'] = target_path\n"
                                                                f"        fixed = True\n"
                                                                f"        print(f'Fixed dataset_preparation_path: {{dataset_prep_path}} -> {{target_path}}')\n"
                                                                f"    else:\n"
                                                                f"        print(f'Dataset preparation path already correct: {{target_path}}')\n"
                                                                f"    \n"
                                                                f"    # Ensure the directory exists and create subdirectories\n"
                                                                f"    os.makedirs(target_path, exist_ok=True)\n"
                                                                f"    # Also create the last_run_prepared subdirectory that Axolotl uses\n"
                                                                f"    last_run_dir = os.path.join(target_path, 'last_run_prepared')\n"
                                                                f"    os.makedirs(last_run_dir, exist_ok=True)\n"
                                                                f"    # Also create it in the working directory as a fallback (Axolotl runs from /workspace/axolotl)\n"
                                                                f"    working_dir_fallback = '/workspace/axolotl/last_run_prepared'\n"
                                                                f"    os.makedirs(working_dir_fallback, exist_ok=True)\n"
                                                                f"    print(f'Ensured directory exists: {{target_path}}')\n"
                                                                f"    print(f'Created last_run_prepared subdirectory: {{last_run_dir}}')\n"
                                                                f"    print(f'Created fallback last_run_prepared in working dir: {{working_dir_fallback}}')\n"
                                                                f"    \n"
                                                                f"    # Fix tokenizer_type if needed\n"
                                                                f"    tokenizer_type = config.get('tokenizer_type', '')\n"
                                                                f"    base_model = config.get('base_model', '').lower()\n"
                                                                f"    \n"
                                                                f"    if 'gemma' in base_model and tokenizer_type != 'GemmaTokenizer':\n"
                                                                f"        config['tokenizer_type'] = 'GemmaTokenizer'\n"
                                                                f"        fixed = True\n"
                                                                f"        print('Fixed tokenizer_type to GemmaTokenizer')\n"
                                                                f"    elif 'mistral' in base_model and tokenizer_type not in ['MistralTokenizer', 'AutoTokenizer']:\n"
                                                                f"        config['tokenizer_type'] = 'MistralTokenizer'\n"
                                                                f"        fixed = True\n"
                                                                f"        print('Fixed tokenizer_type to MistralTokenizer')\n"
                                                                f"    elif 'phi' in base_model and tokenizer_type != 'PhiTokenizer':\n"
                                                                f"        config['tokenizer_type'] = 'PhiTokenizer'\n"
                                                                f"        fixed = True\n"
                                                                f"        print('Fixed tokenizer_type to PhiTokenizer')\n"
                                                                f"    elif 'qwen' in base_model and tokenizer_type != 'Qwen2Tokenizer':\n"
                                                                f"        config['tokenizer_type'] = 'Qwen2Tokenizer'\n"
                                                                f"        fixed = True\n"
                                                                f"        print('Fixed tokenizer_type to Qwen2Tokenizer')\n"
                                                                f"    \n"
                                                                f"    # Fix model_type if needed\n"
                                                                f"    model_type = config.get('model_type', '')\n"
                                                                f"    if 'gemma' in base_model:\n"
                                                                f"        # Gemma 3 uses different architecture - remove model_type to let auto-detect\n"
                                                                f"        if 'gemma-3' in base_model or 'gemma3' in base_model:\n"
                                                                f"            if 'model_type' in config:\n"
                                                                f"                del config['model_type']\n"
                                                                f"                fixed = True\n"
                                                                f"                print('Removed model_type for Gemma 3 (auto-detect)')\n"
                                                                f"        elif 'GemmaForCausalLM' not in model_type:\n"
                                                                f"            config['model_type'] = 'GemmaForCausalLM'\n"
                                                                f"            fixed = True\n"
                                                                f"            print('Fixed model_type to GemmaForCausalLM')\n"
                                                                f"    elif 'mistral' in base_model and 'MistralForCausalLM' not in model_type:\n"
                                                                f"        config['model_type'] = 'MistralForCausalLM'\n"
                                                                f"        fixed = True\n"
                                                                f"        print('Fixed model_type to MistralForCausalLM')\n"
                                                                f"    \n"
                                                                f"    if fixed:\n"
                                                                f"        with open('axolotl_config.yaml', 'w') as f:\n"
                                                                f"            yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                                f"        print('Config file updated successfully')\n"
                                                                f"    else:\n"
                                                                f"        print('Config file already correct')\n"
                                                                f"except Exception as e:\n"
                                                                f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                                                f"    sys.exit(1)\n"
                                                                f"PYTHON_EOF"
                                                            )
                                                            fix_config_cmd_redo = [
                                                                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                                f"root@{ssh_host}",
                                                                fix_config_remote_cmd_redo
                                                            ]
                                                            fix_config_result_redo = subprocess.run(fix_config_cmd_redo, capture_output=True, text=True, timeout=30)
                                                            
                                                            if fix_config_result_redo.returncode == 0:
                                                                stdout_filtered_fix = filter_malloc_warnings(fix_config_result_redo.stdout)
                                                                if "updated successfully" in fix_config_result_redo.stdout:
                                                                    terminal_output.append(f"[SUCCESS] Config file verified and fixed: {stdout_filtered_fix.strip()}")
                                                                elif "already correct" in fix_config_result_redo.stdout:
                                                                    terminal_output.append(f"[INFO] Config file verified: {stdout_filtered_fix.strip()}")
                                                                else:
                                                                    terminal_output.append(f"[INFO] Config check completed: {stdout_filtered_fix.strip()}")
                                                                
                                                                # Verify the actual config value
                                                                verify_path_cmd = [
                                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                                    f"root@{ssh_host}",
                                                                    "grep 'dataset_preparation_path' /workspace/data/axolotl_config.yaml"
                                                                ]
                                                                verify_path_result = subprocess.run(verify_path_cmd, capture_output=True, text=True, timeout=15)
                                                                if verify_path_result.returncode == 0:
                                                                    path_line = verify_path_result.stdout.strip()
                                                                    terminal_output.append(f"[VERIFY] Config contains: {path_line}")
                                                                    if "/workspace/axolotl/prepared_data" not in path_line:
                                                                        terminal_output.append(f"[WARNING] Path might not be absolute in config file!")
                                                            else:
                                                                error_msg_fix = filter_malloc_warnings(fix_config_result_redo.stderr or fix_config_result_redo.stdout)
                                                                terminal_output.append(f"[WARNING] Config verification failed: {error_msg_fix[:200]}")
                                                    else:
                                                        error_output = stderr_filtered or stdout_filtered
                                                        terminal_output.append(f"[ERROR] Config re-upload failed after 3 attempts (exit code {scp_config_result.returncode})")
                                                        terminal_output.append(f"[ERROR] {error_output[:500]}")
                                                        terminal_output.append(f"[INFO] You may need to manually upload the config file via SSH")
                                            else:
                                                terminal_output.append(f"[WARNING] Could not map base_model '{base_model}' to HF model. Using existing config.")
                                        except Exception as e:
                                            error_msg = str(e)
                                            terminal_output.append(f"[ERROR] Failed to regenerate config: {error_msg}")
                                            import traceback
                                            terminal_output.append(f"[ERROR] Traceback: {traceback.format_exc()[:300]}")
                                    
                                    # Step 3: Restart training
                                    terminal_output.append(f"[SSH] Restarting training...")
                                    
                                    # Get Hugging Face token for gated models
                                    hf_token = get_hf_token()
                                    
                                    # Debug: Check what we got (without showing the actual token)
                                    if hf_token:
                                        token_length = len(str(hf_token))
                                        token_type = type(hf_token).__name__
                                        terminal_output.append(f"[DEBUG] HF token retrieved: type={token_type}, length={token_length}")
                                    else:
                                        terminal_output.append(f"[DEBUG] HF token not found or is None/empty")
                                    
                                    # Build HF token export section - escape single quotes properly
                                    hf_token_export = ""
                                    if hf_token and isinstance(hf_token, str) and hf_token.strip():
                                        # Remove any newlines or control characters that could break the command
                                        hf_token_clean = hf_token.strip().replace('\n', '').replace('\r', '').replace(';', '').replace('`', '')
                                        if hf_token_clean and len(hf_token_clean) > 0:  # Double-check it's not empty after cleaning
                                            try:
                                                # Escape single quotes by replacing ' with '\'' (end quote, escaped quote, start quote)
                                                hf_token_escaped = hf_token_clean.replace("'", "'\"'\"'")
                                                # Test that the export command would be valid
                                                test_export = f"export HF_TOKEN='{hf_token_escaped}'"
                                                if "'" in test_export and test_export.count("'") >= 2:  # Should have at least opening and closing quotes
                                                    hf_token_export = f"export HF_TOKEN='{hf_token_escaped}' && export HUGGING_FACE_HUB_TOKEN='{hf_token_escaped}' && "
                                                    terminal_output.append(f"[INFO] HF token prepared for export (length: {len(hf_token_clean)})")
                                                else:
                                                    terminal_output.append(f"[WARNING] HF token export construction failed - skipping token export")
                                                    hf_token_export = ""
                                            except Exception as e:
                                                terminal_output.append(f"[WARNING] Error preparing HF token export: {str(e)[:100]} - skipping token export")
                                                hf_token_export = ""
                                        else:
                                            hf_token_export = ""
                                            terminal_output.append(f"[WARNING] HF token is empty after cleaning - gated models may fail")
                                    else:
                                        hf_token_export = ""
                                        if not hf_token:
                                            terminal_output.append(f"[WARNING] No HF token found - gated models (e.g., Gemma) may fail to load")
                                        elif not isinstance(hf_token, str):
                                            terminal_output.append(f"[WARNING] HF token is not a string (type: {type(hf_token).__name__}) - skipping token export")
                                    
                                    # Unified training start: find Python with accelerate and axolotl, then use python -m accelerate.commands.launch
                                    # We'll verify the process started separately
                                    # Final safety check - if export looks wrong, clear it
                                    if hf_token_export:
                                        # Check if export command is incomplete or malformed
                                        if (not hf_token_export.strip().endswith(" && ") and 
                                            not hf_token_export.strip().endswith(" &&") and
                                            "HF_TOKEN=" not in hf_token_export):
                                            terminal_output.append(f"[ERROR] HF token export format invalid - clearing export")
                                            hf_token_export = ""
                                    # Ensure hf_token_export ends with && if it's not empty, or is empty string
                                    if hf_token_export and not hf_token_export.endswith(" && "):
                                        if hf_token_export.endswith(" &&"):
                                            hf_token_export = hf_token_export + " "
                                        elif not hf_token_export.endswith(" "):
                                            hf_token_export = hf_token_export + " && "
                                    
                                    # Build the command as a single line to avoid issues with SSH command parsing
                                    restart_command_parts = []
                                    if hf_token_export:
                                        restart_command_parts.append(hf_token_export.rstrip())
                                    restart_command_parts.extend([
                                        "cd /workspace/axolotl && ",
                                        "TRAIN_PYTHON='' && ",
                                        "if /opt/conda/bin/python -c 'import accelerate; import axolotl' 2>/dev/null; then ",
                                        " TRAIN_PYTHON='/opt/conda/bin/python'; ",
                                        "elif python3 -c 'import accelerate; import axolotl' 2>/dev/null; then ",
                                        " TRAIN_PYTHON='python3'; ",
                                        "elif python -c 'import accelerate; import axolotl' 2>/dev/null; then ",
                                        " TRAIN_PYTHON='python'; ",
                                        "fi && ",
                                        "if [ -z \"$TRAIN_PYTHON\" ]; then ",
                                        " echo 'ERROR: accelerate or axolotl not found'; exit 1; ",
                                        "fi && ",
                                        "if [[ \"$TRAIN_PYTHON\" == /* ]]; then ",
                                        " PYTHON_DIR=$(dirname \"$TRAIN_PYTHON\"); ",
                                        " ACCELERATE_CMD=\"$PYTHON_DIR/accelerate\"; ",
                                        "else ",
                                        " ACCELERATE_CMD=$(command -v accelerate 2>/dev/null || echo \"accelerate\"); ",
                                        "fi && ",
                                        "if [ -f \"$ACCELERATE_CMD\" ] || command -v accelerate >/dev/null 2>&1; then ",
                                        " nohup $ACCELERATE_CMD launch -m axolotl.cli.train /workspace/data/axolotl_config.yaml ",
                                        "> /workspace/output/training/training.log 2>&1 < /dev/null &; ",
                                        "else ",
                                        " nohup $TRAIN_PYTHON -m accelerate.commands.launch -m axolotl.cli.train /workspace/data/axolotl_config.yaml ",
                                        "> /workspace/output/training/training.log 2>&1 < /dev/null &; ",
                                        "fi"
                                    ])
                                    restart_command = "".join(restart_command_parts)
                                    
                                    # Ensure no newlines that could break SSH command parsing
                                    restart_command = restart_command.replace('\n', ' ').replace('\r', ' ')
                                    
                                    # Final debug - show the actual command length and first/last parts
                                    terminal_output.append(f"[DEBUG] Final command length: {len(restart_command)} chars")
                                    terminal_output.append(f"[DEBUG] Command first 150 chars: {repr(restart_command[:150])}")
                                    if len(restart_command) > 300:
                                        terminal_output.append(f"[DEBUG] Command last 150 chars: {repr(restart_command[-150:])}")
                                    # Final validation - ensure export command is well-formed
                                    if hf_token_export:
                                        # Check for common malformed patterns - if token appears empty in export, skip it
                                        if ("export HF_TOKEN=''" in hf_token_export or 
                                            "export HF_TOKEN=' &&" in hf_token_export or 
                                            hf_token_export.strip() == "export" or 
                                            (hf_token_export.strip().startswith("export ") and "HF_TOKEN" not in hf_token_export) or
                                            "HF_TOKEN='' &&" in hf_token_export or
                                            "HF_TOKEN=' &&" in hf_token_export):
                                            terminal_output.append(f"[ERROR] HF token export appears malformed or empty - skipping token export")
                                            terminal_output.append(f"[DEBUG] Malformed export (first 100 chars): {repr(hf_token_export[:100])}")
                                            hf_token_export = ""
                                            terminal_output.append(f"[WARNING] Training will proceed without HF token - gated models will fail")
                                        else:
                                            # Verify the export actually contains a token value (not just empty quotes)
                                            if "HF_TOKEN=''" in hf_token_export or "HF_TOKEN=' &&" in hf_token_export:
                                                terminal_output.append(f"[ERROR] HF token export contains empty value - skipping")
                                                hf_token_export = ""
                                            else:
                                                terminal_output.append(f"[INFO] Training command includes HF token for gated model access")
                                    elif not hf_token:
                                        terminal_output.append(f"[WARNING] No HF token available - gated models may fail")
                                    
                                    # Debug: Show what we're about to send (first 200 chars, sanitized)
                                    if hf_token_export:
                                        debug_cmd_start = (hf_token_export + "cd /workspace/axolotl && ")[:200]
                                        terminal_output.append(f"[DEBUG] Command start (sanitized): {repr(debug_cmd_start)}")
                                    
                                    # Use a here-document to pass the command more reliably through SSH
                                    # This avoids issues with quote escaping and command length
                                    # Format token exports for here-document (one per line, no &&)
                                    token_exports = ""
                                    if hf_token and isinstance(hf_token, str) and hf_token.strip():
                                        hf_token_clean = hf_token.strip().replace('\n', '').replace('\r', '').replace(';', '').replace('`', '')
                                        if hf_token_clean:
                                            hf_token_escaped = hf_token_clean.replace("'", "'\"'\"'")
                                            token_exports = f"export HF_TOKEN='{hf_token_escaped}'\nexport HUGGING_FACE_HUB_TOKEN='{hf_token_escaped}'\n"
                                    
                                    ssh_command = f"""bash << 'REMOTE_SCRIPT'
set -e
{token_exports}cd /workspace/axolotl
TRAIN_PYTHON=''
if /opt/conda/bin/python -c 'import accelerate; import axolotl' 2>/dev/null; then
 TRAIN_PYTHON='/opt/conda/bin/python'
elif python3 -c 'import accelerate; import axolotl' 2>/dev/null; then
 TRAIN_PYTHON='python3'
elif python -c 'import accelerate; import axolotl' 2>/dev/null; then
 TRAIN_PYTHON='python'
fi
if [ -z "$TRAIN_PYTHON" ]; then
 echo 'ERROR: accelerate or axolotl not found'
 exit 1
fi
if [[ "$TRAIN_PYTHON" == /* ]]; then
 PYTHON_DIR=$(dirname "$TRAIN_PYTHON")
 ACCELERATE_CMD="$PYTHON_DIR/accelerate"
else
 ACCELERATE_CMD=$(command -v accelerate 2>/dev/null || echo "accelerate")
fi
if [ -f "$ACCELERATE_CMD" ] || command -v accelerate >/dev/null 2>&1; then
 nohup $ACCELERATE_CMD launch -m axolotl.cli.train /workspace/data/axolotl_config.yaml > /workspace/output/training/training.log 2>&1 < /dev/null &
else
 nohup $TRAIN_PYTHON -m accelerate.commands.launch -m axolotl.cli.train /workspace/data/axolotl_config.yaml > /workspace/output/training/training.log 2>&1 < /dev/null &
fi
REMOTE_SCRIPT"""
                                    
                                    restart_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                        f"root@{ssh_host}",
                                        ssh_command
                                    ]
                                    try:
                                        restart_result = subprocess.run(restart_cmd, capture_output=True, text=True, timeout=5)
                                        if restart_result.returncode == 0:
                                            terminal_output.append(f"[SSH] Training command executed")
                                            if restart_result.stdout:
                                                stdout_filtered = filter_malloc_warnings(restart_result.stdout)
                                                if stdout_filtered.strip():
                                                    terminal_output.append(f"[SSH] {stdout_filtered}")
                                        else:
                                            # Even if return code is non-zero, the process might have started
                                            terminal_output.append(f"[SSH] Training command sent (checking process status...)")
                                            if restart_result.stderr:
                                                stderr_filtered = filter_malloc_warnings(restart_result.stderr)
                                                if stderr_filtered.strip():
                                                    terminal_output.append(f"[SSH] Command stderr: {stderr_filtered[:200]}")
                                    except subprocess.TimeoutExpired:
                                        # Timeout is expected - the process is running in background
                                        terminal_output.append(f"[SSH] Training command sent (timeout expected for background process)")
                                    
                                    # Wait a moment and then check if the process actually started and if logs are being written
                                    import time
                                    time.sleep(3)
                                    
                                    # Quick check to see if log file is being created
                                    quick_log_check_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "test -f /workspace/output/training/training.log && (wc -l /workspace/output/training/training.log && tail -10 /workspace/output/training/training.log) || echo 'log_not_created'"
                                    ]
                                    quick_log_check = subprocess.run(quick_log_check_cmd, capture_output=True, text=True, timeout=10)
                                    if "log_not_created" not in quick_log_check.stdout:
                                        log_output = quick_log_check.stdout.strip()
                                        # Split into lines count and content
                                        lines = log_output.split('\n')
                                        if len(lines) > 0:
                                            log_lines = lines[0]
                                            terminal_output.append(f"[SSH] Log file created: {log_lines}")
                                            if len(lines) > 1:
                                                log_content = '\n'.join(lines[1:])
                                                if log_content.strip():
                                                    terminal_output.append(f"[SSH] Recent log content:")
                                                    for line in log_content.strip().split('\n')[:5]:
                                                        if line.strip():
                                                            terminal_output.append(f"[SSH]   {line[:200]}")
                                    else:
                                        terminal_output.append(f"[WARNING] Log file not created yet - training may not have started")
                                        
                                        # Check if there are any Python processes that might have failed
                                        check_python_errors_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "ps aux | grep python | grep -E '(axolotl|accelerate|train)' | grep -v grep | head -3 || echo 'no_python_process'"
                                        ]
                                        python_process_check = subprocess.run(check_python_errors_cmd, capture_output=True, text=True, timeout=10)
                                        if "no_python_process" not in python_process_check.stdout and python_process_check.stdout.strip():
                                            terminal_output.append(f"[DIAGNOSTICS] Found Python processes:")
                                            for line in python_process_check.stdout.strip().split('\n')[:3]:
                                                if line.strip():
                                                    terminal_output.append(f"[DIAGNOSTICS]   {line[:200]}")
                                    
                                    # Always check if process started, regardless of command result
                                    terminal_output.append(f"[SSH] Verifying training process started...")
                                    
                                    # Wait a moment and verify training process is running
                                    verify_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "sleep 3 && ps aux | grep -E '(accelerate|axolotl)' | grep -v grep | head -2 || echo 'no_process'"
                                    ]
                                    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=15)
                                    if "no_process" not in verify_result.stdout:
                                        stdout_filtered = filter_malloc_warnings(verify_result.stdout)
                                        terminal_output.append(f"[SSH] Training process found:")
                                        for line in stdout_filtered.strip().split("\n")[:2]:
                                            if line.strip():
                                                terminal_output.append(f"[SSH]   {line[:150]}")
                                        terminal_output.append(f"[SUCCESS] Phase 3 redo complete - training restarted")
                                        
                                        # Reset training status
                                        active_job["training_status"] = {"status": "training", "restarted": True}
                                        training_manager._save_job(active_job)
                                    else:
                                            terminal_output.append(f"[WARNING] Training command executed but process not found yet")
                                            terminal_output.append(f"[INFO] Training may still be starting. Use 'Check Training Status' to verify.")
                                            terminal_output.append(f"[INFO] Check /workspace/output/training/training.log for details")
                                            # Reset training status so "Start Training" button appears if training didn't start
                                            active_job["training_status"] = {"status": "not_started", "message": "Training command sent but process not verified"}
                                            training_manager._save_job(active_job)
                                
                                st.session_state[terminal_output_key] = terminal_output
                                st.rerun()
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] Failed to redo Phase 3: {error_msg}")
                                # Reset training status so "Start Training" button appears after error
                                active_job["training_status"] = {"status": "not_started", "error": error_msg}
                                training_manager._save_job(active_job)
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                    
                    with col4:
                        training_status = active_job.get("training_status", {})
                        if training_status.get("status") == "completed":
                            st.success("✅ Training completed! Click 'Next Phase' to finalize.")
                            if st.button("➡️ Next Phase", key="next_phase_3", type="primary"):
                                st.session_state[phase_key] = 4
                                # Clear terminal for next phase
                                st.session_state[terminal_output_key] = []
                                st.rerun()
                
                # Phase 4: Finalize
                elif current_phase == 4:
                    # Ensure datetime is available in this scope
                    from datetime import datetime
                    
                    st.markdown(f"### {phases[4]['icon']} Phase 4: {phases[4]['name']}")
                    st.caption(phases[4]['description'])
                    
                    # Training Summary Section
                    st.markdown("#### 📊 Training Summary")
                    summary_container = st.container()
                    with summary_container:
                        # Get job queue information
                        job_queue = active_job.get("job_queue", [])
                        total_jobs = len(job_queue) if job_queue else 1
                        
                        # Try to get training statistics
                        instance_id = active_job.get("instance_id")
                        ssh_host = active_job.get("ssh_host")
                        ssh_port = active_job.get("ssh_port", 22)
                        
                        summary_lines = []
                        summary_lines.append(f"✅ **Training Completed Successfully**")
                        summary_lines.append(f"")
                        summary_lines.append(f"**Total Jobs Completed:** {total_jobs}")
                        summary_lines.append(f"")
                        
                        # Try to retrieve training statistics from logs
                        all_stats = []
                        if ssh_host and instance_id:
                            try:
                                import subprocess
                                # Get training logs from instance to extract statistics
                                log_command = (
                                    "if [ -f /workspace/output/training/training.log ]; then "
                                    "cat /workspace/output/training/training.log; "
                                    "elif [ -f /workspace/axolotl/training.log ]; then "
                                    "cat /workspace/axolotl/training.log; "
                                    "elif [ -f /workspace/output/training/debug.log ]; then "
                                    "cat /workspace/output/training/debug.log; "
                                    "else echo 'no_logs'; fi"
                                )
                                log_cmd = [
                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                    f"root@{ssh_host}",
                                    log_command
                                ]
                                log_result = subprocess.run(log_cmd, capture_output=True, text=True, timeout=30)
                                
                                if "no_logs" not in log_result.stdout and log_result.stdout.strip():
                                    # Extract dataset statistics from full logs
                                    full_logs = filter_malloc_warnings(log_result.stdout)
                                    stats = extract_dataset_stats(full_logs)
                                    if stats:
                                        all_stats.append(stats)
                            except Exception as e:
                                # If we can't retrieve logs, that's okay - we'll show what we have
                                pass
                        
                        # Also check for stored stats in job results
                        if job_queue:
                            for job_idx, job_item in enumerate(job_queue):
                                job_results_path = Path(f"models/{model_name}/training/queue/job_{job_idx + 1}_results")
                                if job_results_path.exists():
                                    # Check for any stored stats files
                                    stats_files = list(job_results_path.glob("*stats*.json"))
                                    for stats_file in stats_files:
                                        try:
                                            import json
                                            with open(stats_file, 'r') as f:
                                                stored_stats = json.load(f)
                                                if stored_stats:
                                                    all_stats.append(stored_stats)
                                        except:
                                            pass
                        
                        # Display statistics
                        if all_stats:
                            # Aggregate stats across all jobs
                            total_original = sum(s.get('original_count', 0) for s in all_stats)
                            total_final = sum(s.get('final_count', 0) for s in all_stats)
                            total_dropped = sum(s.get('total_dropped', 0) for s in all_stats)
                            total_dropped_long = sum(s.get('dropped_long', 0) for s in all_stats)
                            total_dropped_zero = sum(s.get('dropped_zero_tokens', 0) for s in all_stats)
                            
                            summary_lines.append(f"**Dataset Statistics:**")
                            if total_original > 0:
                                summary_lines.append(f"  • Original training samples: **{total_original:,}**")
                            if total_final > 0:
                                summary_lines.append(f"  • Final training samples: **{total_final:,}**")
                                if total_original > 0:
                                    success_rate = (total_final / total_original * 100) if total_original > 0 else 0
                                    summary_lines.append(f"  • Success rate: **{success_rate:.1f}%**")
                            if total_dropped > 0:
                                summary_lines.append(f"  • Samples dropped: **{total_dropped:,}**")
                                if total_original > 0:
                                    drop_rate = (total_dropped / total_original * 100) if total_original > 0 else 0
                                    summary_lines.append(f"  • Drop rate: **{drop_rate:.1f}%**")
                            if total_dropped_long > 0:
                                summary_lines.append(f"    - Dropped (too long): {total_dropped_long:,}")
                            if total_dropped_zero > 0:
                                summary_lines.append(f"    - Dropped (zero tokens): {total_dropped_zero:,}")
                            
                            # Show per-job breakdown if multiple jobs
                            if total_jobs > 1 and len(all_stats) > 0:
                                summary_lines.append(f"")
                                summary_lines.append(f"**Per-Job Breakdown:**")
                                for job_idx, stats in enumerate(all_stats[:total_jobs]):
                                    job_num = job_idx + 1
                                    job_final = stats.get('final_count', 0)
                                    job_original = stats.get('original_count', 0)
                                    if job_final > 0 or job_original > 0:
                                        summary_lines.append(f"  • Job {job_num}: {job_final:,} samples trained" + 
                                                           (f" (from {job_original:,} original)" if job_original > 0 else ""))
                        else:
                            summary_lines.append(f"**Note:** Detailed statistics not available. Training completed successfully.")
                        
                        summary_lines.append(f"")
                        summary_lines.append(f"---")
                        
                        # Display summary
                        st.info("\n".join(summary_lines))
                    
                    # Terminal output area (scrollable)
                    st.markdown("#### Terminal Output")
                    terminal_container = st.container()
                    with terminal_container:
                        if terminal_output:
                            # Keep only last 200 lines
                            display_output = terminal_output[-200:] if len(terminal_output) > 200 else terminal_output
                            output_text = "\n".join(display_output)
                            st.code(output_text, language="text")
                            if len(terminal_output) > 200:
                                st.caption(f"Showing last 200 of {len(terminal_output)} lines")
                        else:
                            st.info("No output yet. Click 'Finalize Training' to start.")
                    
                    # Clear button at the bottom
                    if st.button("🗑️ Clear Terminal", key="clear_terminal_phase3", help="Clear terminal output"):
                        st.session_state[terminal_output_key] = []
                        st.rerun()
                    
                    # Check if weights have been downloaded
                    weights_downloaded = active_job.get("weights_downloaded", False)
                    version_dir_str = active_job.get("version_dir")
                    
                    # Action buttons
                    if not weights_downloaded:
                        # Step 1: Download weights
                        st.markdown("#### 📥 Step 1: Download Weights")
                        st.info("Download the trained weights from the instance before finalizing.")
                        
                        if st.button("📥 Download Weights", key="download_weights", type="primary"):
                            try:
                                instance_id = active_job.get("instance_id")
                                
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting weight download...")
                                
                                # Step 1: Create version directory
                                terminal_output.append(f"[LOCAL] Creating version directory...")
                                from utils.model_manager import ModelManager
                                model_manager = ModelManager()
                                version_dir = model_manager.create_version_folder(model_name)
                                terminal_output.append(f"[LOCAL] Version directory created: {version_dir}")
                                
                                # Step 2: Get SSH info for weight download
                                # Prefer saved SSH details from job over API
                                ssh_host = active_job.get("ssh_host")
                                ssh_port = active_job.get("ssh_port", 22)
                                
                                # If not in job, get from API
                                if not ssh_host:
                                    terminal_output.append(f"[API] Getting instance status for SSH info...")
                                    job_status = training_manager.get_job_status(instance_id)
                                    ssh_host = job_status.get("ssh_host")
                                    ssh_port = job_status.get("ssh_port", 22)
                                    # Save to job for future use
                                    if ssh_host:
                                        active_job["ssh_host"] = ssh_host
                                        active_job["ssh_port"] = ssh_port
                                        training_manager._save_job(active_job)
                                
                                if not ssh_host:
                                    terminal_output.append(f"[ERROR] SSH host not available. Cannot download weights.")
                                    st.error("SSH host not available. Cannot download weights.")
                                else:
                                    terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                    
                                    # Step 3: Check for weight files on remote instance
                                    terminal_output.append(f"[SSH] Checking for weight files on remote instance...")
                                    import subprocess
                                    check_files_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "find /workspace/output -type f \\( -name '*.bin' -o -name '*.safetensors' -o -name 'adapter_config.json' \\) 2>/dev/null | head -20"
                                    ]
                                    check_result = subprocess.run(check_files_cmd, capture_output=True, text=True, timeout=30)
                                    if check_result.returncode == 0 and check_result.stdout.strip():
                                        stdout_filtered = filter_malloc_warnings(check_result.stdout)
                                        available_files = [f.strip() for f in stdout_filtered.strip().split('\n') if f.strip()]
                                        terminal_output.append(f"[SSH] Found {len(available_files)} weight file(s) on remote instance")
                                        for f in available_files[:5]:
                                            terminal_output.append(f"[SSH]   - {f}")
                                    else:
                                        terminal_output.append(f"[SSH] No weight files found in standard locations")
                                    
                                    # Step 4: Download weights
                                    weights_dir = version_dir / "weights"
                                    weights_dir.mkdir(parents=True, exist_ok=True)
                                    terminal_output.append(f"[SCP] Downloading weights to: {weights_dir}")
                                    
                                    # Try downloading from adapter directory first
                                    scp_adapter_cmd = [
                                        "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", "-r",
                                        f"root@{ssh_host}:/workspace/output/training/adapter/*",
                                        str(weights_dir) + "/"
                                    ]
                                    scp_result = subprocess.run(scp_adapter_cmd, capture_output=True, text=True, timeout=300)
                                    if scp_result.returncode == 0:
                                        terminal_output.append(f"[SCP] Downloaded weights from /adapter directory")
                                    else:
                                        # Try root output directory
                                        terminal_output.append(f"[SCP] Trying root output directory...")
                                        scp_output_cmd = [
                                            "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", "-r",
                                            f"root@{ssh_host}:/workspace/output/training/*",
                                            str(weights_dir) + "/"
                                        ]
                                        scp_result = subprocess.run(scp_output_cmd, capture_output=True, text=True, timeout=300)
                                        if scp_result.returncode == 0:
                                            terminal_output.append(f"[SCP] Downloaded weights from output directory")
                                        else:
                                            error_output = scp_result.stderr or scp_result.stdout
                                            # Filter MallocStackLogging warnings
                                            error_output = filter_malloc_warnings(error_output)
                                            if "Welcome to vast.ai" in error_output:
                                                lines = error_output.split('\n')
                                                actual_errors = [line for line in lines if line and 'Welcome to vast.ai' not in line and 'Have fun!' not in line]
                                                error_output = '\n'.join(actual_errors) if actual_errors else error_output
                                            terminal_output.append(f"[SCP] Download error: {error_output[:300]}")
                                    
                                    # Verify downloaded files
                                    downloaded_files = list(weights_dir.rglob("*"))
                                    downloaded_files = [f for f in downloaded_files if f.is_file()]
                                    if downloaded_files:
                                        terminal_output.append(f"[LOCAL] Downloaded {len(downloaded_files)} weight file(s)")
                                        for f in downloaded_files[:5]:
                                            terminal_output.append(f"[FILE]   - {f.name}")
                                        
                                        # Mark weights as downloaded
                                        active_job["weights_downloaded"] = True
                                        active_job["version_dir"] = str(version_dir)
                                        active_job["weights_path"] = str(weights_dir)
                                        training_manager._save_job(active_job)
                                        
                                        terminal_output.append(f"[SUCCESS] Weights downloaded successfully!")
                                        terminal_output.append(f"[INFO] Weights saved to: {weights_dir}")
                                        
                                        st.session_state[terminal_output_key] = terminal_output
                                        st.success("✅ Weights downloaded successfully! You can now proceed to end training.")
                                        st.rerun()
                                    else:
                                        terminal_output.append(f"[WARNING] No weight files found after download")
                                        st.session_state[terminal_output_key] = terminal_output
                                        st.warning("⚠️ No weight files found after download. Please check the instance manually.")
                                
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] {error_msg}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                    else:
                        # Step 2: End Training (cleanup and finalize)
                        st.markdown("#### ✅ Step 2: End Training")
                        st.success("✅ Weights downloaded. Click 'End Training' to finalize and clean up.")
                        
                        if st.button("✅ End Training", key="end_training", type="primary"):
                            try:
                                instance_id = active_job.get("instance_id")
                                
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting finalization...")
                                
                                # Get version directory (should already exist from download step)
                                version_dir_str = active_job.get("version_dir")
                                if not version_dir_str:
                                    # Fallback: create if not exists
                                    from utils.model_manager import ModelManager
                                    model_manager = ModelManager()
                                    version_dir = model_manager.create_version_folder(model_name)
                                    version_dir_str = str(version_dir)
                                else:
                                    from pathlib import Path
                                    version_dir = Path(version_dir_str)
                                
                                # Step 1: Move files from queue (only files from current job)
                                terminal_output.append(f"[LOCAL] Moving files from queue to version folder...")
                                from utils.config import get_model_queue_dir
                                from pathlib import Path
                                import shutil
                                queue_dir = get_model_queue_dir(model_name)
                                training_dir = version_dir / "training"
                                training_dir.mkdir(parents=True, exist_ok=True)
                                moved_files = []
                                
                                # Get current job's file group to only move those files
                                job_queue = active_job.get("job_queue")
                                current_job_index = active_job.get("current_job_index")
                                current_job_files = set()
                                
                                if job_queue and current_job_index is not None:
                                    current_job = job_queue[current_job_index]
                                    file_group = current_job.get("file_group", [])
                                    for file_meta in file_group:
                                        filename = file_meta.get("filename")
                                        if filename:
                                            current_job_files.add(filename)
                                            # Also check for metadata filename
                                            current_job_files.add(f"{Path(filename).stem}_metadata.json")
                                
                                if queue_dir.exists():
                                    for file_path in queue_dir.iterdir():
                                        if file_path.is_file():
                                            # Only move files that belong to current job (or all if no queue)
                                            if not job_queue or file_path.name in current_job_files:
                                                try:
                                                    dest_path = training_dir / file_path.name
                                                    shutil.move(str(file_path), str(dest_path))
                                                    moved_files.append(file_path.name)
                                                    terminal_output.append(f"[FILE] Moved: {file_path.name}")
                                                except Exception as e:
                                                    terminal_output.append(f"[ERROR] Failed to move {file_path.name}: {str(e)}")
                                
                                terminal_output.append(f"[LOCAL] Moved {len(moved_files)} file(s) to version folder")
                                
                                # File Processing Debugging - Show expected vs actual files
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] === File Processing Debugging ===")
                                
                                # Get expected files from job queue or package_info
                                expected_files = []
                                expected_metadata_files = []
                                
                                if job_queue and current_job_index is not None:
                                    current_job = job_queue[current_job_index]
                                    file_group = current_job.get("file_group", [])
                                    terminal_output.append(f"[FILE DEBUG] Job queue index: {current_job_index}")
                                    terminal_output.append(f"[FILE DEBUG] Expected files from job queue: {len(file_group)} file(s)")
                                    
                                    for file_meta in file_group:
                                        filename = file_meta.get("filename")
                                        if filename:
                                            expected_files.append(filename)
                                            # Also expect metadata file
                                            from pathlib import Path
                                            metadata_filename = f"{Path(filename).stem}_metadata.json"
                                            expected_metadata_files.append(metadata_filename)
                                            terminal_output.append(f"[FILE DEBUG]   Expected: {filename}")
                                            terminal_output.append(f"[FILE DEBUG]   Expected metadata: {metadata_filename}")
                                else:
                                    # Fallback: check package_info or try to infer from moved files
                                    terminal_output.append(f"[FILE DEBUG] No job queue - inferring from package_info")
                                    # If we have package_info, we can check dataset_path
                                    dataset_path = package_info.get("dataset_path")
                                    if dataset_path:
                                        terminal_output.append(f"[FILE DEBUG] Dataset path: {dataset_path}")
                                
                                terminal_output.append(f"[FILE DEBUG] ========================================")
                                terminal_output.append(f"[FILE DEBUG] Expected files: {len(expected_files)}")
                                terminal_output.append(f"[FILE DEBUG] Expected metadata files: {len(expected_metadata_files)}")
                                
                                # Get actual moved files
                                actual_files = [f for f in moved_files if not f.endswith("_metadata.json")]
                                actual_metadata_files = [f for f in moved_files if f.endswith("_metadata.json")]
                                
                                terminal_output.append(f"[FILE DEBUG] ========================================")
                                terminal_output.append(f"[FILE DEBUG] Actually moved files: {len(actual_files)}")
                                for f in actual_files:
                                    terminal_output.append(f"[FILE DEBUG]   Moved: {f}")
                                
                                terminal_output.append(f"[FILE DEBUG] Actually moved metadata files: {len(actual_metadata_files)}")
                                for f in actual_metadata_files:
                                    terminal_output.append(f"[FILE DEBUG]   Moved metadata: {f}")
                                
                                # Compare expected vs actual
                                terminal_output.append(f"[FILE DEBUG] ========================================")
                                if expected_files:
                                    expected_set = set(expected_files)
                                    actual_set = set(actual_files)
                                    
                                    missing_files = expected_set - actual_set
                                    extra_files = actual_set - expected_set
                                    
                                    if not missing_files and not extra_files:
                                        terminal_output.append(f"[FILE DEBUG] ✓ All expected files were processed successfully!")
                                        terminal_output.append(f"[FILE DEBUG]   Files matched: {len(expected_set)}/{len(expected_set)}")
                                    else:
                                        if missing_files:
                                            terminal_output.append(f"[FILE DEBUG] ✗ Missing files ({len(missing_files)}):")
                                            for f in sorted(missing_files):
                                                terminal_output.append(f"[FILE DEBUG]   - {f}")
                                        if extra_files:
                                            terminal_output.append(f"[FILE DEBUG] ⚠ Extra files found ({len(extra_files)}):")
                                            for f in sorted(extra_files):
                                                terminal_output.append(f"[FILE DEBUG]   + {f}")
                                    
                                    # Check metadata files
                                    expected_metadata_set = set(expected_metadata_files)
                                    actual_metadata_set = set(actual_metadata_files)
                                    
                                    missing_metadata = expected_metadata_set - actual_metadata_set
                                    if missing_metadata:
                                        terminal_output.append(f"[FILE DEBUG] ✗ Missing metadata files ({len(missing_metadata)}):")
                                        for f in sorted(missing_metadata):
                                            terminal_output.append(f"[FILE DEBUG]   - {f}")
                                else:
                                    terminal_output.append(f"[FILE DEBUG] ⚠ Could not determine expected files (no job queue)")
                                    terminal_output.append(f"[FILE DEBUG]   Processed {len(actual_files)} file(s) total")
                                
                                # Also check what files were processed into the training dataset
                                dataset_path = package_info.get("dataset_path")
                                if dataset_path and Path(dataset_path).exists():
                                    terminal_output.append(f"[FILE DEBUG] Checking training dataset: {dataset_path}")
                                    try:
                                        dataset_line_count = 0
                                        with open(dataset_path, 'r', encoding='utf-8') as f:
                                            for line in f:
                                                if line.strip():
                                                    dataset_line_count += 1
                                        terminal_output.append(f"[FILE DEBUG] Dataset contains {dataset_line_count} training examples")
                                        
                                        # Try to infer source files from dataset if possible
                                        # (This is approximate - dataset combines all files)
                                        if expected_files:
                                            terminal_output.append(f"[FILE DEBUG] Dataset should contain data from {len(expected_files)} source file(s)")
                                    except Exception as e:
                                        terminal_output.append(f"[FILE DEBUG] Could not analyze dataset: {str(e)}")
                                else:
                                    terminal_output.append(f"[FILE DEBUG] Dataset file not found at: {dataset_path}")
                                
                                # Final summary
                                terminal_output.append(f"[FILE DEBUG] ========================================")
                                if expected_files:
                                    if len(expected_files) == len(actual_files):
                                        terminal_output.append(f"[FILE DEBUG] ✓ SUMMARY: All {len(expected_files)} expected file(s) were processed")
                                    else:
                                        terminal_output.append(f"[FILE DEBUG] ✗ SUMMARY: Expected {len(expected_files)} file(s), but {len(actual_files)} were moved")
                                else:
                                    terminal_output.append(f"[FILE DEBUG] ⚠ SUMMARY: Processed {len(actual_files)} file(s) (expected count unknown)")
                                
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] === End File Processing Debugging ===")
                                
                                # Step 2: Finalize (do NOT stop or destroy instance)
                                terminal_output.append(f"[INFO] Training finalized successfully!")
                                terminal_output.append(f"[IMPORTANT] The Vast.ai instance is still running. You must shut it down manually in Vast.ai to stop charges.")
                                
                                # Step 3: Update job
                                active_job["status"] = "completed"
                                active_job["finalized"] = True
                                active_job["version"] = version_dir.name.replace("V", "")
                                training_manager._save_job(active_job)
                                
                                terminal_output.append(f"[SUCCESS] All training jobs finalized!")
                                terminal_output.append(f"[INFO] Version: {active_job.get('version')}")
                                weights_path = active_job.get("weights_path")
                                if weights_path:
                                    terminal_output.append(f"[INFO] Weights: {weights_path}")
                                
                                st.session_state[terminal_output_key] = terminal_output
                                st.success("✅ All training jobs finalized successfully!")
                                instance_id = active_job.get("instance_id")
                                if instance_id:
                                    st.warning(f"⚠️ **Important:** The Vast.ai instance ({instance_id[:8]}...) is still running. You must stop or destroy it manually in Vast.ai to stop charges.")
                                st.rerun()
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] {error_msg}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                    
                    # Additional actions
                    if active_job.get("finalized"):
                        st.success("✅ Training finalized! All phases complete.")
                        instance_id = active_job.get("instance_id")
                        if instance_id:
                            st.warning(f"⚠️ **Important:** The Vast.ai instance ({instance_id[:8]}...) may still be running. Check Vast.ai and stop or destroy it manually to stop charges.")
                    else:
                        # Allow going back to Phase 3 to redo training if needed
                        if st.button("🔄 Redo Phase 3", key="redo_phase_3_from_4", type="secondary", help="Go back to Phase 3 to restart training"):
                            st.session_state[phase_key] = 3
                            st.rerun()
                
            except Exception as e:
                st.warning(f"Could not load training jobs: {str(e)}")
                import traceback
                with st.expander("Error Details", expanded=False):
                    st.code(traceback.format_exc())
        
        # Troubleshooting section - at bottom, collapsed by default
        st.markdown("---")
        with st.expander("🔧 Troubleshooting & Debug Info", expanded=False):
            st.write(f"**Debug Info:**")
            st.write(f"- Queue directory exists: {queue_dir.exists() if queue_dir else False}")
            st.write(f"- Queue directory path: `{queue_dir}`")
            st.write(f"- Files detected: {len(queued_files)}")
            st.write(f"- has_training_data: {has_training_data}")
            
            if len(queued_files) > 0:
                st.write(f"**Detected Files:**")
                for f in queued_files:
                    st.write(f"  - {f.get('filename', 'unknown')}")
            else:
                st.write("**No files detected. Checking directory contents...**")
                if queue_dir and queue_dir.exists():
                    all_files = list(queue_dir.iterdir())
                    st.write(f"Files in directory ({len(all_files)}):")
                    for f in all_files:
                        st.write(f"  - {f.name} ({'file' if f.is_file() else 'dir'})")
                        if f.name.endswith('_metadata.json'):
                            try:
                                import json as json_module
                                with open(f, 'r', encoding='utf-8') as mf:
                                    meta = json_module.load(mf)
                                    st.write(f"    Model: {meta.get('model')}, Filename: {meta.get('filename')}, Is YAML: {meta.get('is_yaml', False)}")
                            except Exception as e:
                                error_type = type(e).__name__
                                st.write(f"    ❌ Error ({error_type}): {str(e)}")
            
            # Job reset options
            st.markdown("---")
            st.write("**Job Management:**")
            if vast_api_key:
                try:
                    from utils.vast_training_manager import VastTrainingManager
                    training_manager = VastTrainingManager(model_name, vast_api_key)
                    all_jobs = training_manager.list_jobs()
                    if all_jobs:
                        st.write(f"**Found {len(all_jobs)} job(s) in database:**")
                        for idx, job in enumerate(all_jobs):
                            job_id = job.get("instance_id", "no-id")
                            status = job.get("status", "unknown")
                            finalized = job.get("finalized", False)
                            dismissed = job.get("dismissed", False)
                            st.write(f"  - Job {idx+1}: ID={job_id}, Status={status}, Finalized={finalized}, Dismissed={dismissed}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Dismiss All Jobs", key="dismiss_all_jobs", help="Mark all jobs as dismissed to start fresh"):
                                try:
                                    for job in all_jobs:
                                        job["dismissed"] = True
                                        job["dismissed_at"] = datetime.now().isoformat()
                                        training_manager._save_job(job)
                                    st.success("✅ All jobs dismissed. Refresh the page to see changes.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        with col2:
                            if st.button("🔄 Clear All Job Data", key="clear_all_jobs", help="⚠️ WARNING: This will delete all job records (files remain safe)"):
                                try:
                                    # Delete the jobs file
                                    jobs_file = training_manager.jobs_file
                                    if jobs_file.exists():
                                        jobs_file.unlink()
                                        st.success("✅ All job data cleared. Refresh the page to see changes.")
                                        st.rerun()
                                    else:
                                        st.info("No job file found to clear.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                    else:
                        st.write("No jobs found in database.")
                except Exception as e:
                    st.write(f"Could not check job state: {str(e)}")
            else:
                st.info("Enter Vast.ai API key above to manage jobs.")
    
    # Tab 2: Context Upload
    with tab2:
        st.markdown("### 📄 Context Upload")
        st.caption("Upload JSON, JSONL, or TXT files to queue them for training. Upload YAML config files to attach to training jobs.")
        
        from utils.config import get_model_queue_dir
        from datetime import datetime  # Import here to avoid scoping issues
        from pathlib import Path  # Import here to avoid scoping issues
        queue_dir = get_model_queue_dir(model_name)
        queue_dir.mkdir(parents=True, exist_ok=True)
        
        # YAML Files Section
        st.markdown("#### 📋 YAML Config Files")
        st.caption("Upload YAML configuration files that can be attached to training jobs")
        
        yaml_uploader = st.file_uploader(
            "Upload YAML Config File",
            type=['yaml', 'yml'],
            help="Upload YAML configuration files (.yaml or .yml) that can be attached to JSON training files. Each YAML config will create a separate training job.",
            key="yaml_uploader"
        )
        
        if yaml_uploader is not None:
            try:
                # Save YAML file to queue directory
                yaml_path = queue_dir / yaml_uploader.name
                with open(yaml_path, 'wb') as f:
                    f.write(yaml_uploader.getbuffer())
                
                # Create metadata for YAML file
                yaml_metadata = {
                    "filename": yaml_uploader.name,
                    "file_type": "yaml",
                    "date": datetime.now().isoformat(),
                    "model": model_name,
                    "size": yaml_uploader.size,
                    "is_yaml": True
                }
                yaml_metadata_path = queue_dir / f"{Path(yaml_uploader.name).stem}_metadata.json"
                with open(yaml_metadata_path, 'w') as f:
                    json.dump(yaml_metadata, f, indent=2)
                
                st.success(f"✅ YAML file '{yaml_uploader.name}' uploaded successfully!")
                # Use st.session_state to prevent rerun loop - just clear the uploader
                if "yaml_uploader" in st.session_state:
                    del st.session_state["yaml_uploader"]
                st.rerun()
            except Exception as e:
                st.error(f"Error uploading YAML file: {str(e)}")
        
        # List YAML files
        yaml_files = []
        if queue_dir.exists():
            import json as json_module_tab2  # Use alias to avoid scoping issues
            for metadata_file in queue_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json_module_tab2.load(f)
                        if metadata.get("model") == model_name and metadata.get("is_yaml"):
                            filename = metadata.get("filename")
                            if filename:
                                file_path = queue_dir / filename
                                if file_path.exists():
                                    yaml_files.append(metadata)
                except Exception as e:
                    pass
        
        if yaml_files:
            st.markdown("**Available YAML Configs:**")
            for idx, yaml_meta in enumerate(yaml_files):
                yaml_filename = yaml_meta['filename']
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📋 {yaml_filename}")
                with col2:
                    yaml_delete_key = f"delete_yaml_{idx}_{yaml_filename}"
                    yaml_confirm_key = f"confirm_delete_yaml_{idx}_{yaml_filename}"
                    yaml_cancel_key = f"cancel_delete_yaml_{idx}_{yaml_filename}"
                    yaml_state_key = f"delete_yaml_state_{idx}_{yaml_filename}"
                    
                    if yaml_state_key not in st.session_state:
                        st.session_state[yaml_state_key] = False
                    
                    if not st.session_state[yaml_state_key]:
                        if st.button("🗑️", key=yaml_delete_key, help="Delete YAML file"):
                            st.session_state[yaml_state_key] = True
                            st.rerun()
                    else:
                        # Check for queued files that have this YAML attached
                        affected_files = []
                        if queue_dir.exists():
                            import json as json_module_tab2_attach  # Use alias to avoid scoping issues
                            for metadata_file in queue_dir.glob("*_metadata.json"):
                                try:
                                    with open(metadata_file, 'r', encoding='utf-8') as f:
                                        file_metadata = json_module_tab2_attach.load(f)
                                        if (file_metadata.get("model") == model_name and 
                                            file_metadata.get("attached_yaml") == yaml_filename and
                                            not file_metadata.get("is_yaml", False)):
                                            affected_files.append(file_metadata.get("filename", "unknown"))
                                except:
                                    pass
                        
                        if affected_files:
                            st.warning(f"⚠️ **Warning:** This YAML is attached to {len(affected_files)} queued file(s).")
                            st.write("**Files that will be affected:**")
                            for aff_file in affected_files:
                                st.write(f"  • {aff_file}")
                            st.write("**Action:** The YAML file will be deleted and removed from all attached files.")
                        else:
                            st.info("ℹ️ This YAML is not attached to any queued files.")
                        
                        if st.button("✅ Confirm Delete", key=yaml_confirm_key, help="Confirm delete", type="primary"):
                            try:
                                # Delete the YAML file
                                file_path = queue_dir / yaml_filename
                                if file_path.exists():
                                    file_path.unlink()
                                
                                # Delete the YAML metadata file
                                metadata_path = queue_dir / f"{Path(yaml_filename).stem}_metadata.json"
                                if metadata_path.exists():
                                    metadata_path.unlink()
                                
                                # Remove this YAML from all queued files that reference it
                                files_updated = 0
                                if queue_dir.exists():
                                    import json as json_module_detach  # Use alias to avoid scoping issues
                                    for metadata_file in queue_dir.glob("*_metadata.json"):
                                        try:
                                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                                file_metadata = json_module_detach.load(f)
                                            
                                            # Check if this file has the YAML attached
                                            if (file_metadata.get("model") == model_name and 
                                                file_metadata.get("attached_yaml") == yaml_filename and
                                                not file_metadata.get("is_yaml", False)):
                                                # Remove the attached_yaml reference
                                                file_metadata["attached_yaml"] = None
                                                # Save updated metadata
                                                with open(metadata_file, 'w', encoding='utf-8') as f:
                                                    json.dump(file_metadata, f, indent=2)
                                                files_updated += 1
                                        except Exception as update_error:
                                            st.warning(f"⚠️ Could not update {metadata_file.name}: {update_error}")
                                
                                success_msg = f"✅ Removed '{yaml_filename}'"
                                if files_updated > 0:
                                    success_msg += f" and removed it from {files_updated} queued file(s)"
                                st.success(success_msg)
                                
                                if yaml_state_key in st.session_state:
                                    del st.session_state[yaml_state_key]
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        if st.button("❌", key=yaml_cancel_key, help="Cancel"):
                            if yaml_state_key in st.session_state:
                                del st.session_state[yaml_state_key]
                            st.rerun()
        else:
            st.info("No YAML config files uploaded yet.")
        
        st.markdown("---")
        
        # Training File Upload Section
        st.markdown("#### 📄 Training Files")
        st.caption("Upload JSON, JSONL, or TXT files and optionally attach a YAML config, then click 'Process and Queue'")
        
        # File uploader (always show)
        uploader_key = "training_file_uploader"
        
        # Check if we just processed a file (to clear the display)
        file_processed_key = "training_file_just_processed"
        just_processed = file_processed_key in st.session_state
        
        if just_processed:
            # Clear the flag
            del st.session_state[file_processed_key]
            # Force clear the file uploader widget state to ensure it resets
            if uploader_key in st.session_state:
                del st.session_state[uploader_key]
        
        # Always show the file uploader widget
        uploaded_file = st.file_uploader(
            "Upload Training File",
            type=['json', 'jsonl', 'txt'],
            help="Upload JSON, JSONL, or TXT files to be used for training. After uploading, select a YAML config (optional) and click 'Process and Queue'.",
            key=uploader_key
        )
        
        # Add a reset button to manually clear the uploader if needed
        reset_uploader_key = "reset_uploader_clicked"
        if st.button("🔄 Reset File Uploader", key="reset_uploader_btn", help="Click to reset the file uploader if it's not showing properly"):
            if uploader_key in st.session_state:
                del st.session_state[uploader_key]
            st.rerun()
        
        # Show uploaded file info if file is uploaded (but not if we just processed it)
        if uploaded_file is not None and not just_processed:
            st.info(f"📄 **File uploaded:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            # YAML selector (only show if YAML files exist and file is uploaded)
            selected_yaml = None
            if yaml_files:
                yaml_options = ["None"] + [y['filename'] for y in yaml_files]
                selected_yaml_name = st.selectbox(
                    "Attach YAML Config (optional)",
                    options=yaml_options,
                    help="Select a YAML config file to attach to this training file. Files with the same YAML will be grouped into one training job.",
                    key="yaml_selector"
                )
                if selected_yaml_name != "None":
                    selected_yaml = selected_yaml_name
            else:
                st.info("💡 Upload YAML config files above to attach them to training files.")
            
            # Process and Queue button
            if st.button("✅ Process and Queue", key="process_and_queue", type="primary"):
                try:
                    # Save file to queue directory
                    file_path = queue_dir / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Create metadata file
                    metadata = {
                        "filename": uploaded_file.name,
                        "file_type": uploaded_file.name.split('.')[-1].lower(),
                        "date": datetime.now().isoformat(),
                        "model": model_name,
                        "size": uploaded_file.size,
                        "queued": True,
                        "attached_yaml": selected_yaml if selected_yaml else None
                    }
                    metadata_path = queue_dir / f"{Path(uploaded_file.name).stem}_metadata.json"
                    # Write metadata with explicit encoding and ensure it's flushed
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                        f.flush()  # Ensure data is written to disk
                        import os
                        if hasattr(f, 'fileno'):
                            os.fsync(f.fileno())  # Force write to disk
                    
                    # Verify the file was written correctly by reading it back
                    try:
                        import json as json_module_verify  # Use alias to avoid scoping issues
                        with open(metadata_path, 'r', encoding='utf-8') as verify_f:
                            verify_metadata = json_module_verify.load(verify_f)
                            verify_yaml = verify_metadata.get('attached_yaml')
                            if verify_yaml != (selected_yaml if selected_yaml else None):
                                st.warning(f"⚠️ Warning: Metadata verification failed. Expected: {selected_yaml}, Got: {verify_yaml}")
                    except Exception as verify_error:
                        st.warning(f"⚠️ Could not verify metadata file: {verify_error}")
                    
                    yaml_msg = f" with YAML config '{selected_yaml}'" if selected_yaml else ""
                    st.success(f"✅ File '{uploaded_file.name}' queued successfully{yaml_msg}!")
                    
                    # Clear the uploader and selector to remove file from drag-and-drop section
                    # This will reset the file uploader widget
                    if "training_file_uploader" in st.session_state:
                        del st.session_state["training_file_uploader"]
                    if "yaml_selector" in st.session_state:
                        del st.session_state["yaml_selector"]  # Delete to reset on next rerun
                    
                    # Set flag to hide file info on next rerun
                    st.session_state["training_file_just_processed"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        # Show queued files
        st.markdown("---")
        st.markdown("#### Queued Files")
        
        queued_files = []
        if queue_dir.exists():
            # Get files from metadata (exclude YAML files)
            import json as json_module_tab2_list  # Use alias to avoid scoping issues
            for metadata_file in queue_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json_module_tab2_list.load(f)
                        if metadata.get("model") == model_name:
                            # Skip YAML files - they're shown in the YAML section only
                            if metadata.get("is_yaml", False):
                                continue
                            filename = metadata.get("filename")
                            if filename:
                                # Also check file extension to be safe
                                file_ext = Path(filename).suffix.lower()
                                if file_ext in ['.yaml', '.yml']:
                                    continue
                                file_path = queue_dir / filename
                                if file_path.exists():
                                    # Create a fresh dict copy to preserve all fields
                                    file_metadata = dict(metadata)
                                    queued_files.append(file_metadata)
                except Exception as e:
                    pass
            
            # Also check for files without metadata (exclude YAML files)
            for file_path in queue_dir.iterdir():
                if file_path.is_file() and not file_path.name.endswith("_metadata.json"):
                    filename = file_path.name
                    # Skip YAML files
                    file_ext = file_path.suffix.lower()
                    if file_ext in ['.yaml', '.yml']:
                        continue
                    if not any(f.get("filename") == filename for f in queued_files):
                        file_size = file_path.stat().st_size
                        file_type = file_ext[1:] if file_ext.startswith('.') else file_ext
                        queued_files.append({
                            "filename": filename,
                            "file_type": file_type,
                            "date": "Unknown",
                            "model": model_name,
                            "size": file_size,
                            "queued": True
                        })
        
        # Filter out YAML files from training files display (double-check)
        training_files = [f for f in queued_files if not f.get('is_yaml', False) and Path(f.get('filename', '')).suffix.lower() not in ['.yaml', '.yml']]
        
        if training_files:
            st.info(f"📦 {len(training_files)} file(s) queued for training")
            for idx, file_meta in enumerate(training_files):
                filename = file_meta['filename']
                attached_yaml = file_meta.get('attached_yaml')
                with st.expander(f"📄 {filename}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        file_type = file_meta.get('file_type', 'unknown').upper()
                        st.write(f"**Date Queued:** {file_meta.get('date', 'Unknown')}")
                        st.write(f"**Type:** {file_type}")
                        st.write(f"**Size:** {file_meta.get('size', 0):,} bytes")
                        if attached_yaml:
                            st.write(f"**Attached YAML:** 📋 {attached_yaml}")
                            st.info(f"✅ This file will be grouped with other files using '{attached_yaml}' in one training job")
                        else:
                            st.info("✅ This file will be grouped with other files without YAML in one training job")
                    
                    with col2:
                        st.markdown("**Actions**")
                        delete_key = f"delete_queue_tab2_{idx}_{filename}"
                        confirm_key = f"confirm_delete_tab2_{idx}_{filename}"
                        cancel_key = f"cancel_delete_tab2_{idx}_{filename}"
                        state_key = f"delete_state_tab2_{idx}_{filename}"
                        
                        if state_key not in st.session_state:
                            st.session_state[state_key] = False
                        
                        if not st.session_state[state_key]:
                            if st.button("🗑️ Remove", key=delete_key, type="secondary"):
                                st.session_state[state_key] = True
                                st.rerun()
                        else:
                            st.warning(f"⚠️ Remove {filename}?")
                            col_confirm, col_cancel = st.columns(2)
                            with col_confirm:
                                if st.button("✅ Confirm", key=confirm_key, type="primary"):
                                    try:
                                        file_path = queue_dir / filename
                                        if file_path.exists():
                                            file_path.unlink()
                                        metadata_path = queue_dir / f"{Path(filename).stem}_metadata.json"
                                        if metadata_path.exists():
                                            metadata_path.unlink()
                                        st.success(f"✅ Removed '{filename}' from queue")
                                        if state_key in st.session_state:
                                            del st.session_state[state_key]
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                            with col_cancel:
                                if st.button("❌ Cancel", key=cancel_key):
                                    if state_key in st.session_state:
                                        del st.session_state[state_key]
                                    st.rerun()
        else:
            st.info("No files queued yet. Upload JSON, JSONL, or TXT files above to queue them for training.")
    
    # Tab 3: Interact (formerly Learning Session)
    with tab3:
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Get base model and check for available versions
        from utils.model_manager import ModelManager
        model_manager = ModelManager()
        metadata = model_manager.get_model_metadata(model_name)
        base_model = metadata.get("base_model", "llama2") if metadata else "llama2"
        
        # Get available versions
        available_versions = model_manager.list_available_versions(model_name)
        most_recent_version = model_manager.get_most_recent_version(model_name)
        
        # Initialize version selection in session state
        version_key = f"selected_version_{model_name}"
        if version_key not in st.session_state:
            # Default to most recent version, or "base" if no versions
            st.session_state[version_key] = most_recent_version if most_recent_version else "base"
        
        # Version selection dropdown
        version_options = ["base"] + [f"V{v}" for v in available_versions]
        version_labels = ["Base (no weights)"] + [f"V{v}" for v in available_versions]
        
        # Find current selection index
        current_selection = st.session_state[version_key]
        if isinstance(current_selection, int):
            current_index = available_versions.index(current_selection) + 1 if current_selection in available_versions else 0
        else:
            current_index = 0
        
        selected_version_label = st.selectbox(
            "Select Model Version:",
            options=version_labels,
            index=current_index,
            key=f"version_select_{model_name}",
            help="Choose which version of weights to use. Defaults to most recent version."
        )
        
        # Update session state based on selection
        if selected_version_label == "Base (no weights)":
            selected_version = "base"
        else:
            # Extract version number from "V1", "V2", etc.
            selected_version = int(selected_version_label[1:])
        
        st.session_state[version_key] = selected_version
        
        # Initialize Ollama client for status checks (always needed)
        ollama_client = OllamaClient(base_model)
        
        # Initialize client based on selected version
        client = None
        use_fine_tuned = False
        
        if selected_version != "base" and selected_version in available_versions:
            # Load weights from selected version
            weights_path = model_manager.get_version_weights_path(model_name, selected_version)
            if weights_path and weights_path.exists():
                try:
                    from utils.fine_tuned_client import FineTunedModelClient
                    # Get token from session state if available
                    hf_token = st.session_state.get('hf_token', '')
                    
                    # Get base model name from version metadata
                    version_metadata = model_manager.get_version_metadata(model_name, selected_version)
                    hf_model_name = None
                    if version_metadata:
                        # Axolotl uses "hf_model", old system uses "hf_model_name"
                        hf_model_name = version_metadata.get("hf_model") or version_metadata.get("hf_model_name")
                    
                    # If not in version metadata, try to get from axolotl prep
                    if not hf_model_name:
                        from utils.axolotl_prep import AxolotlDataPrep
                        axolotl_prep = AxolotlDataPrep(model_name)
                        hf_model_name = axolotl_prep.get_hf_model_name(base_model)
                    
                    # FineTunedModelClient expects fine_tune_metadata.json with hf_model_name
                    # Create compatibility file if it doesn't exist (for Axolotl-trained models)
                    weights_dir = weights_path.parent if weights_path.name == "adapter" else weights_path
                    fine_tune_metadata_path = weights_dir / "fine_tune_metadata.json"
                    if not fine_tune_metadata_path.exists() and hf_model_name:
                        # Create compatibility metadata file
                        compatibility_metadata = {
                            "use_lora": True,
                            "hf_model_name": hf_model_name,
                            "base_model": base_model
                        }
                        with open(fine_tune_metadata_path, 'w') as f:
                            json.dump(compatibility_metadata, f, indent=2)
                    
                    # FineTunedModelClient expects the weights directory (parent of adapter/)
                    # If weights_path is the adapter directory, use its parent
                    model_path_for_client = weights_dir
                    
                    client = FineTunedModelClient(str(model_path_for_client), hf_token=hf_token if hf_token else None)
                    use_fine_tuned = True
                    st.success(f"✅ Using weights from {selected_version_label}")
                except Exception as e:
                    st.warning(f"⚠️ Failed to load weights from {selected_version_label}: {str(e)}")
                    st.info("Falling back to base Ollama model...")
                    use_fine_tuned = False
            else:
                st.warning(f"⚠️ Weights not found for {selected_version_label}")
                st.info("Falling back to base Ollama model...")
                use_fine_tuned = False
        
        if not use_fine_tuned:
            client = ollama_client
        
        data_manager = TrainingDataManager(model_name)
        
        # Check all statuses (for validation and inline display)
        if not use_fine_tuned:
            ollama_running = False
            try:
                import subprocess
                result = subprocess.run(['ollama', 'list'], capture_output=True, timeout=5)
                ollama_running = result.returncode == 0
            except:
                ollama_running = False
        
        available_models = ollama_client.get_available_models()
        model_exists = ollama_client.model_exists()
        
        try:
            import ollama
            python_lib_ok = True
        except ImportError:
            python_lib_ok = False
        
        http_api_ok = False
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            http_api_ok = response.status_code == 200
        except:
            http_api_ok = False
        else:
            # Fine-tuned model - no Ollama needed
            ollama_running = True
            model_exists = True
            python_lib_ok = True
            http_api_ok = True
            available_models = []
        
        
        # Create status icons HTML
        status_icons_html = []
        if use_fine_tuned:
            status_icons_html.append(f'<span title="Fine-Tuned Model: Loaded" style="font-size: 0.7em; cursor: help; margin: 0 1px;">🟢</span>')
        else:
            status_icons_html.append(f'<span title="Ollama Server: {"Running" if ollama_running else "Not Running"}" style="font-size: 0.7em; cursor: help; margin: 0 1px;">{"🟢" if ollama_running else "🔴"}</span>')
        status_icons_html.append(f'<span title="Model {base_model}: {"Available" if model_exists else "Missing"}" style="font-size: 0.7em; cursor: help; margin: 0 1px;">{"🟢" if model_exists else "🔴"}</span>')
        status_html = " ".join(status_icons_html)
        
        st.markdown(f'### Interact {status_html}', unsafe_allow_html=True)
        
        # Action buttons row
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🗑️ Clear Chat", key="clear_chat_btn", help="Clear chat history and start fresh"):
                st.session_state.chat_history = []
                st.success("Chat cleared!")
                st.rerun()
        
        st.markdown("Chat with your model")
        
        # Show detailed status if there are issues (only for Ollama)
        if not use_fine_tuned and (not ollama_running or not model_exists):
            st.markdown("---")
            with st.expander("🔧 Status Details & Fixes", expanded=True):
                if not ollama_running:
                    st.error("**Issue:** Ollama server is not running")
                    st.markdown("""
                    **To fix:**
                    1. Make sure Ollama is installed
                    2. Start Ollama server (usually runs automatically)
                    3. Check with: `ollama list` in terminal
                    """)
                
                if not model_exists:
                    st.error(f"**Issue:** Model '{base_model}' is not available")
                    st.markdown(f"**To fix:**")
                    st.code(f"ollama pull {base_model}", language="bash")
                    
                    if available_models:
                        st.info(f"**Available models:** {', '.join(available_models)}")
                        st.markdown(f"**Your profile uses:** `{base_model}`")
                        st.markdown("**Note:** Model names are case-insensitive. Make sure the base model matches one above.")
                    else:
                        st.warning("No models found. Install at least one model first.")
                
                st.markdown("---")
                st.markdown("**Debug Information:**")
                st.json({
                    "ollama_running": ollama_running,
                    "model_exists": model_exists,
                    "base_model": base_model,
                    "available_models": available_models,
                    "python_lib_ok": python_lib_ok,
                    "http_api_ok": http_api_ok
                })
            
            if not ollama_running or not model_exists:
                st.stop()
        
        # Ensure client is initialized
        if client is None:
            st.error("❌ Failed to initialize model client. Please check your model configuration.")
            st.stop()
        
        # Prepend text configuration (collapsible)
        with st.expander("⚙️ Prompt Settings", expanded=False):
            prepend_key = f"prepend_text_{model_name}"
            summary_key = f"include_summary_{model_name}"
            
            # Load saved preferences if not already in session state
            if prepend_key not in st.session_state or summary_key not in st.session_state:
                preferences = get_model_preferences(model_name)
            if prepend_key not in st.session_state:
                st.session_state[prepend_key] = preferences.get("prepend_text", "")
            if summary_key not in st.session_state:
                st.session_state[summary_key] = preferences.get("include_summary", False)
            
            prepend_text = st.text_area(
                "Prepend Text (added invisibly to all prompts)",
                value=st.session_state[prepend_key],
                help="This text will be prepended to all prompts before sending to the model. It's not visible in the chat but affects all responses.",
                key=f"prepend_textarea_{model_name}",
                height=100
            )
            
            include_summary = st.checkbox(
                "Include conversation summary request",
                value=st.session_state[summary_key],
                help="If checked, the model will be asked to attach a summary of the conversation at the end of each response, marked with ###SUMMARY###",
                key=f"include_summary_checkbox_{model_name}"
            )
            
            # Save preferences whenever they change
            # Sync widget values to our session state keys and save to file
            textarea_key = f"prepend_textarea_{model_name}"
            checkbox_key = f"include_summary_checkbox_{model_name}"
            
            # Sync widget session state to our preference keys
            if textarea_key in st.session_state:
                st.session_state[prepend_key] = st.session_state[textarea_key]
            if checkbox_key in st.session_state:
                st.session_state[summary_key] = st.session_state[checkbox_key]
            
            # Check if current values differ from saved preferences and save if needed
            preferences = get_model_preferences(model_name)
            needs_save = False
            
            # Use widget return values (they reflect current state)
            if prepend_text != preferences.get("prepend_text", ""):
                preferences["prepend_text"] = prepend_text
                st.session_state[prepend_key] = prepend_text
                needs_save = True
            
            if include_summary != preferences.get("include_summary", False):
                preferences["include_summary"] = include_summary
                st.session_state[summary_key] = include_summary
                needs_save = True
            
            if needs_save:
                save_model_preferences(model_name, preferences)
        
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Type your message...", key="learning_session_chat_input")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Add temporary "Thinking..." message
            st.session_state.chat_history.append({"role": "assistant", "content": "🤔 Thinking..."})
            st.rerun()
            return
        
        # Check if last message is "Thinking..." and process it
        if (st.session_state.chat_history and 
            st.session_state.chat_history[-1].get("content") == "🤔 Thinking..." and
            len(st.session_state.chat_history) >= 2 and
            st.session_state.chat_history[-2].get("role") == "user"):
            
            # Get the user's last message
            user_input = st.session_state.chat_history[-2]["content"]
            
            # Prepare messages for Ollama (no rule injection)
            messages = []
            
            # Get prepend text and summary setting
            prepend_key = f"prepend_text_{model_name}"
            prepend_text = st.session_state.get(prepend_key, "")
            include_summary = st.session_state.get(f"include_summary_{model_name}", False)
            
            for idx, msg in enumerate(st.session_state.chat_history[:-1]):  # Exclude the "Thinking..." message
                # Ensure role is valid (user or assistant)
                role = msg.get("role", "user")
                if role not in ["user", "assistant", "system"]:
                    role = "user"  # Default to user if invalid
                
                content = str(msg.get("content", ""))
                
                # For user messages, prepend the prepend text (invisibly) and add summary request if enabled
                if role == "user":
                    # Build the full content with prepend and summary request
                    full_content_parts = []
                    
                    # Add prepend text first (if exists)
                    if prepend_text:
                        full_content_parts.append(prepend_text)
                    
                    # Add the original user message
                    full_content_parts.append(content)
                    
                    # Add summary request if enabled (after prepend and user message)
                    if include_summary:
                        # Create a summary of the conversation so far (up to this message)
                        conversation_summary = ""
                        for prev_msg in st.session_state.chat_history[:idx]:  # All messages before this one
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
            
            # Remove the "Thinking..." message
            st.session_state.chat_history.pop()
            
            # Get response from model
            try:
                if use_fine_tuned:
                    # Fine-tuned models
                    response = client.chat(messages, max_length=1024, temperature=0.7)
                    if response and not (isinstance(response, str) and response.startswith("Error:")):
                        # Strip summary section if present
                        if "###SUMMARY###" in response:
                            parts = response.split("###SUMMARY###")
                            main_response = parts[0].strip()
                            # Store the summary separately if needed (for future use)
                            summary = parts[1].strip() if len(parts) > 1 else ""
                            response = main_response
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        data_manager.log_conversation_turn(user_input, response)
                    else:
                        st.error(response if response else "Error: Received empty response from model.")
                    st.rerun()
                else:
                    # Ollama models
                    try:
                        response = client.chat(messages, stream=False, timeout=120)
                        
                        # Check if response is valid
                        if response and not (isinstance(response, str) and response.startswith("Error:")):
                            # Strip summary section if present
                            if "###SUMMARY###" in response:
                                parts = response.split("###SUMMARY###")
                                main_response = parts[0].strip()
                                # Store the summary separately if needed (for future use)
                                summary = parts[1].strip() if len(parts) > 1 else ""
                                response = main_response
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            data_manager.log_conversation_turn(user_input, response)
                        else:
                            st.error(response if response else "Error: Received empty response from model.")
                            # Remove the user message since we failed
                            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                                st.session_state.chat_history.pop()
                        st.rerun()
                    except Exception as chat_error:
                        if "timeout" in str(chat_error).lower():
                            st.error("⏱️ **Request timed out after 120 seconds.**")
                            st.warning("**Ollama is responding very slowly.** Suggestions:")
                            st.markdown(f"""
                            - **Pre-load the model:** Run `ollama run {base_model}` in terminal first
                            - **Use a smaller/faster model:** Consider using a smaller model
                            - **Check system resources:** Make sure you have enough RAM/CPU
                            - **Restart Ollama:** Sometimes restarting helps: `pkill ollama && ollama serve`
                            """)
                        else:
                            st.error(f"Error: {str(chat_error)}")
                        # Remove the user message since we failed
                        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                            st.session_state.chat_history.pop()
                        st.rerun()
                        return
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                # Remove the user message since we failed
                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                    st.session_state.chat_history.pop()
                st.rerun()
    


