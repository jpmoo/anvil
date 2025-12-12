"""Model Configuration page"""

import streamlit as st
import json
import requests
from pathlib import Path
from datetime import datetime
from utils.training_data import TrainingDataManager
from utils.fine_tuner import FineTuner
from utils.model_status import ModelStatus
from utils.ollama_client import OllamaClient
from utils.config import get_model_context_dir, get_model_behavioral_path


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


def render():
    """Render Model Configuration interface"""
    
    if not st.session_state.selected_model:
        st.warning("⚠️ Please select a model from the sidebar first")
        return
    
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
        
        st.markdown("---")
        st.caption("Find your API key in your [Vast.ai account settings](https://vast.ai/account)")
    
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
            for metadata_file in queue_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        if metadata.get("model") == model_name:
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
                    pass
            
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
        
        # Get marked conversation pairs
        learned_pairs = data_manager.get_learned_pairs()
        
        # Display summary
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
            st.markdown("#### 💬 Marked Conversation Pairs")
            if learned_pairs:
                st.metric("Pairs Marked", len(learned_pairs))
                total_chars = sum(
                    len(p.get("question", "")) + len(p.get("answer", ""))
                    for p in learned_pairs
                )
                st.caption(f"Total content: {total_chars:,} characters")
                
                # Show pairs list
                with st.expander(f"View {len(learned_pairs)} pair(s)", expanded=False):
                    for i, pair in enumerate(learned_pairs, 1):
                        question = pair.get("question", "")[:100]
                        if len(pair.get("question", "")) > 100:
                            question += "..."
                        st.write(f"**Pair {i}:** {question}")
            else:
                st.info("No pairs marked yet")
                st.caption("Mark Q&A pairs in Tab 3 to include them in training")
        
        # Overall status
        has_training_data = len(queued_files) > 0 or len(learned_pairs) > 0
        if has_training_data:
            total_examples = len(queued_files) + len(learned_pairs)
            st.success(f"✅ Ready to train with {total_examples} data source(s): {len(queued_files)} file(s) + {len(learned_pairs)} pair(s)")
        else:
            st.warning("⚠️ No training data available. Add files in Tab 2 and/or mark pairs in Tab 3 before training.")
        
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
        if (len(queued_files) > 0 or len(learned_pairs) > 0) and not has_active_jobs:
            st.markdown("### 🚀 Launch Training")
            
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
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gpu_name = st.text_input(
                    "GPU Name (optional)",
                    value="",
                    help="Preferred GPU (e.g., RTX 3090, A100). Leave empty for any GPU.",
                    key="gpu_name_input"
                )
                min_gpu_ram = st.number_input(
                    "Min GPU RAM (GB)",
                    min_value=8,
                    max_value=128,
                    value=24,
                    help="Minimum GPU RAM per GPU required",
                    key="min_gpu_ram_input"
                )
                num_gpus = st.number_input(
                    "Number of GPUs",
                    min_value=1,
                    max_value=8,
                    value=1,
                    help="Number of GPUs required",
                    key="num_gpus_input"
                )
            
            with col2:
                max_price = st.number_input(
                    "Max Price/Hour ($)",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.5,
                    step=0.1,
                    help="Maximum price per hour",
                    key="max_price_input"
                )
                disk_space = st.number_input(
                    "Disk Space (GB)",
                    min_value=50,
                    max_value=500,
                    value=100,
                    help="Disk space needed",
                    key="disk_space_input"
                )
            
            with col3:
                epochs = st.number_input(
                "Epochs", 
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Number of training epochs",
                    key="epochs_input"
                )
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=2e-4,
                    format="%.6f",
                    help="Learning rate for training",
                    key="learning_rate_input"
                )
            
            if st.button("🚀 Launch Training Job", key="launch_training", type="primary"):
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
                            for metadata_file in queue_dir.glob("*_metadata.json"):
                                try:
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
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
                                        job_info = training_manager.launch_training_job(
                                            gpu_name=gpu_name if gpu_name else None,
                                            min_gpu_ram=min_gpu_ram,
                                            max_price=max_price if max_price > 0 else None,
                                            disk_space=disk_space,
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            hf_model_override=hf_model_override.strip() if hf_model_override and hf_model_override.strip() else None,
                                            num_gpus=num_gpus if num_gpus > 0 else None,
                                            job_queue=job_queue  # Pass entire job queue
                                        )
                                        
                                        success_msg = f"✅ Launched training instance with {len(job_queue)} job(s) queued:\n"
                                        for idx, queue_item in enumerate(job_queue, 1):
                                            yaml_desc = f" (YAML: {queue_item['yaml_filename']})" if queue_item['yaml_filename'] else " (no YAML)"
                                            success_msg += f"  • Job {idx}/{len(job_queue)}: {queue_item['file_count']} file(s){yaml_desc}\n"
                                        st.success(success_msg)
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
                
                # Show queue info for single job if it has a queue
                job_queue = active_job.get("job_queue")
                if job_queue and len(job_queue) > 1:
                    current_job_index = active_job.get("current_job_index", 0)
                    st.info(f"📋 Processing {len(job_queue)} job(s) sequentially on this instance (currently job {current_job_index + 1}/{len(job_queue)})")
        
        if has_active_jobs and active_job:
            st.markdown("### 📊 Active Training Job")
            
            try:
                from utils.vast_training_manager import VastTrainingManager
                training_manager = VastTrainingManager(model_name, vast_api_key)
                
                # Initialize phase tracking in session state
                phase_key = f"training_phase_{active_job.get('instance_id')}"
                terminal_output_key = f"terminal_output_{active_job.get('instance_id')}"
                
                if phase_key not in st.session_state:
                    # Determine initial phase based on job status
                    job_status = active_job.get('status', 'unknown')
                    if job_status == 'launching':
                        st.session_state[phase_key] = 1  # Starting instance
                    elif job_status == 'running' and not active_job.get('files_uploaded'):
                        st.session_state[phase_key] = 2  # Upload file
                    elif job_status == 'running' and active_job.get('files_uploaded'):
                        st.session_state[phase_key] = 3  # Do training
                    elif job_status == 'completed':
                        st.session_state[phase_key] = 4  # Finalize
                    else:
                        st.session_state[phase_key] = 1  # Default to phase 1
                
                if terminal_output_key not in st.session_state:
                    st.session_state[terminal_output_key] = []
                
                current_phase = st.session_state[phase_key]
                terminal_output = st.session_state[terminal_output_key]
                
                # Phase definitions
                phases = {
                    1: {"name": "Starting Instance", "icon": "🚀", "description": "Launching Vast.ai instance and waiting for it to be ready"},
                    2: {"name": "Upload File", "icon": "📤", "description": "Uploading training files to the instance via SSH/SCP"},
                    3: {"name": "Do Training", "icon": "⚙️", "description": "Monitoring training progress"},
                    4: {"name": "Finalize", "icon": "✅", "description": "Downloading weights, destroying instance, and cleaning up"}
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
                
                # Phase 1: Starting Instance
                if current_phase == 1:
                    # Ensure datetime is available in this scope
                    from datetime import datetime
                    
                    st.markdown(f"### {phases[1]['icon']} Phase 1: {phases[1]['name']}")
                    st.caption(phases[1]['description'])
                    st.info("💡 **Instructions:** Click 'Check Status' to check if the instance is ready. If the instance is ready (status is 'running' and SSH is available), it will automatically advance to Phase 2.")
                    
                    # Terminal output area (scrollable)
                    st.markdown("#### Terminal Output")
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
                            st.info("No output yet. Click 'Check Status' to start monitoring.")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("🔄 Check Status", key="check_instance_status"):
                            try:
                                instance_id = active_job.get("instance_id")
                                
                                # Add API call output
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Checking instance status...")
                                terminal_output.append(f"[API] GET /instances/{instance_id}/")
                                
                                try:
                                    job_status = training_manager.get_job_status(instance_id)
                                    active_job.update(job_status)
                                    training_manager._save_job(active_job)
                                    
                                    status = job_status.get('status', 'unknown')
                                    terminal_output.append(f"[API] Response: status = {status}")
                                    
                                    # Show Vast.ai actual status if available
                                    vast_status = job_status.get("vast_status", {})
                                    if vast_status:
                                        if "instances" in vast_status and isinstance(vast_status["instances"], dict):
                                            instance_data = vast_status["instances"]
                                        else:
                                            instance_data = vast_status
                                        
                                        actual_status = instance_data.get("actual_status", "unknown")
                                        terminal_output.append(f"[API] Vast.ai actual_status: {actual_status}")
                                        
                                        # Check if instance is ready (has SSH info AND status is running)
                                        if job_status.get("ssh_host") and actual_status == "running":
                                            ssh_host = job_status.get("ssh_host")
                                            ssh_port = job_status.get("ssh_port", 22)
                                            terminal_output.append(f"[SSH] Instance ready: {ssh_host}:{ssh_port}")
                                            terminal_output.append(f"[SUCCESS] Phase 1 complete! Instance is ready.")
                                            
                                            # Auto-advance to phase 2
                                            # Don't clear terminal - Phase 2 will add its initialization messages
                                            st.session_state[phase_key] = 2
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.rerun()
                                        else:
                                            if job_status.get("ssh_host") and actual_status != "running":
                                                terminal_output.append(f"[INFO] SSH available but instance status is '{actual_status}' (waiting for 'running')...")
                                            else:
                                                terminal_output.append(f"[INFO] Instance status: {actual_status}")
                                            terminal_output.append(f"[INFO] Waiting for instance to be ready...")
                                    else:
                                        terminal_output.append(f"[INFO] Waiting for instance to be ready...")
                                    
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.rerun()
                                except Exception as api_error:
                                    error_msg = str(api_error)
                                    terminal_output.append(f"[API] Error: {error_msg}")
                                    if "429" in error_msg or "RATE_LIMIT" in error_msg:
                                        terminal_output.append(f"[INFO] Rate limited - please wait before refreshing")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.rerun()
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] {error_msg}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                    
                    with col2:
                        if st.button("🔄 Redo Phase", key="retry_phase_1"):
                            try:
                                # Clear terminal before redoing phase
                                st.session_state[terminal_output_key] = []
                                terminal_output = []
                                
                                instance_id = active_job.get("instance_id")
                                if instance_id:
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Destroying current instance...")
                                    terminal_output.append(f"[API] DELETE /instances/{instance_id}/")
                                    
                                    # Destroy current instance
                                    destroyed = training_manager.vast_client.destroy_instance(instance_id)
                                    if destroyed:
                                        terminal_output.append(f"[API] Instance destroyed successfully")
                                    else:
                                        terminal_output.append(f"[WARNING] Instance destruction returned False")
                                    
                                    # Get launch parameters from active_job
                                    package_info = active_job.get("package_info", {})
                                    gpu_name = active_job.get("gpu_name", "")
                                    min_gpu_ram = active_job.get("min_gpu_ram", 24)
                                    max_price = active_job.get("max_price", 0.50)
                                    disk_space = active_job.get("disk_space", 100)
                                    epochs = package_info.get("config", {}).get("num_epochs", 3) if package_info else 3
                                    learning_rate = package_info.get("config", {}).get("learning_rate", 2e-4) if package_info else 2e-4
                                    hf_model_override = active_job.get("hf_model_override", "")
                                    num_gpus = active_job.get("num_gpus")
                                    
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Launching new instance...")
                                    terminal_output.append(f"[API] Searching for offers and creating instance...")
                                    
                                    # Launch new instance (will search for offers and create instance)
                                    new_job = training_manager.launch_training_job(
                                        gpu_name=gpu_name if gpu_name else None,
                                        min_gpu_ram=min_gpu_ram,
                                        max_price=max_price,
                                        disk_space=disk_space,
                                        epochs=epochs,
                                        learning_rate=learning_rate,
                                        hf_model_override=hf_model_override if hf_model_override else None,
                                        num_gpus=num_gpus
                                    )
                                    
                                    # Update active_job with new instance_id
                                    new_instance_id = new_job.get("instance_id")
                                    if new_instance_id:
                                        active_job["instance_id"] = new_instance_id
                                        active_job["status"] = "launching"
                                        active_job["created_at"] = datetime.now().isoformat()
                                        # Clear any previous phase completion flags
                                        active_job.pop("files_uploaded", None)
                                        training_manager._save_job(active_job)
                                        
                                        terminal_output.append(f"[SUCCESS] New instance launched: {new_instance_id}")
                                        terminal_output.append(f"[INFO] Phase 1 restarted with new instance")
                                        
                                        # Update phase key to use new instance_id
                                        new_phase_key = f"training_phase_{new_instance_id}"
                                        new_terminal_key = f"terminal_output_{new_instance_id}"
                                        st.session_state[new_phase_key] = 1
                                        st.session_state[new_terminal_key] = terminal_output
                                        
                                        st.success("✅ Instance destroyed and new instance launched!")
                                        st.rerun()
                                    else:
                                        terminal_output.append(f"[ERROR] Failed to get new instance ID")
                                        st.error("Failed to launch new instance")
                                else:
                                    terminal_output.append(f"[ERROR] No instance ID found in job")
                                    st.error("No instance ID found")
                                
                                st.session_state[terminal_output_key] = terminal_output
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] Failed to redo phase: {error_msg}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                    
                    with col3:
                        # Check if instance is ready (get fresh status if needed)
                        try:
                            instance_id = active_job.get("instance_id")
                            if instance_id:
                                job_status_check = training_manager.get_job_status(instance_id)
                                vast_status_check = job_status_check.get("vast_status", {})
                                if vast_status_check:
                                    if "instances" in vast_status_check and isinstance(vast_status_check["instances"], dict):
                                        instance_data_check = vast_status_check["instances"]
                                    else:
                                        instance_data_check = vast_status_check
                                    actual_status_check = instance_data_check.get("actual_status", "unknown")
                                    
                                    if job_status_check.get("ssh_host") and actual_status_check == "running":
                                        st.success("✅ Instance is ready! Click 'Next Phase' to continue.")
                                        if st.button("➡️ Next Phase", key="next_phase_1", type="primary"):
                                            st.session_state[phase_key] = 2
                                            # Don't clear terminal - Phase 2 will add its initialization messages
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.rerun()
                        except:
                            pass  # If we can't check, just don't show the button
                    
                    # Cancel Job button (always visible) with confirmation
                    st.markdown("---")
                    cancel_confirm_key = f"cancel_confirm_{active_job.get('instance_id')}"
                    
                    if cancel_confirm_key not in st.session_state:
                        st.session_state[cancel_confirm_key] = False
                    
                    if not st.session_state[cancel_confirm_key]:
                        if st.button("❌ Cancel Job", key="cancel_job_phase_1", type="secondary"):
                            st.session_state[cancel_confirm_key] = True
                            st.rerun()
                    else:
                        st.warning("⚠️ Are you sure you want to cancel this job? This will destroy the Vast.ai instance and cannot be undone.")
                        col_confirm, col_cancel = st.columns([1, 1])
                        with col_confirm:
                            if st.button("✅ Confirm Cancel", key="confirm_cancel_phase_1", type="primary"):
                                try:
                                    # Ensure datetime is available
                                    from datetime import datetime
                                    instance_id = active_job.get("instance_id")
                                    if instance_id:
                                        terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Cancelling job and destroying instance...")
                                        terminal_output.append(f"[API] DELETE /instances/{instance_id}/")
                                        
                                        # Destroy instance
                                        destroyed = training_manager.vast_client.destroy_instance(instance_id)
                                        if destroyed:
                                            terminal_output.append(f"[API] Instance destroyed successfully")
                                            
                                            # Mark job as cancelled
                                            active_job["status"] = "cancelled"
                                            active_job["cancelled_at"] = datetime.now().isoformat()
                                            training_manager._save_job(active_job)
                                            
                                            terminal_output.append(f"[SUCCESS] Job cancelled successfully")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.session_state[cancel_confirm_key] = False
                                            st.success("✅ Job cancelled and instance destroyed.")
                                            st.rerun()
                                        else:
                                            terminal_output.append(f"[WARNING] Instance destruction returned False")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.warning("Instance destruction may have failed. Please check Vast.ai dashboard.")
                                except Exception as e:
                                    error_msg = str(e)
                                    terminal_output.append(f"[ERROR] Failed to cancel job: {error_msg}")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.error(f"Error cancelling job: {error_msg}")
                                finally:
                                    st.session_state[cancel_confirm_key] = False
                        with col_cancel:
                            if st.button("❌ Cancel", key="cancel_confirm_phase_1"):
                                st.session_state[cancel_confirm_key] = False
                                st.rerun()
                
                # Phase 2: Upload File
                elif current_phase == 2:
                    # Ensure datetime is available in this scope
                    from datetime import datetime
                    
                    st.markdown(f"### {phases[2]['icon']} Phase 2: {phases[2]['name']}")
                    st.caption(phases[2]['description'])
                    
                    # Initialize Phase 2: Check SSH and create directories (show in terminal, don't auto-upload)
                    phase2_init_key = f"phase2_init_{active_job.get('instance_id')}"
                    if not active_job.get("files_uploaded", False) and phase2_init_key not in st.session_state:
                        try:
                            instance_id = active_job.get("instance_id")
                            if instance_id:
                                # Get SSH info
                                job_status = training_manager.get_job_status(instance_id)
                                ssh_host = job_status.get("ssh_host")
                                ssh_port = job_status.get("ssh_port", 22)
                                
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
                                    
                                    # Create directories
                                    terminal_output.append(f"[SSH] Creating directories on remote instance...")
                                    mkdir_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "mkdir -p /workspace/data && mkdir -p /workspace/output/training && echo 'Directories ready'"
                                    ]
                                    mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=15)
                                    if mkdir_result.returncode == 0:
                                        terminal_output.append(f"[SSH] Directories created successfully")
                                        terminal_output.append(f"[SUCCESS] File structure ready. Click 'Upload Files' button to proceed.")
                                    else:
                                        stderr_filtered = filter_malloc_warnings(mkdir_result.stderr)
                                        terminal_output.append(f"[SSH] Directory creation: {stderr_filtered[:200]}")
                                    
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
                        # Only show upload button if files haven't been uploaded yet
                        files_already_uploaded = active_job.get("files_uploaded", False)
                        if not files_already_uploaded:
                            if st.button("📤 Upload Files", key="upload_files"):
                                try:
                                    instance_id = active_job.get("instance_id")
                                    package_info = active_job.get("package_info")
                                    
                                    if not package_info:
                                        st.error("Package info not found. Please restart training.")
                                    else:
                                        terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting file upload...")
                                        terminal_output.append(f"[INFO] Dataset: {package_info.get('dataset_path', 'N/A')}")
                                        terminal_output.append(f"[INFO] Config: {package_info.get('config_path', 'N/A')}")
                                        
                                        # Get SSH info
                                        job_status = training_manager.get_job_status(instance_id)
                                        ssh_host = job_status.get("ssh_host")
                                        ssh_port = job_status.get("ssh_port", 22)
                                        
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
                                                # Directories don't exist, create them
                                                terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                                terminal_output.append(f"[SSH] Creating directories on remote instance...")
                                                mkdir_cmd = [
                                                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                                    f"root@{ssh_host}",
                                                    "mkdir -p /workspace/data && mkdir -p /workspace/output/training && echo 'Directories ready'"
                                                ]
                                                mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=15)
                                                if mkdir_result.returncode == 0:
                                                    terminal_output.append(f"[SSH] Directories created successfully")
                                                else:
                                                    stderr_filtered = filter_malloc_warnings(mkdir_result.stderr)
                                                    terminal_output.append(f"[SSH] Directory creation: {stderr_filtered[:200]}")
                                            else:
                                                # Directories already exist
                                                terminal_output.append(f"[SSH] Directories already exist, proceeding with upload...")
                                            
                                            # Upload config file FIRST (before training data)
                                            # This ensures the YAML config is in place before data files that reference it
                                            config_path = package_info.get('config_path')
                                            terminal_output.append(f"[SCP] Uploading config file first: {config_path}")
                                            scp_config_cmd = [
                                                "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                                                str(config_path),
                                                f"root@{ssh_host}:/workspace/data/axolotl_config.yaml"
                                            ]
                                            scp_config_result = subprocess.run(scp_config_cmd, capture_output=True, text=True, timeout=300)
                                            if scp_config_result.returncode == 0:
                                                terminal_output.append(f"[SCP] Config uploaded successfully")
                                            else:
                                                error_output = scp_config_result.stderr or scp_config_result.stdout
                                                # Filter MallocStackLogging warnings
                                                error_output = filter_malloc_warnings(error_output)
                                                if "Welcome to vast.ai" in error_output:
                                                    lines = error_output.split('\n')
                                                    actual_errors = [line for line in lines if line and 'Welcome to vast.ai' not in line and 'Have fun!' not in line]
                                                    error_output = '\n'.join(actual_errors) if actual_errors else error_output
                                                terminal_output.append(f"[SCP] Config upload error: {error_output[:300]}")
                                            
                                            # Upload training data file AFTER config
                                            dataset_path = package_info.get('dataset_path')
                                            terminal_output.append(f"[SCP] Uploading training data: {dataset_path}")
                                            scp_cmd = [
                                                "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                                                str(dataset_path),
                                                f"root@{ssh_host}:/workspace/data/training_data.jsonl"
                                            ]
                                            scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
                                            if scp_result.returncode == 0:
                                                terminal_output.append(f"[SCP] Training data uploaded successfully")
                                            else:
                                                error_output = scp_result.stderr or scp_result.stdout
                                                # Filter MallocStackLogging warnings
                                                error_output = filter_malloc_warnings(error_output)
                                                # Filter Vast.ai welcome message
                                                if "Welcome to vast.ai" in error_output:
                                                    lines = error_output.split('\n')
                                                    actual_errors = [line for line in lines if line and 'Welcome to vast.ai' not in line and 'Have fun!' not in line]
                                                    error_output = '\n'.join(actual_errors) if actual_errors else error_output
                                                terminal_output.append(f"[SCP] Upload error: {error_output[:300]}")
                                            
                                            # Check if both uploads succeeded
                                            if scp_result.returncode == 0 and scp_config_result.returncode == 0:
                                                terminal_output.append(f"[SUCCESS] All files uploaded successfully!")
                                                terminal_output.append(f"[INFO] Phase 2 complete! Ready to proceed to training.")
                                                active_job["files_uploaded"] = True
                                                training_manager._save_job(active_job)
                                                
                                                # Save terminal output before rerun
                                                st.session_state[terminal_output_key] = terminal_output
                                                
                                                # Don't auto-advance - let user click "Next Phase" button
                                                st.success("✅ Files uploaded successfully! Click 'Next Phase' to continue.")
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
                            # Files already uploaded - show status
                            st.info("✅ Files already uploaded")
                    
                    with col2:
                        if st.button("🔄 Redo Phase", key="retry_phase_2"):
                            try:
                                # Clear terminal before redoing phase
                                st.session_state[terminal_output_key] = []
                                terminal_output = []
                                
                                instance_id = active_job.get("instance_id")
                                if instance_id:
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Redoing Phase 2 - cleaning up uploaded files...")
                                    
                                    # Get SSH info
                                    job_status = training_manager.get_job_status(instance_id)
                                    ssh_host = job_status.get("ssh_host")
                                    ssh_port = job_status.get("ssh_port", 22)
                                    
                                    if ssh_host:
                                        terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                        import subprocess
                                        
                                        # Delete uploaded files and directories
                                        cleanup_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "rm -rf /workspace/data/* /workspace/output/training/* && echo 'Cleanup complete'"
                                        ]
                                        cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30)
                                        if cleanup_result.returncode == 0:
                                            terminal_output.append(f"[SSH] Files and directories deleted successfully")
                                        else:
                                            stderr_filtered = filter_malloc_warnings(cleanup_result.stderr)
                                            terminal_output.append(f"[SSH] Cleanup output: {stderr_filtered[:200]}")
                                        
                                        # Recreate directories
                                        mkdir_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                            f"root@{ssh_host}",
                                            "mkdir -p /workspace/data && mkdir -p /workspace/output/training && echo 'Directories recreated'"
                                        ]
                                        mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=15)
                                        if mkdir_result.returncode == 0:
                                            terminal_output.append(f"[SSH] Directories recreated successfully")
                                        else:
                                            stderr_filtered = filter_malloc_warnings(mkdir_result.stderr)
                                            terminal_output.append(f"[SSH] Directory recreation: {stderr_filtered[:200]}")
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
                    
                    # Cancel Job button (always visible) with confirmation
                    st.markdown("---")
                    cancel_confirm_key = f"cancel_confirm_{active_job.get('instance_id')}"
                    
                    if cancel_confirm_key not in st.session_state:
                        st.session_state[cancel_confirm_key] = False
                    
                    if not st.session_state[cancel_confirm_key]:
                        if st.button("❌ Cancel Job", key="cancel_job_phase_2", type="secondary"):
                            st.session_state[cancel_confirm_key] = True
                            st.rerun()
                    else:
                        st.warning("⚠️ Are you sure you want to cancel this job? This will destroy the Vast.ai instance and cannot be undone.")
                        col_confirm, col_cancel = st.columns([1, 1])
                        with col_confirm:
                            if st.button("✅ Confirm Cancel", key="confirm_cancel_phase_2", type="primary"):
                                try:
                                    # Ensure datetime is available
                                    from datetime import datetime
                                    instance_id = active_job.get("instance_id")
                                    if instance_id:
                                        terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Cancelling job and destroying instance...")
                                        terminal_output.append(f"[API] DELETE /instances/{instance_id}/")
                                        
                                        # Destroy instance
                                        destroyed = training_manager.vast_client.destroy_instance(instance_id)
                                        if destroyed:
                                            terminal_output.append(f"[API] Instance destroyed successfully")
                                            
                                            # Mark job as cancelled
                                            active_job["status"] = "cancelled"
                                            active_job["cancelled_at"] = datetime.now().isoformat()
                                            training_manager._save_job(active_job)
                                            
                                            terminal_output.append(f"[SUCCESS] Job cancelled successfully")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.session_state[cancel_confirm_key] = False
                                            st.success("✅ Job cancelled and instance destroyed.")
                                            st.rerun()
                                        else:
                                            terminal_output.append(f"[WARNING] Instance destruction returned False")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.warning("Instance destruction may have failed. Please check Vast.ai dashboard.")
                                except Exception as e:
                                    error_msg = str(e)
                                    terminal_output.append(f"[ERROR] Failed to cancel job: {error_msg}")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.error(f"Error cancelling job: {error_msg}")
                                finally:
                                    st.session_state[cancel_confirm_key] = False
                        with col_cancel:
                            if st.button("❌ Cancel", key="cancel_confirm_phase_2"):
                                st.session_state[cancel_confirm_key] = False
                                st.rerun()
                
                # Phase 3: Do Training
                elif current_phase == 3:
                    # Ensure datetime is available in this scope
                    from datetime import datetime
                    
                    st.markdown(f"### {phases[3]['icon']} Phase 3: {phases[3]['name']}")
                    st.caption(phases[3]['description'])
                    
                    # Show job queue status if multiple jobs
                    job_queue = active_job.get("job_queue")
                    current_job_index = active_job.get("current_job_index")
                    if job_queue and len(job_queue) > 1 and current_job_index is not None:
                        current_job_num = current_job_index + 1
                        total_jobs = len(job_queue)
                        current_job_info = job_queue[current_job_index]
                        yaml_desc = f" (YAML: {current_job_info.get('yaml_filename')})" if current_job_info.get('yaml_filename') else " (no YAML)"
                        st.info(f"📋 Processing job {current_job_num}/{total_jobs}: {current_job_info.get('file_count', 0)} file(s){yaml_desc}")
                    
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
                            job_status = training_manager.get_job_status(instance_id)
                            ssh_host = job_status.get("ssh_host")
                            ssh_port = job_status.get("ssh_port", 22)
                            
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
                            st.info("No output yet. Click 'Check Training Status' to start monitoring.")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("🔄 Check Training Status", key="check_training_status"):
                            try:
                                instance_id = active_job.get("instance_id")
                                
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Checking training status...")
                                
                                # Get SSH info
                                job_status = training_manager.get_job_status(instance_id)
                                ssh_host = job_status.get("ssh_host")
                                ssh_port = job_status.get("ssh_port", 22)
                                
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
                                    if "files_missing" in files_result.stdout:
                                        terminal_output.append(f"[WARNING] Training files not found in /workspace/data/")
                                    else:
                                        stdout_filtered = filter_malloc_warnings(files_result.stdout)
                                        terminal_output.append(f"[SSH] Training files found:")
                                        for line in stdout_filtered.strip().split("\n")[:5]:
                                            if line.strip():
                                                terminal_output.append(f"[SSH]   {line}")
                                    
                                    # Check for onstart script status
                                    terminal_output.append(f"[SSH] Checking onstart script status...")
                                    check_onstart_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "ps aux | grep -E '(bash|onstart|training)' | grep -v grep | head -3 || echo 'no_scripts'"
                                    ]
                                    onstart_result = subprocess.run(check_onstart_cmd, capture_output=True, text=True, timeout=15)
                                    if "no_scripts" not in onstart_result.stdout:
                                        stdout_filtered = filter_malloc_warnings(onstart_result.stdout)
                                        terminal_output.append(f"[SSH] Running processes:")
                                        for line in stdout_filtered.strip().split("\n")[:3]:
                                            if line.strip():
                                                terminal_output.append(f"[SSH]   {line[:150]}")
                                    
                                    # Check debug.log if it exists
                                    terminal_output.append(f"[SSH] Checking debug.log for errors...")
                                    debug_log_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "if [ -f /workspace/output/training/debug.log ]; then tail -30 /workspace/output/training/debug.log; else echo 'no_debug_log'; fi"
                                    ]
                                    debug_result = subprocess.run(debug_log_cmd, capture_output=True, text=True, timeout=15)
                                    if "no_debug_log" not in debug_result.stdout and debug_result.stdout.strip():
                                        stdout_filtered = filter_malloc_warnings(debug_result.stdout)
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
                                        log_lines = stdout_filtered.strip().split("\n")[-10:]
                                        terminal_output.append(f"[TRAINING LOGS] Last 10 lines:")
                                        for line in log_lines:
                                            if line.strip():
                                                terminal_output.append(f"[TRAINING] {line}")
                                
                                training_status = training_manager.check_training_status(instance_id)
                                active_job["training_status"] = training_status
                                training_manager._save_job(active_job)
                                
                                status_val = training_status.get("status", "unknown")
                                terminal_output.append(f"[INFO] Training status: {status_val}")
                                
                                # Add detailed diagnostics for unknown status
                                if status_val == "unknown":
                                    training_files_exist = training_status.get("training_files_exist", False)
                                    onstart_running = training_status.get("onstart_running", False)
                                    onstart_status = training_status.get("onstart_status", "unknown")
                                    
                                    terminal_output.append(f"[DIAGNOSTICS] Training files exist: {training_files_exist}")
                                    terminal_output.append(f"[DIAGNOSTICS] Onstart script running: {onstart_running}")
                                    terminal_output.append(f"[DIAGNOSTICS] Onstart status: {onstart_status}")
                                    
                                    if not training_files_exist:
                                        terminal_output.append(f"[WARNING] Training files not found on instance!")
                                        terminal_output.append(f"[WARNING] Files may not have been uploaded correctly in Phase 2.")
                                        terminal_output.append(f"[ACTION] Go back to Phase 2 and re-upload files.")
                                    elif onstart_running and onstart_status == "waiting_for_files" and training_files_exist:
                                        terminal_output.append(f"[WARNING] Onstart script is waiting for files, but files exist!")
                                        terminal_output.append(f"[WARNING] The script may be stuck. Training may need to be manually started.")
                                    elif not onstart_running and training_files_exist:
                                        terminal_output.append(f"[WARNING] Onstart script is not running, but files exist!")
                                        terminal_output.append(f"[WARNING] The onstart script may have timed out or failed.")
                                        terminal_output.append(f"[WARNING] Training may need to be manually started via SSH.")
                                        terminal_output.append(f"[INFO] Check debug.log above for error details.")
                                    
                                    onstart_logs = training_status.get("onstart_logs")
                                    if onstart_logs and "No onstart log file found" not in onstart_logs:
                                        terminal_output.append(f"[ONSTART LOGS] Last 20 lines:")
                                        for line in onstart_logs.strip().split("\n")[-10:]:
                                            if line.strip():
                                                terminal_output.append(f"[ONSTART] {line}")
                                
                                if status_val == "completed":
                                    terminal_output.append(f"[SUCCESS] Training completed!")
                                    
                                    # Check if there are more jobs in the queue
                                    job_queue = active_job.get("job_queue")
                                    current_job_index = active_job.get("current_job_index")
                                    
                                    if job_queue and current_job_index is not None and current_job_index + 1 < len(job_queue):
                                        # More jobs in queue - prepare and start next job
                                        next_job_index = current_job_index + 1
                                        next_job = job_queue[next_job_index]
                                        
                                        terminal_output.append(f"[QUEUE] Job {current_job_index + 1} complete. Starting job {next_job_index + 1}/{len(job_queue)}...")
                                        
                                        # Prepare new package for next job
                                        try:
                                            new_package_info = training_manager.prepare_training_package(
                                                epochs=active_job.get("epochs", 3),
                                                learning_rate=active_job.get("learning_rate", 2e-4),
                                                hf_model_override=active_job.get("hf_model_override"),
                                                yaml_config_path=next_job.get("yaml_path"),
                                                file_group=next_job.get("file_group")
                                            )
                                            
                                            # Update job with new package info and increment index
                                            active_job["package_info"] = new_package_info
                                            active_job["current_job_index"] = next_job_index
                                            active_job["files_uploaded"] = False  # Reset for new job
                                            
                                            # Update YAML config in package_info for display
                                            if next_job.get("yaml_path"):
                                                from pathlib import Path
                                                yaml_filename = Path(next_job.get("yaml_path")).name
                                                new_package_info["yaml_config"] = yaml_filename
                                            else:
                                                new_package_info["yaml_config"] = None
                                            
                                            training_manager._save_job(active_job)
                                            
                                            terminal_output.append(f"[QUEUE] Next job prepared. Resetting to Phase 2 for file upload...")
                                            terminal_output.append(f"[INFO] Job {next_job_index + 1}/{len(job_queue)}: {next_job.get('file_count', 0)} file(s)")
                                            
                                            # Reset to Phase 2 for next job
                                            st.session_state[phase_key] = 2
                                            # Reset phase 2 init so directories are created again
                                            phase2_init_key = f"phase2_init_{active_job.get('instance_id')}"
                                            if phase2_init_key in st.session_state:
                                                del st.session_state[phase2_init_key]
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.rerun()
                                        except Exception as e:
                                            terminal_output.append(f"[ERROR] Failed to prepare next job: {str(e)}")
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
                                else:
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
                        if st.button("🔄 Redo Phase", key="redo_phase_3", type="secondary"):
                            try:
                                instance_id = active_job.get("instance_id")
                                
                                # Clear terminal output
                                terminal_output = []
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Redoing Phase 3 - killing processes and cleaning up...")
                                
                                # Get SSH info
                                job_status = training_manager.get_job_status(instance_id)
                                ssh_host = job_status.get("ssh_host")
                                ssh_port = job_status.get("ssh_port", 22)
                                
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
                                    
                                    # Step 2.5: Regenerate and re-upload config file with correct model name
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
                                                    num_epochs=old_config.get("num_epochs", 3),
                                                    learning_rate=old_config.get("learning_rate", 2e-4),
                                                    lora_r=old_config.get("lora_r", 8),
                                                    lora_alpha=old_config.get("lora_alpha", 16),
                                                    lora_dropout=old_config.get("lora_dropout", 0.05),
                                                    batch_size=old_config.get("micro_batch_size", 4),
                                                    gradient_accumulation_steps=old_config.get("gradient_accumulation_steps", 4)
                                                )
                                                
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
                                                    
                                                    # Re-upload config file
                                                    terminal_output.append(f"[SCP] Re-uploading config file with corrected model name...")
                                                    scp_config_cmd = [
                                                        "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                                                        str(config_path),
                                                        f"root@{ssh_host}:/workspace/data/axolotl_config.yaml"
                                                    ]
                                                    scp_config_result = subprocess.run(scp_config_cmd, capture_output=True, text=True, timeout=300)
                                                    stdout_filtered = filter_malloc_warnings(scp_config_result.stdout)
                                                    stderr_filtered = filter_malloc_warnings(scp_config_result.stderr)
                                                    
                                                    if scp_config_result.returncode == 0:
                                                        terminal_output.append(f"[SCP] Config file re-uploaded successfully")
                                                        if stdout_filtered.strip():
                                                            terminal_output.append(f"[SCP] {stdout_filtered}")
                                                        
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
                                                    else:
                                                        error_output = stderr_filtered or stdout_filtered
                                                        terminal_output.append(f"[ERROR] Config re-upload failed (exit code {scp_config_result.returncode})")
                                                        terminal_output.append(f"[ERROR] {error_output[:500]}")
                                            else:
                                                terminal_output.append(f"[WARNING] Could not map base_model '{base_model}' to HF model. Using existing config.")
                                        except Exception as e:
                                            error_msg = str(e)
                                            terminal_output.append(f"[ERROR] Failed to regenerate config: {error_msg}")
                                            import traceback
                                            terminal_output.append(f"[ERROR] Traceback: {traceback.format_exc()[:300]}")
                                    
                                    # Step 3: Restart training
                                    terminal_output.append(f"[SSH] Restarting training...")
                                    # Use a single-line command that SSH can handle properly
                                    # The command changes to axolotl directory, runs training in background, and captures PID
                                    restart_command = (
                                        "cd /workspace/axolotl && "
                                        "nohup accelerate launch -m axolotl.cli.train /workspace/data/axolotl_config.yaml "
                                        "> /workspace/output/training/training.log 2>&1 & "
                                        "TRAIN_PID=$! && "
                                        "echo 'Training restarted (PID: '$TRAIN_PID')' && "
                                        "sleep 1 && "
                                        "ps -p $TRAIN_PID > /dev/null && echo 'Training process confirmed running' || echo 'Warning: Process may have failed to start'"
                                    )
                                    restart_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        restart_command
                                    ]
                                    restart_result = subprocess.run(restart_cmd, capture_output=True, text=True, timeout=30)
                                    if restart_result.returncode == 0:
                                        stdout_filtered = filter_malloc_warnings(restart_result.stdout)
                                        terminal_output.append(f"[SSH] Training restarted successfully")
                                        if stdout_filtered.strip():
                                            terminal_output.append(f"[SSH] {stdout_filtered}")
                                        terminal_output.append(f"[SUCCESS] Phase 3 redo complete - training restarted")
                                        
                                        # Reset training status
                                        active_job["training_status"] = {"status": "training", "restarted": True}
                                        training_manager._save_job(active_job)
                                    else:
                                        stderr_filtered = filter_malloc_warnings(restart_result.stderr)
                                        terminal_output.append(f"[ERROR] Failed to restart training: {stderr_filtered[:200]}")
                                        terminal_output.append(f"[ERROR] Please check SSH connection and try again")
                                
                                st.session_state[terminal_output_key] = terminal_output
                                st.rerun()
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] Failed to redo Phase 3: {error_msg}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                    
                    with col3:
                        training_status = active_job.get("training_status", {})
                        if training_status.get("status") == "completed":
                            st.success("✅ Training completed! Click 'Next Phase' to finalize.")
                            if st.button("➡️ Next Phase", key="next_phase_3", type="primary"):
                                st.session_state[phase_key] = 4
                                # Clear terminal for next phase
                                st.session_state[terminal_output_key] = []
                                st.rerun()
                    
                    # Cancel Job button (always visible) with confirmation
                    st.markdown("---")
                    cancel_confirm_key = f"cancel_confirm_{active_job.get('instance_id')}"
                    
                    if cancel_confirm_key not in st.session_state:
                        st.session_state[cancel_confirm_key] = False
                    
                    if not st.session_state[cancel_confirm_key]:
                        if st.button("❌ Cancel Job", key="cancel_job_phase_3", type="secondary"):
                            st.session_state[cancel_confirm_key] = True
                            st.rerun()
                    else:
                        st.warning("⚠️ Are you sure you want to cancel this job? This will destroy the Vast.ai instance and cannot be undone.")
                        col_confirm, col_cancel = st.columns([1, 1])
                        with col_confirm:
                            if st.button("✅ Confirm Cancel", key="confirm_cancel_phase_3", type="primary"):
                                try:
                                    # Ensure datetime is available
                                    from datetime import datetime
                                    instance_id = active_job.get("instance_id")
                                    if instance_id:
                                        terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Cancelling job and destroying instance...")
                                        terminal_output.append(f"[API] DELETE /instances/{instance_id}/")
                                        
                                        # Destroy instance
                                        destroyed = training_manager.vast_client.destroy_instance(instance_id)
                                        if destroyed:
                                            terminal_output.append(f"[API] Instance destroyed successfully")
                                            
                                            # Mark job as cancelled
                                            active_job["status"] = "cancelled"
                                            active_job["cancelled_at"] = datetime.now().isoformat()
                                            training_manager._save_job(active_job)
                                            
                                            terminal_output.append(f"[SUCCESS] Job cancelled successfully")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.session_state[cancel_confirm_key] = False
                                            st.success("✅ Job cancelled and instance destroyed.")
                                            st.rerun()
                                        else:
                                            terminal_output.append(f"[WARNING] Instance destruction returned False")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.warning("Instance destruction may have failed. Please check Vast.ai dashboard.")
                                except Exception as e:
                                    error_msg = str(e)
                                    terminal_output.append(f"[ERROR] Failed to cancel job: {error_msg}")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.error(f"Error cancelling job: {error_msg}")
                                finally:
                                    st.session_state[cancel_confirm_key] = False
                        with col_cancel:
                            if st.button("❌ Cancel", key="cancel_confirm_phase_3"):
                                st.session_state[cancel_confirm_key] = False
                                st.rerun()
                
                # Phase 4: Finalize
                elif current_phase == 4:
                    # Ensure datetime is available in this scope
                    from datetime import datetime
                    
                    st.markdown(f"### {phases[4]['icon']} Phase 4: {phases[4]['name']}")
                    st.caption(phases[4]['description'])
                    
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
                    
                    # Action buttons
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("✅ Finalize Training", key="finalize_training", type="primary"):
                            try:
                                instance_id = active_job.get("instance_id")
                                
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting finalization...")
                                
                                # Step 1: Create version directory
                                terminal_output.append(f"[LOCAL] Creating version directory...")
                                from utils.model_manager import ModelManager
                                model_manager = ModelManager()
                                version_dir = model_manager.create_version_folder(model_name)
                                terminal_output.append(f"[LOCAL] Version directory created: {version_dir}")
                                
                                # Step 2: Move files from queue (only files from current job)
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
                                
                                # Step 3: Get SSH info for weight download
                                terminal_output.append(f"[API] Getting instance status for SSH info...")
                                job_status = training_manager.get_job_status(instance_id)
                                ssh_host = job_status.get("ssh_host")
                                ssh_port = job_status.get("ssh_port", 22)
                                
                                if not ssh_host:
                                    terminal_output.append(f"[ERROR] SSH host not available. Cannot download weights.")
                                    st.error("SSH host not available. Cannot download weights automatically.")
                                else:
                                    terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                    
                                    # Step 4: Check for weight files on remote instance
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
                                    
                                    # Step 5: Download weights
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
                                    else:
                                        terminal_output.append(f"[WARNING] No weight files found after download")
                                
                                # All jobs should be complete by now (handled in Phase 3)
                                # Destroy instance and finalize
                                terminal_output.append(f"[API] Destroying instance on Vast.ai...")
                                try:
                                    training_manager.vast_client.destroy_instance(instance_id)
                                    terminal_output.append(f"[API] Instance destroyed successfully")
                                except Exception as e:
                                    terminal_output.append(f"[WARNING] Instance destruction failed: {str(e)}")
                                
                                # Update job
                                active_job["status"] = "completed"
                                active_job["finalized"] = True
                                active_job["version_dir"] = str(version_dir)
                                active_job["version"] = version_dir.name.replace("V", "")
                                training_manager._save_job(active_job)
                                
                                terminal_output.append(f"[SUCCESS] All training jobs finalized!")
                                terminal_output.append(f"[INFO] Version: {active_job.get('version')}")
                                terminal_output.append(f"[INFO] Weights: {weights_dir}")
                                
                                st.session_state[terminal_output_key] = terminal_output
                                st.success("✅ All training jobs finalized successfully!")
                                st.rerun()
                            except Exception as e:
                                error_msg = str(e)
                                terminal_output.append(f"[ERROR] {error_msg}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                    
                    with col2:
                        if active_job.get("finalized"):
                            st.success("✅ Training finalized! All phases complete.")
                    
                    # Instance Information
                    st.markdown("---")
                    st.markdown("#### 💻 Instance Information")
                    instance_info_cols = st.columns(5)
                    
                    with instance_info_cols[0]:
                        gpu_info = active_job.get("gpu_info", "Unknown")
                        gpu_name_display = active_job.get("gpu_name", gpu_info)
                        st.metric("GPU", gpu_name_display if gpu_name_display else gpu_info)
                    
                    with instance_info_cols[1]:
                        # Show actual GPU RAM, not minimum
                        actual_gpu_ram = active_job.get("actual_gpu_ram")
                        if isinstance(actual_gpu_ram, (int, float)) and actual_gpu_ram > 0:
                            st.metric("GPU RAM", f"{actual_gpu_ram:.1f} GB")
                        else:
                            # Fallback to min_gpu_ram if actual not available
                            min_gpu_ram = active_job.get("min_gpu_ram")
                            if isinstance(min_gpu_ram, (int, float)):
                                st.metric("GPU RAM", f"{min_gpu_ram} GB (min)")
                            else:
                                st.metric("GPU RAM", "Unknown")
                    
                    with instance_info_cols[2]:
                        price_per_hour = active_job.get("price_per_hour", 0)
                        if isinstance(price_per_hour, (int, float)) and price_per_hour > 0:
                            st.metric("Price/Hour", f"${price_per_hour:.3f}")
                        else:
                            st.metric("Price/Hour", "Unknown")
                    
                    with instance_info_cols[3]:
                        actual_disk_space = active_job.get("actual_disk_space")
                        if isinstance(actual_disk_space, (int, float)) and actual_disk_space > 0:
                            st.metric("Disk Space", f"{actual_disk_space} GB")
                        else:
                            # Fallback to requested disk_space
                            disk_space = active_job.get("disk_space")
                            if isinstance(disk_space, (int, float)):
                                st.metric("Disk Space", f"{disk_space} GB")
                            else:
                                st.metric("Disk Space", "Unknown")
                    
                    with instance_info_cols[4]:
                        location = active_job.get("location", "Unknown")
                        num_gpus = active_job.get("num_gpus")
                        if num_gpus and num_gpus > 0:
                            st.metric("GPUs", num_gpus)
                        else:
                            st.metric("Location", str(location))
                    
                    # Cancel Job button (always visible, but warn if already finalized) with confirmation
                    st.markdown("---")
                    cancel_confirm_key = f"cancel_confirm_{active_job.get('instance_id')}"
                    
                    if cancel_confirm_key not in st.session_state:
                        st.session_state[cancel_confirm_key] = False
                    
                    if active_job.get("finalized"):
                        st.warning("⚠️ Training already finalized. Cancelling will only destroy the instance.")
                    
                    if not st.session_state[cancel_confirm_key]:
                        if st.button("❌ Cancel Job", key="cancel_job_phase_4", type="secondary"):
                            st.session_state[cancel_confirm_key] = True
                            st.rerun()
                    else:
                        st.warning("⚠️ Are you sure you want to cancel this job? This will destroy the Vast.ai instance and cannot be undone.")
                        col_confirm, col_cancel = st.columns([1, 1])
                        with col_confirm:
                            if st.button("✅ Confirm Cancel", key="confirm_cancel_phase_4", type="primary"):
                                try:
                                    # Ensure datetime is available
                                    from datetime import datetime
                                    instance_id = active_job.get("instance_id")
                                    if instance_id:
                                        terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Cancelling job and destroying instance...")
                                        terminal_output.append(f"[API] DELETE /instances/{instance_id}/")
                                        
                                        # Destroy instance
                                        destroyed = training_manager.vast_client.destroy_instance(instance_id)
                                        if destroyed:
                                            terminal_output.append(f"[API] Instance destroyed successfully")
                                            
                                            # Mark job as cancelled (even if finalized)
                                            active_job["status"] = "cancelled"
                                            active_job["cancelled_at"] = datetime.now().isoformat()
                                            training_manager._save_job(active_job)
                                            
                                            terminal_output.append(f"[SUCCESS] Job cancelled successfully")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.session_state[cancel_confirm_key] = False
                                            st.success("✅ Job cancelled and instance destroyed.")
                                            st.rerun()
                                        else:
                                            terminal_output.append(f"[WARNING] Instance destruction returned False")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.warning("Instance destruction may have failed. Please check Vast.ai dashboard.")
                                except Exception as e:
                                    error_msg = str(e)
                                    terminal_output.append(f"[ERROR] Failed to cancel job: {error_msg}")
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.error(f"Error cancelling job: {error_msg}")
                                finally:
                                    st.session_state[cancel_confirm_key] = False
                        with col_cancel:
                            if st.button("❌ Cancel", key="cancel_confirm_phase_4"):
                                st.session_state[cancel_confirm_key] = False
                                st.rerun()
                
            except Exception as e:
                st.warning(f"Could not load training jobs: {str(e)}")
                import traceback
                with st.expander("Error Details", expanded=False):
                    st.code(traceback.format_exc())
    
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
            for metadata_file in queue_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
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
                            for metadata_file in queue_dir.glob("*_metadata.json"):
                                try:
                                    with open(metadata_file, 'r', encoding='utf-8') as f:
                                        file_metadata = json.load(f)
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
                                    for metadata_file in queue_dir.glob("*_metadata.json"):
                                        try:
                                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                                file_metadata = json.load(f)
                                            
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
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Training File",
            type=['json', 'jsonl', 'txt'],
            help="Upload JSON, JSONL, or TXT files to be used for training. After uploading, select a YAML config (optional) and click 'Process and Queue'.",
            key="training_file_uploader"
        )
        
        # Show uploaded file info if file is uploaded
        if uploaded_file is not None:
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
                        with open(metadata_path, 'r', encoding='utf-8') as verify_f:
                            verify_metadata = json.load(verify_f)
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
                        st.session_state["yaml_selector"] = "None"  # Reset to "None" option
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        # Show queued files
        st.markdown("---")
        st.markdown("#### Queued Files")
        
        queued_files = []
        if queue_dir.exists():
            # Get files from metadata (exclude YAML files)
            for metadata_file in queue_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
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
        
        if 'marked_pairs' not in st.session_state:
            st.session_state.marked_pairs = []
        
        if 'show_marked_pairs' not in st.session_state:
            st.session_state.show_marked_pairs = False
        
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
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Clear Chat", key="clear_chat_btn", help="Clear chat history and start fresh"):
                st.session_state.chat_history = []
                st.session_state.marked_pairs = []
                st.success("Chat cleared!")
                st.rerun()
        
        with col2:
            if st.button("📋 View Marked Pairs", key="view_pairs_btn", help="View and manage Q&A pairs marked for rule generation"):
                st.session_state.show_marked_pairs = not st.session_state.get("show_marked_pairs", False)
                st.rerun()
        
        st.markdown("Chat with your model and mark important Q&A pairs for rule generation")
        
        # Show marked pairs if requested
        if st.session_state.get("show_marked_pairs", False):
            st.markdown("---")
            st.subheader("📋 Marked Q&A Pairs for Rule Generation")
            marked_pairs = data_manager.get_learned_pairs()
            
            if marked_pairs:
                for idx, pair in enumerate(marked_pairs):
                    with st.expander(f"Pair {idx + 1}: {pair.get('question', '')[:50]}...", expanded=False):
                        col_q, col_a = st.columns(2)
                        with col_q:
                            st.write("**Question:**")
                            st.write(pair.get('question', ''))
                        with col_a:
                            st.write("**Answer:**")
                            st.write(pair.get('answer', ''))
                        
                        if st.button("🗑️ Delete", key=f"delete_pair_{idx}"):
                            # Delete this pair
                            all_pairs = data_manager.get_learned_pairs()
                            if idx < len(all_pairs):
                                all_pairs.pop(idx)
                                # Save updated list
                                learned_pairs_path = data_manager.data_dir / "learned_pairs.json"
                                with open(learned_pairs_path, 'w') as f:
                                    json.dump({"pairs": all_pairs}, f, indent=2)
                                st.success("Pair deleted!")
                                st.rerun()
            else:
                st.info("No pairs marked yet. Mark Q&A pairs during conversation to see them here.")
            
            st.markdown("---")
        
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
        
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Mark to learn button
                if message["role"] == "assistant" and i > 0:
                    pair_key = f"mark_pair_{i}"
                    if st.button("📌 Mark to Learn", key=pair_key):
                        user_msg = st.session_state.chat_history[i-1]["content"]
                        assistant_msg = message["content"]
                        pair = data_manager.save_learned_pair(user_msg, assistant_msg)
                        st.session_state.marked_pairs.append(pair)
                        st.success("✅ Pair saved for rule generation!")
        
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
    


