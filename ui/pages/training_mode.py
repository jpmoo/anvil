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
# Ollama removed - using HuggingFace models directly
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


def get_ssh_base_options(ssh_port: int) -> list:
    """
    Get base SSH command options that prevent interactive prompts.
    This handles firewall prompts and host key verification automatically.
    
    Args:
        ssh_port: SSH port number
        
    Returns:
        List of SSH command arguments (port and options)
    """
    return [
        "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10"
    ]


def run_ssh_command(ssh_host: str, ssh_port: int, command: str, timeout: int = 30, 
                    terminal_output: list = None, input_text: str = None) -> subprocess.CompletedProcess:
    """
    Run an SSH command with automatic prompt handling.
    Automatically answers "yes" to any prompts from firewalls or host verification.
    
    Args:
        ssh_host: SSH hostname
        ssh_port: SSH port
        command: Command to execute on remote host
        timeout: Command timeout in seconds
        terminal_output: Optional list to append output messages to
        input_text: Optional input to pipe to stdin (defaults to "yes\n" for prompts)
        
    Returns:
        CompletedProcess result from subprocess.run
    """
    ssh_cmd = [
        "ssh"
    ] + get_ssh_base_options(ssh_port) + [
        f"root@{ssh_host}",
        command
    ]
    
    # Default input to "yes\n" to handle any prompts
    if input_text is None:
        input_text = "yes\n"
    
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_text
        )
        return result
    except subprocess.TimeoutExpired:
        if terminal_output:
            terminal_output.append(f"[SSH] Command timed out after {timeout}s")
        raise


def kill_all_training_processes(ssh_host: str, ssh_port: int, terminal_output: list, max_retries: int = 5) -> bool:
    """
    Kill all training-related processes on the remote instance.
    This function is comprehensive and verifies processes are actually killed.
    Uses aggressive killing including process groups to handle child processes.
    
    Args:
        ssh_host: SSH hostname
        ssh_port: SSH port
        terminal_output: List to append output messages to
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if all processes were killed, False otherwise
    """
    import subprocess
    import time
    
    for attempt in range(max_retries):
        if attempt > 0:
            terminal_output.append(f"[SSH] Retry {attempt + 1}/{max_retries}: Killing remaining training processes...")
            time.sleep(3)  # Longer wait between retries
        
        # Comprehensive kill command that handles multiple scenarios
        # This kills processes AND their entire process groups to catch child processes
        kill_command = (
            # First, get all PIDs and kill process groups (more aggressive)
            "PIDS=$(ps aux | grep -E 'accelerate|axolotl|axolotl.cli.train' | grep -v grep | awk '{print $2}' | tr '\\n' ' '); "
            "if [ -n \"$PIDS\" ]; then "
            "  for pid in $PIDS; do "
            "    kill -9 -$pid 2>/dev/null || kill -9 $pid 2>/dev/null || true; "  # Kill process group with -PID
            "  done; "
            "fi; "
            # Kill by process name patterns (pkill kills process groups automatically)
            "pkill -9 -f 'accelerate' 2>/dev/null || true; "
            "pkill -9 -f 'axolotl' 2>/dev/null || true; "
            "pkill -9 -f 'axolotl.cli.train' 2>/dev/null || true; "
            "pkill -9 -f 'axolotl.cli' 2>/dev/null || true; "
            # Kill Python processes running training-related commands (handle both xargs -r and without)
            "ps aux | grep -E 'python.*train|python.*axolotl|python.*accelerate' | grep -v grep | awk '{print $2}' | "
            "  (xargs -r kill -9 2>/dev/null || xargs kill -9 2>/dev/null || true); "
            # Kill processes in /workspace/axolotl directory
            "ps aux | grep '/workspace/axolotl' | grep -v grep | awk '{print $2}' | "
            "  (xargs -r kill -9 2>/dev/null || xargs kill -9 2>/dev/null || true); "
            # Kill any process with axolotl_config.yaml in command
            "ps aux | grep 'axolotl_config.yaml' | grep -v grep | awk '{print $2}' | "
            "  (xargs -r kill -9 2>/dev/null || xargs kill -9 2>/dev/null || true); "
            # Kill any process with /workspace/data/axolotl_config.yaml
            "ps aux | grep '/workspace/data/axolotl_config.yaml' | grep -v grep | awk '{print $2}' | "
            "  (xargs -r kill -9 2>/dev/null || xargs kill -9 2>/dev/null || true); "
            # Wait longer for processes to die and their children
            "sleep 5; "
            # Count remaining processes
            "remaining=$(ps aux | grep -E 'accelerate|axolotl|axolotl.cli.train' | grep -v grep | wc -l); "
            "echo 'Kill attempt $((attempt+1)): Remaining processes: '$remaining"
        )
        
        kill_cmd = [
            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
            f"root@{ssh_host}",
            kill_command
        ]
        
        try:
            kill_result = subprocess.run(kill_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
            stdout_filtered = filter_malloc_warnings(kill_result.stdout)
            stderr_filtered = filter_malloc_warnings(kill_result.stderr)
            
            if kill_result.returncode == 0:
                if stdout_filtered.strip():
                    terminal_output.append(f"[SSH] {stdout_filtered.strip()}")
                
                # Extract remaining count from output
                import re
                remaining_match = re.search(r'Remaining processes:\s*(\d+)', stdout_filtered)
                if remaining_match:
                    remaining = int(remaining_match.group(1))
                    if remaining == 0:
                        terminal_output.append(f"[SSH] ✓ All training processes killed successfully")
                        return True
                    else:
                        terminal_output.append(f"[SSH] ⚠️ {remaining} process(es) still running after kill attempt")
                else:
                    # If we can't parse, assume success if no error
                    terminal_output.append(f"[SSH] ✓ Kill command completed")
                    # Verify by checking processes again
                    verify_cmd = [
                        "ssh"
                    ] + get_ssh_base_options(ssh_port) + [
                        f"root@{ssh_host}",
                        "ps aux | grep -E 'accelerate|axolotl|axolotl.cli.train' | grep -v grep | wc -l"
                    ]
                    verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
                    if verify_result.returncode == 0:
                        remaining = int(verify_result.stdout.strip())
                        if remaining == 0:
                            terminal_output.append(f"[SSH] ✓ Verified: All training processes killed")
                            return True
                        else:
                            terminal_output.append(f"[SSH] ⚠️ Verification shows {remaining} process(es) still running")
            else:
                if stderr_filtered.strip():
                    terminal_output.append(f"[SSH] Warning: {stderr_filtered.strip()}")
        except subprocess.TimeoutExpired:
            terminal_output.append(f"[SSH] ⚠️ Kill command timed out (attempt {attempt + 1})")
        except Exception as e:
            terminal_output.append(f"[SSH] ⚠️ Error killing processes: {str(e)}")
    
    # Final verification with multiple checks
    terminal_output.append(f"[SSH] Performing final verification (waiting for processes to fully terminate)...")
    time.sleep(2)  # Extra wait before final check
    
    # Check multiple times to ensure processes are really gone
    all_clear = False
    for verify_attempt in range(3):
        verify_cmd = [
            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
            f"root@{ssh_host}",
            "ps aux | grep -E 'accelerate|axolotl|axolotl.cli.train' | grep -v grep"
        ]
        try:
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
            if verify_result.returncode == 0 and verify_result.stdout.strip():
                # There are still processes
                process_lines = verify_result.stdout.strip().split('\n')
                if verify_attempt < 2:  # Not the last attempt
                    terminal_output.append(f"[SSH] ⚠️ Verification {verify_attempt + 1}: {len(process_lines)} process(es) still running, waiting...")
                    time.sleep(3)
                    # Try one more aggressive kill
                    final_kill_cmd = [
                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                        f"root@{ssh_host}",
                        "pkill -9 -f 'accelerate' 2>/dev/null || true; "
                        "pkill -9 -f 'axolotl' 2>/dev/null || true; "
                        "ps aux | grep -E 'accelerate|axolotl' | grep -v grep | awk '{print $2}' | "
                        "  (xargs -r kill -9 2>/dev/null || xargs kill -9 2>/dev/null || true); "
                        "sleep 2"
                    ]
                    subprocess.run(final_kill_cmd, capture_output=True, text=True, timeout=20, input="yes\n")
                else:
                    # Last attempt - show details
                    terminal_output.append(f"[SSH] ⚠️ WARNING: {len(process_lines)} training process(es) still running after all kill attempts:")
                    for line in process_lines[:10]:  # Show first 10
                        terminal_output.append(f"[SSH]   {line.strip()}")
                    terminal_output.append(f"[SSH] ⚠️ These processes may need manual intervention")
                    return False
            else:
                all_clear = True
                break
        except Exception as e:
            if verify_attempt == 2:  # Last attempt
                terminal_output.append(f"[SSH] ⚠️ Could not verify process status: {str(e)}")
                return False
            time.sleep(2)
    
    if all_clear:
        terminal_output.append(f"[SSH] ✓ Final verification: No training processes running - safe to proceed")
        return True
    else:
        return False


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
    
    # Check for active jobs BEFORE tabs (so all tabs can access active_job)
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
    
    tab1, tab2 = st.tabs(["🚀 Training", "📄 Context Upload"])
    
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
        if len(queued_files) > 0:
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
            
            # Build jobs list with sizes for display
            jobs_list = []
            for yaml_key, files_in_group in files_by_yaml.items():
                # Calculate total size for this job
                job_size = sum(f.get('size', 0) for f in files_in_group)
                yaml_filename = yaml_key if yaml_key != 'No YAML' else None
                jobs_list.append({
                    "yaml_filename": yaml_filename,
                    "file_count": len(files_in_group),
                    "job_size": job_size,
                    "files": files_in_group
                })
            
            # Sort jobs by size (descending)
            jobs_list.sort(key=lambda x: x["job_size"], reverse=True)
            
            # Display jobs in order of size
            st.markdown("#### 📋 Queued Jobs (sorted by size, largest first)")
            for idx, job in enumerate(jobs_list, 1):
                job_size_mb = job["job_size"] / (1024 * 1024)
                yaml_desc = f"YAML: {job['yaml_filename']}" if job['yaml_filename'] else "No YAML (stock config)"
                job_title = f"Job {idx}: {job['file_count']} file(s), {job_size_mb:.2f} MB - {yaml_desc}"
                
                with st.expander(job_title, expanded=False):
                    # Show files for this job beneath the job header
                    for file_meta in job['files']:
                        file_type = file_meta.get('file_type', 'unknown').upper()
                        st.write(f"  • **{file_meta['filename']}** ({file_type}, {file_meta.get('size', 0):,} bytes)")
        else:
            st.info("No files queued yet")
            st.caption("Upload files in Tab 2 to queue them for training")
        
        # Overall status
        has_training_data = len(queued_files) > 0
        if not has_training_data:
            st.warning("⚠️ No training data available. Add files in Tab 2 before training.")
        
        st.markdown("")  # Break between sections
        
        # active_job is now defined before tabs, so all tabs can access it
        
        # Launch Training Section - only show if no active (non-dismissed) jobs
        # Show this section if there's training data AND no active jobs
        if len(queued_files) > 0 and not has_active_jobs:
            st.markdown("### 🚀 Launch Training")
            
            selected_existing_instance = None
            instances_detected = False
            
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
                            instances_detected = True
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
            
            # Show note only when no instances are detected
            if not instances_detected:
                st.info("ℹ️ **Note:** You must have an existing Vast.ai instance running. The program will validate the instance before starting training.")
            
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
                            # Build all jobs and calculate sizes - select largest to run
                            all_jobs = []
                            
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
                                
                                # Calculate total size of files in this job
                                job_size = 0
                                for file_meta in file_group:
                                    filename = file_meta.get("filename")
                                    if filename:
                                        file_path = queue_dir / filename
                                        if file_path.exists():
                                            job_size += file_path.stat().st_size
                                
                                # Add to all jobs list
                                all_jobs.append({
                                    "yaml_filename": yaml_filename,
                                    "yaml_path": yaml_path,
                                    "file_group": file_group,
                                    "file_count": len(file_group),
                                    "job_size": job_size  # Total size in bytes
                                })
                            
                            if all_jobs:
                                # Sort jobs by size (descending) and select largest
                                all_jobs.sort(key=lambda x: x["job_size"], reverse=True)
                                selected_job = all_jobs[0]
                                remaining_jobs = all_jobs[1:]  # Keep others queued for next time
                                
                                # Display job selection info
                                job_size_mb = selected_job["job_size"] / (1024 * 1024)
                                st.info(f"📊 Selected largest job ({job_size_mb:.2f} MB, {selected_job['file_count']} file(s)) to run. {len(remaining_jobs)} job(s) will remain queued.")
                                
                                # Create single-item job list for the selected job
                                job_queue = [selected_job]
                                
                                # Launch single instance with only the selected job
                                with st.spinner(f"Launching training instance with selected job ({selected_job['file_count']} file(s), {job_size_mb:.2f} MB)..."):
                                    try:
                                        # Get HF token from session state or config
                                        from utils.config import get_hf_token
                                        hf_token = st.session_state.get('hf_token') or get_hf_token()
                                        
                                        # Must use existing instance
                                        if not selected_existing_instance:
                                            st.error("❌ Please select an existing instance.")
                                        else:
                                            # Initialize terminal output key and terminal output
                                            # Use selected_existing_instance for the key (will be updated after job_info is received)
                                            terminal_output_key = f"terminal_output_{selected_existing_instance}"
                                            terminal_output = st.session_state.get(terminal_output_key, [])
                                            if not terminal_output:
                                                terminal_output = []
                                            
                                            # Get SSH port override if set (only use if not default 22)
                                            ssh_port_override_value = None
                                            if ssh_port_override != 22:
                                                ssh_port_override_value = ssh_port_override
                                            
                                            # Log before API call
                                            terminal_output.append(f"[API] Calling launch_training_job with instance_id={selected_existing_instance}")
                                            st.session_state[terminal_output_key] = terminal_output
                                            
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
                                            
                                            # Log API response
                                            terminal_output.append(f"[API] launch_training_job response received")
                                            terminal_output.append(f"[API] job_info keys: {list(job_info.keys())}")
                                            terminal_output.append(f"[API] instance_id: {job_info.get('instance_id')}")
                                            terminal_output.append(f"[API] status: {job_info.get('status')}")
                                            terminal_output.append(f"[API] gpu_info: {job_info.get('gpu_info')}")
                                            if job_info.get('ssh_port'):
                                                terminal_output.append(f"[API] ssh_port: {job_info.get('ssh_port')}")
                                            
                                            # Update terminal_output_key if instance_id changed
                                            instance_id = job_info.get("instance_id")
                                            if instance_id and instance_id != selected_existing_instance:
                                                # Instance ID changed, update the key
                                                terminal_output_key = f"terminal_output_{instance_id}"
                                            
                                            st.session_state[terminal_output_key] = terminal_output
                                            
                                            if len(job_queue) > 0:
                                                queue_item = job_queue[0]
                                                job_yaml = queue_item.get('yaml_filename')
                                                yaml_desc = f" (YAML: {job_yaml})" if job_yaml else " (stock/default config)"
                                                job_size_mb = queue_item.get('job_size', 0) / (1024 * 1024)
                                                success_msg = f"✅ Launched training job:\n"
                                                success_msg += f"  • {queue_item['file_count']} file(s), {job_size_mb:.2f} MB{yaml_desc}\n"
                                                if len(all_jobs) > 1:
                                                    success_msg += f"  • {len(remaining_jobs)} job(s) remain queued for next time\n"
                                            st.success(success_msg)
                                            
                                            # Clear terminal output for new job (will be repopulated in phase tracking)
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
                    
                    # Show job info
                    yaml_info = job.get("package_info", {}).get("yaml_config")
                    if yaml_info:
                        queue_display = f"YAML: {yaml_info}"
                    elif job_queue and len(job_queue) > 0:
                        queue_display = "Job queued"
                    else:
                        queue_display = "No YAML"
                    
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
                                
                                # Reset phase to 1 and clear session state BEFORE deleting job
                                phase_key = f"training_phase_{instance_id}"
                                terminal_output_key = f"terminal_output_{instance_id}"
                                
                                # Clear session state for this job first (clears phase and terminal output)
                                if phase_key in st.session_state:
                                    del st.session_state[phase_key]
                                if terminal_output_key in st.session_state:
                                    del st.session_state[terminal_output_key]
                                
                                # Delete the job entirely from the jobs list (as if it never existed)
                                jobs = training_manager._load_jobs()
                                jobs = [j for j in jobs if j.get("instance_id") != instance_id]
                                # Save the updated jobs list using the same method as _save_job
                                import json
                                training_manager.training_dir.mkdir(parents=True, exist_ok=True)
                                with open(training_manager.jobs_file, 'w') as f:
                                    json.dump(jobs, f, indent=2)
                                
                                st.session_state[dismiss_confirm_key] = False
                                st.success("✅ Job dismissed and deleted. You can now launch a new training job.")
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
                # But respect manual phase navigation - don't force phase changes if user is already in a phase
                if job_status == 'launching':
                    target_phase = 1  # Validate instance
                elif job_status == 'validated' and not active_job.get('files_uploaded'):
                    # Go to Phase 2 when validated but files not uploaded yet
                    target_phase = 2  # Upload file
                elif job_status == 'validated' and active_job.get('files_uploaded'):
                    # Files uploaded but status still 'validated' - allow Phase 3 to proceed
                    # But if user is currently in Phase 2, let them stay there (they might want to verify/re-upload)
                    if current_phase == 2:
                        target_phase = 2  # Stay in Phase 2 if already there
                    else:
                        target_phase = 3  # Otherwise go to Phase 3
                elif job_status == 'running' and not active_job.get('files_uploaded'):
                    # If status is 'running' but not validated, go to Phase 1 to validate
                    target_phase = 1  # Validate instance
                elif job_status == 'running' and active_job.get('files_uploaded'):
                    target_phase = 3  # Do training
                elif job_status == 'completed':
                    target_phase = 4  # Finalize
                else:
                    # Also check training_status for completion (more reliable than job_status)
                    training_status_check = active_job.get("training_status", {})
                    if training_status_check.get("status") == "completed":
                        target_phase = 4  # Finalize
                    else:
                        target_phase = 1  # Default to phase 1 (validate instance)
                
                # Set phase if not set, or if it needs to be updated based on status
                # But only change phase if it's not already set (to respect manual navigation)
                # EXCEPTION: Always allow transition to Phase 4 if training is completed
                phase_changed = False
                
                # Check if phase was manually set to 4 (e.g., by button click) - preserve it
                current_phase_in_state = st.session_state.get(phase_key)
                if current_phase_in_state == 4:
                    # Phase 4 was manually set - check if we should keep it
                    training_status_check = active_job.get("training_status", {})
                    if training_status_check.get("status") == "completed":
                        # Keep Phase 4 - training is completed
                        current_phase = 4
                        phase_changed = False
                    else:
                        # Phase 4 was set but training not completed - might be premature, but respect it
                        current_phase = 4
                        phase_changed = False
                elif phase_key not in st.session_state:
                    # Phase not set yet - use target phase
                    st.session_state[phase_key] = target_phase
                    current_phase = target_phase
                    phase_changed = (target_phase != current_phase_in_state) if current_phase_in_state else False
                elif current_phase != target_phase:
                    # If target is Phase 4 and training is completed, always allow the transition
                    if target_phase == 4:
                        training_status_check = active_job.get("training_status", {})
                        if training_status_check.get("status") == "completed":
                            phase_changed = True
                            st.session_state[phase_key] = 4
                            current_phase = 4
                        elif current_phase == 3:
                            # If we're in Phase 3 and target is 4, but status isn't "completed" yet,
                            # don't force the change (let user or button trigger it)
                            current_phase = current_phase_in_state
                        else:
                            # Normal phase change
                            phase_changed = True
                            st.session_state[phase_key] = target_phase
                            current_phase = target_phase
                    else:
                        # Phase is changing - clear terminal output
                        phase_changed = True
                        st.session_state[phase_key] = target_phase
                        current_phase = target_phase
                else:
                    # Phase matches target
                    current_phase = current_phase_in_state
                
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
                    
                    # Initialize Phase 1: Reset flags and clean up old files on instance
                    phase1_init_key = f"phase1_init_{active_job.get('instance_id')}"
                    if phase1_init_key not in st.session_state:
                        try:
                            instance_id = active_job.get("instance_id")
                            if instance_id:
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Phase 1 - resetting flags and cleaning up old files...")
                                
                                # Reset all flags to false
                                active_job["files_uploaded"] = False
                                # No current_job_index needed - single job mode
                                # Reset any other training-related flags
                                if "training_started" in active_job:
                                    active_job["training_started"] = False
                                if "training_completed" in active_job:
                                    active_job["training_completed"] = False
                                training_manager._save_job(active_job)
                                terminal_output.append(f"[INFO] Reset all flags to false")
                                
                                # Get SSH info for cleanup
                                ssh_host = active_job.get("ssh_host")
                                ssh_port_override = active_job.get("ssh_port_override")
                                if ssh_port_override:
                                    ssh_port = ssh_port_override
                                else:
                                    ssh_port = active_job.get("ssh_port", 22)
                                
                                # If SSH info not available, try to get it
                                if not ssh_host:
                                    try:
                                        ssh_info = training_manager.get_instance_ssh_info(instance_id)
                                        ssh_host = ssh_info.get("host")
                                        api_ssh_port = ssh_info.get("port", 22)
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                        else:
                                            ssh_port = api_ssh_port
                                        if ssh_host:
                                            active_job["ssh_host"] = ssh_host
                                            active_job["ssh_port"] = ssh_port
                                            training_manager._save_job(active_job)
                                    except Exception as e:
                                        terminal_output.append(f"[WARNING] Could not get SSH info for cleanup: {str(e)[:200]}")
                                
                                # Clean up old files on instance if SSH info is available
                                if ssh_host:
                                    terminal_output.append(f"[SSH] Cleaning up old training files on instance...")
                                    cleanup_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        "cd /workspace/data && "
                                        "echo 'Cleaning up all old training files...' && "
                                        "# Remove all training files (numbered and active) "
                                        "rm -f axolotl_config_0.yaml training_data_0.jsonl && "
                                        "rm -f axolotl_config_*.yaml training_data_*.jsonl && "
                                        "rm -f axolotl_config.yaml training_data.jsonl && "
                                        "# Also clean up any output directories "
                                        "rm -rf /workspace/output/training/* 2>/dev/null || true && "
                                        "rm -rf /workspace/axolotl/prepared_data/* 2>/dev/null || true && "
                                        "echo 'Cleanup complete' && "
                                        "ls -la /workspace/data/ 2>/dev/null | grep -E '(training_data|axolotl_config)' || echo 'No training files found (clean)'"
                                    ]
                                    cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                    if cleanup_result.returncode == 0:
                                        terminal_output.append(f"[SUCCESS] Old files cleaned up on instance")
                                        if cleanup_result.stdout:
                                            for line in cleanup_result.stdout.strip().split("\n"):
                                                if line.strip() and "Cleaning up" not in line and "complete" not in line and "No training files" not in line:
                                                    terminal_output.append(f"[SSH]   {line}")
                                    else:
                                        stderr_filtered = filter_malloc_warnings(cleanup_result.stderr)
                                        terminal_output.append(f"[WARNING] Cleanup had issues (this is okay if instance is new): {stderr_filtered[:200]}")
                                else:
                                    terminal_output.append(f"[INFO] SSH info not available yet - cleanup will happen after SSH connection is established")
                                
                                st.session_state[phase1_init_key] = True
                                st.session_state[terminal_output_key] = terminal_output
                                st.rerun()
                            else:
                                terminal_output.append(f"[WARNING] No instance ID found - skipping cleanup")
                                st.session_state[phase1_init_key] = True
                        except Exception as e:
                            error_msg = str(e)
                            terminal_output.append(f"[WARNING] Phase 1 initialization error (non-critical): {error_msg[:200]}")
                            st.session_state[phase1_init_key] = True
                            st.session_state[terminal_output_key] = terminal_output
                    
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
                        ssh_command = f"ssh -p {ssh_port} root@{ssh_host}"
                        st.success(f"🔐 **SSH Connection Info:** `{ssh_command}`{port_note}")
                        
                        # Show workspace folder path
                        st.info(f"📁 **Workspace Folder:** `/workspace` (training files: `/workspace/data`, output: `/workspace/output/training`)")

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
                                    
                                    # Reload job to get updated status
                                    updated_job_status = training_manager.get_job_status(instance_id)
                                    if updated_job_status and not updated_job_status.get("error"):
                                        # Update active_job with fresh data
                                        active_job.update(updated_job_status)
                                    
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
                                            
                                            # Reload job to get updated status
                                            updated_job_status = training_manager.get_job_status(instance_id)
                                            if updated_job_status and not updated_job_status.get("error"):
                                                # Update active_job with fresh data
                                                active_job.update(updated_job_status)
                                            
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
                        confirm_key_phase1 = f"confirm_redo_phase_1_{active_job.get('instance_id')}"
                        if confirm_key_phase1 not in st.session_state:
                            st.session_state[confirm_key_phase1] = False
                        
                        if st.session_state[confirm_key_phase1]:
                            if st.button("✅ Confirm Redo Phase 1", key="confirm_redo_phase_1_btn", type="primary"):
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
                                        st.session_state[confirm_key_phase1] = False
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
                            if st.button("❌ Cancel", key="cancel_redo_phase_1"):
                                st.session_state[confirm_key_phase1] = False
                                st.rerun()
                        else:
                            if st.button("🔄 Redo Phase", key="retry_phase_1", type="secondary", help="Reset Phase 1 validation and clear terminal output"):
                                st.session_state[confirm_key_phase1] = True
                                st.rerun()
                
                # Phase 2: Upload File
                elif current_phase == 2:
                    # Ensure datetime is available in this scope
                    from datetime import datetime
                    
                    st.markdown(f"### {phases[2]['icon']} Phase 2: {phases[2]['name']}")
                    st.caption(phases[2]['description'])
                    
                    # Initialize Phase 2: Check SSH and create directories (show in terminal, don't auto-upload)
                    phase2_init_key = f"phase2_init_{active_job.get('instance_id')}"
                    
                    # Verify files are actually uploaded (single job mode)
                    job_queue = active_job.get("job_queue")
                    if job_queue and len(job_queue) > 0:
                        all_package_infos = active_job.get("all_package_infos", [])
                        # If we have a job but all_package_infos is missing or empty, files haven't been uploaded
                        if not all_package_infos or len(all_package_infos) == 0:
                            if active_job.get("files_uploaded", False):
                                active_job["files_uploaded"] = False
                                training_manager._save_job(active_job)
                    
                    if not active_job.get("files_uploaded", False) and phase2_init_key not in st.session_state:
                        try:
                            instance_id = active_job.get("instance_id")
                            if instance_id:
                                # Get SSH info - prefer saved SSH details from job over API
                                ssh_host = active_job.get("ssh_host")
                                
                                # Check for SSH port override first (user-specified port takes precedence)
                                ssh_port_override = active_job.get("ssh_port_override")
                                if ssh_port_override:
                                    ssh_port = ssh_port_override
                                else:
                                    ssh_port = active_job.get("ssh_port", 22)
                                
                                # If not in job, get from API
                                if not ssh_host:
                                    job_status = training_manager.get_job_status(instance_id)
                                    ssh_host = job_status.get("ssh_host")
                                    api_ssh_port = job_status.get("ssh_port", 22)
                                    
                                    # Use override port if set, otherwise use API port
                                    if ssh_port_override:
                                        ssh_port = ssh_port_override
                                    else:
                                        ssh_port = api_ssh_port
                                    
                                    if ssh_host:
                                        active_job["ssh_host"] = ssh_host
                                        active_job["ssh_port"] = ssh_port
                                        training_manager._save_job(active_job)
                                
                                if ssh_host:
                                    # Clean up old files before starting Phase 2
                                    terminal_output.append(f"[SSH] Phase 2: Cleaning up old training files...")
                                    cleanup_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        "cd /workspace/data && "
                                        "echo 'Cleaning up old training files...' && "
                                        "# Remove old numbered files (from old naming scheme) and any stale active files "
                                        "# Explicitly remove _0 files (old naming scheme) and all other numbered files "
                                        "rm -f axolotl_config_0.yaml training_data_0.jsonl && "
                                        "rm -f axolotl_config_*.yaml training_data_*.jsonl axolotl_config.yaml training_data.jsonl && "
                                        "echo 'Old files cleaned up' && "
                                        "ls -la /workspace/data/ 2>/dev/null | grep -E '(training_data|axolotl_config)' || echo 'No training files found (clean)'"
                                    ]
                                    cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                    if cleanup_result.returncode == 0:
                                        terminal_output.append(f"[SSH] Old files cleaned up successfully")
                                        if cleanup_result.stdout:
                                            for line in cleanup_result.stdout.strip().split("\n"):
                                                if line.strip() and "Cleaning up" not in line and "cleaned up" not in line:
                                                    terminal_output.append(f"[SSH]   {line}")
                                    else:
                                        terminal_output.append(f"[WARNING] Cleanup had issues, but continuing...")
                                
                                # Check for SSH port override (user-specified port takes precedence)
                                ssh_port_override = active_job.get("ssh_port_override")
                                if not ssh_host:
                                    job_status = training_manager.get_job_status(instance_id)
                                    ssh_host = job_status.get("ssh_host")
                                    api_ssh_port = job_status.get("ssh_port", 22)
                                    
                                    # Use override port if set, otherwise use API port
                                    if ssh_port_override:
                                        ssh_port = ssh_port_override
                                    else:
                                        ssh_port = api_ssh_port
                                    
                                    # Save to job for future use
                                    if ssh_host:
                                        active_job["ssh_host"] = ssh_host
                                        active_job["ssh_port"] = ssh_port
                                        training_manager._save_job(active_job)
                                elif ssh_port_override:
                                    # SSH host exists, but check if we need to use port override
                                    ssh_port = ssh_port_override
                                else:
                                    ssh_port = active_job.get("ssh_port", 22)
                                
                                if ssh_host:
                                    # Append to existing terminal output (don't clear it)
                                    if not terminal_output:
                                        terminal_output = []
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Phase 2...")
                                    terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                    
                                    # Clean up old files before starting Phase 2
                                    terminal_output.append(f"[SSH] Cleaning up old training files...")
                                    cleanup_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        "cd /workspace/data && "
                                        "echo 'Cleaning up old training files...' && "
                                        "# Remove old numbered files (from old naming scheme) and any stale active files "
                                        "# Explicitly remove _0 files (old naming scheme) and all other numbered files "
                                        "rm -f axolotl_config_0.yaml training_data_0.jsonl && "
                                        "rm -f axolotl_config_*.yaml training_data_*.jsonl axolotl_config.yaml training_data.jsonl && "
                                        "echo 'Old files cleaned up' && "
                                        "ls -la /workspace/data/ 2>/dev/null | grep -E '(training_data|axolotl_config)' || echo 'No training files found (clean)'"
                                    ]
                                    cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                    if cleanup_result.returncode == 0:
                                        terminal_output.append(f"[SSH] Old files cleaned up successfully")
                                        if cleanup_result.stdout:
                                            for line in cleanup_result.stdout.strip().split("\n"):
                                                if line.strip() and "Cleaning up" not in line and "cleaned up" not in line and "No training files" not in line:
                                                    terminal_output.append(f"[SSH]   {line}")
                                    else:
                                        terminal_output.append(f"[WARNING] Cleanup had issues, but continuing...")
                                    
                                    # Test SSH connection
                                    import subprocess
                                    test_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        "echo 'SSH connection test'"
                                    ]
                                    test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
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
                                            "ssh"
                                        ] + get_ssh_base_options(ssh_port) + [
                                            f"root@{ssh_host}",
                                            "echo 'Deleting output directory...' && rm -rf /workspace/output 2>&1 && echo 'Output directory deleted' && mkdir -p /workspace/data && mkdir -p /workspace/output/training && echo 'Directories recreated' && ls -la /workspace/output/ 2>&1 && echo 'Output directories forcefully deleted and recreated'"
                                        ]
                                        mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                        if mkdir_result.returncode == 0:
                                            terminal_output.append(f"[SSH] ✓ Output directories forcefully deleted and recreated")
                                            stdout_filtered = filter_malloc_warnings(mkdir_result.stdout)
                                            if stdout_filtered.strip():
                                                for line in stdout_filtered.strip().split("\n"):
                                                    if line.strip() and "forcefully" not in line.lower() and "Deleting" not in line:
                                                        terminal_output.append(f"[SSH]   {line}")
                                            
                                            # Check for onstart errors in Phase 2
                                            try:
                                                check_onstart_cmd = [
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                    f"root@{ssh_host}",
                                                    "tail -30 /var/log/onstart.log 2>/dev/null || tail -30 /tmp/onstart.log 2>/dev/null || echo 'no_onstart_log'"
                                                ]
                                                onstart_check = subprocess.run(check_onstart_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
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
                                            stdout_filtered = filter_malloc_warnings(mkdir_result.stdout)
                                            terminal_output.append(f"[ERROR] Directory deletion/creation failed (return code: {mkdir_result.returncode})")
                                            if stderr_filtered.strip():
                                                terminal_output.append(f"[ERROR] stderr: {stderr_filtered[:300]}")
                                            if stdout_filtered.strip():
                                                terminal_output.append(f"[ERROR] stdout: {stdout_filtered[:300]}")
                                            if retry < 2:
                                                terminal_output.append(f"[SSH] Retrying directory creation (attempt {retry + 1}/3)...")
                                            else:
                                                terminal_output.append(f"[SSH] Directory creation failed after 3 attempts")
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
                                        
                                        if not all_package_infos or len(all_package_infos) == 0:
                                            terminal_output.append(f"[INFO] Preparing training package...")
                                            all_package_infos = []
                                            
                                            # Only one job now
                                            if len(job_queue) > 0:
                                                job_item = job_queue[0]
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
                                            
                                            # Save package info to job
                                            active_job["all_package_infos"] = all_package_infos
                                            training_manager._save_job(active_job)
                                            terminal_output.append(f"[INFO] Prepared training package")
                                        
                                        # Log what will be uploaded (single job)
                                        if len(job_queue) > 0:
                                            job_item = job_queue[0]  # Only one job now
                                            job_yaml = job_item.get("yaml_filename")
                                            file_group = job_item.get("file_group", [])
                                            
                                            terminal_output.append(f"[PRE-UPLOAD] Files to be uploaded:")
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
                                                if attached_yaml:
                                                    yaml_info = f" (YAML: {attached_yaml})"
                                                else:
                                                    yaml_info = " (using stock/default config)"
                                                terminal_output.append(f"[PRE-UPLOAD]   • {filename} ({file_type}, {file_size:,} bytes){yaml_info}")
                                            
                                            # Show what will be uploaded
                                            job_package = all_package_infos[0]  # Only one job
                                            job_config_path = job_package.get('config_path')
                                            job_dataset_path = job_package.get('dataset_path')
                                            
                                            terminal_output.append(f"[PRE-UPLOAD] Processed files:")
                                            if job_config_path:
                                                job_config_file = Path(job_config_path)
                                                if job_config_file.exists():
                                                    config_size = job_config_file.stat().st_size
                                                    terminal_output.append(f"[PRE-UPLOAD]   • Config: {job_config_file.name} ({config_size:,} bytes)")
                                                    terminal_output.append(f"[PRE-UPLOAD]     → /workspace/data/axolotl_config.yaml")
                                            
                                            if job_dataset_path:
                                                job_dataset_file = Path(job_dataset_path)
                                                if job_dataset_file.exists():
                                                    dataset_size = job_dataset_file.stat().st_size
                                                    terminal_output.append(f"[PRE-UPLOAD]   • Training Data: {job_dataset_file.name} ({dataset_size:,} bytes)")
                                                    terminal_output.append(f"[PRE-UPLOAD]     → /workspace/data/training_data.jsonl")
                                        
                                        terminal_output.append(f"[PRE-UPLOAD] ========================================")
                                        
                                        # Get SSH info - prefer saved SSH details from job over API
                                        ssh_host = active_job.get("ssh_host")
                                        
                                        # Check for SSH port override first (user-specified port takes precedence)
                                        ssh_port_override = active_job.get("ssh_port_override")
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                        else:
                                            ssh_port = active_job.get("ssh_port", 22)
                                        
                                        # If not in job, get from API
                                        if not ssh_host:
                                            job_status = training_manager.get_job_status(instance_id)
                                            ssh_host = job_status.get("ssh_host")
                                            api_ssh_port = job_status.get("ssh_port", 22)
                                            
                                            # Use override port if set, otherwise use API port
                                            if ssh_port_override:
                                                ssh_port = ssh_port_override
                                            else:
                                                ssh_port = api_ssh_port
                                            
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
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "test -d /workspace/data && test -d /workspace/output/training && echo 'exists' || echo 'missing'"
                                            ]
                                            import subprocess
                                            check_result = subprocess.run(check_dirs_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
                                            
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
                                                        "echo 'Deleting output directory...' && rm -rf /workspace/output 2>&1 && echo 'Output directory deleted' && mkdir -p /workspace/data && mkdir -p /workspace/output/training && echo 'Directories recreated' && ls -la /workspace/output/ 2>&1 && echo 'Output directories forcefully deleted and recreated'"
                                                    ]
                                                    mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                                    if mkdir_result.returncode == 0:
                                                        terminal_output.append(f"[SSH] ✓ Output directories forcefully deleted and recreated")
                                                        stdout_filtered = filter_malloc_warnings(mkdir_result.stdout)
                                                        if stdout_filtered.strip():
                                                            for line in stdout_filtered.strip().split("\n"):
                                                                if line.strip() and "forcefully" not in line.lower() and "Deleting" not in line:
                                                                    terminal_output.append(f"[SSH]   {line}")
                                                        mkdir_success = True
                                                        break
                                                    else:
                                                        stderr_filtered = filter_malloc_warnings(mkdir_result.stderr)
                                                        stdout_filtered = filter_malloc_warnings(mkdir_result.stdout)
                                                        terminal_output.append(f"[ERROR] Directory deletion/creation failed (return code: {mkdir_result.returncode})")
                                                        if stderr_filtered.strip():
                                                            terminal_output.append(f"[ERROR] stderr: {stderr_filtered[:300]}")
                                                        if stdout_filtered.strip():
                                                            terminal_output.append(f"[ERROR] stdout: {stdout_filtered[:300]}")
                                                        if retry < 2:
                                                            terminal_output.append(f"[SSH] Retrying directory creation (attempt {retry + 1}/3)...")
                                                        else:
                                                            terminal_output.append(f"[SSH] Directory creation failed after 3 attempts")
                                                if not mkdir_success:
                                                    terminal_output.append(f"[WARNING] Directory creation failed. Uploads may fail.")
                                            else:
                                                # Directories already exist
                                                terminal_output.append(f"[SSH] Directories already exist, proceeding with upload...")
                                            
                                            # Upload single job (no numbered filenames needed)
                                            all_uploads_successful = True
                                            
                                            if len(job_queue) > 0 and len(all_package_infos) > 0:
                                                job_item = job_queue[0]  # Only one job
                                                job_package = all_package_infos[0]  # Only one package
                                                job_yaml = job_item.get("yaml_filename")
                                                file_group = job_item.get("file_group", [])
                                                
                                                terminal_output.append(f"[UPLOAD] --- Uploading Job ---")
                                                
                                                # Upload config file
                                                job_config_path = job_package.get('config_path')
                                                job_config_uploaded = False
                                                
                                                if job_config_path:
                                                    job_config_file = Path(job_config_path)
                                                    if job_config_file.exists():
                                                        # Safety check: Fix YAML adapter issue if present
                                                        import yaml
                                                        try:
                                                            with open(job_config_file, 'r') as f:
                                                                config = yaml.safe_load(f) or {}
                                                            
                                                            # Ensure adapter: 'lora' is set if lora_* parameters exist (required for LoRA mode)
                                                            # BUT only if there's no existing adapter path (for incremental training)
                                                            # The path issue is prevented by NOT setting lora_model_dir to output_dir
                                                            has_lora_params = config.get("lora_r") is not None and config.get("lora_alpha") is not None
                                                            current_adapter = config.get("adapter")
                                                            # Only set adapter: "lora" if:
                                                            # 1. LoRA params exist
                                                            # 2. No adapter is set, OR adapter is None/null
                                                            # 3. Adapter is NOT already set to a path (for incremental training)
                                                            if has_lora_params and (not current_adapter or current_adapter is None):
                                                                # Check if adapter might be a path (incremental training) - don't override it
                                                                if not (isinstance(current_adapter, str) and current_adapter.startswith("/")):
                                                                    config["adapter"] = "lora"
                                                                    terminal_output.append(f"[FIX] Added 'adapter: lora' to config (required for LoRA mode)")
                                                                    # Save the config back with adapter added
                                                                    with open(job_config_file, 'w') as f:
                                                                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                                                                else:
                                                                    terminal_output.append(f"[INFO] Keeping existing adapter path for incremental training: {current_adapter}")
                                                            
                                                            # Auto-adjust for small datasets to prevent empty batch errors
                                                            dataset_path = job_package.get('dataset_path')
                                                            if dataset_path and Path(dataset_path).exists():
                                                                try:
                                                                    with open(dataset_path, 'r') as f:
                                                                        total_examples = sum(1 for line in f if line.strip())
                                                                    
                                                                    if total_examples > 0:
                                                                        min_eval_examples = 2
                                                                        val_set_size = config.get("val_set_size", 0.1)
                                                                        
                                                                        if total_examples * val_set_size < min_eval_examples:
                                                                            if total_examples < 50:
                                                                                config["val_set_size"] = 0.0
                                                                                terminal_output.append(f"[FIX] Dataset has only {total_examples} examples. Disabled validation set.")
                                                                            elif total_examples < 200:
                                                                                if config.get("sample_packing", False):
                                                                                    config["sample_packing"] = False
                                                                                    terminal_output.append(f"[FIX] Dataset has {total_examples} examples. Disabled sample_packing.")
                                                                                min_val_size = min_eval_examples / total_examples
                                                                                if val_set_size < min_val_size:
                                                                                    config["val_set_size"] = min_val_size
                                                                                    terminal_output.append(f"[FIX] Adjusted val_set_size to {min_val_size:.3f} to ensure at least {min_eval_examples} eval examples.")
                                                                            else:
                                                                                min_val_size = min_eval_examples / total_examples
                                                                                if val_set_size < min_val_size:
                                                                                    config["val_set_size"] = min_val_size
                                                                                    terminal_output.append(f"[FIX] Adjusted val_set_size to {min_val_size:.3f} to ensure at least {min_eval_examples} eval examples.")
                                                                except Exception as e:
                                                                    pass
                                                            
                                                            # Write fixed config back if any changes were made
                                                            with open(job_config_file, 'w') as f:
                                                                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                                                        except Exception as e:
                                                            terminal_output.append(f"[WARNING] Could not verify YAML config: {str(e)[:100]}")
                                                        
                                                        config_size = job_config_file.stat().st_size
                                                        remote_config_name = "axolotl_config.yaml"  # Single job, no suffix
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
                                                            
                                                            scp_config_result = subprocess.run(scp_config_cmd, capture_output=True, text=True, timeout=300, input="yes\n")
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
                                                
                                                # Upload dataset file
                                                job_dataset_path = job_package.get('dataset_path')
                                                job_dataset_uploaded = False
                                                
                                                if job_dataset_path:
                                                    job_dataset_file = Path(job_dataset_path)
                                                    if job_dataset_file.exists():
                                                        dataset_size = job_dataset_file.stat().st_size
                                                        remote_dataset_name = "training_data.jsonl"  # Single job, no suffix
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
                                                        
                                                        # Retry logic for SCP uploads
                                                        dataset_upload_success = False
                                                        for retry in range(3):
                                                            if retry > 0:
                                                                wait_time = 2 ** retry
                                                                terminal_output.append(f"[SCP] Retry {retry}/3 after {wait_time}s wait...")
                                                                time.sleep(wait_time)
                                                            
                                                            scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300, input="yes\n")
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
                                                    terminal_output.append(f"[WARNING] No dataset path specified")
                                                    all_uploads_successful = False
                                                
                                                if not (job_config_uploaded and job_dataset_uploaded):
                                                    all_uploads_successful = False
                                            else:
                                                terminal_output.append(f"[ERROR] Job queue or package info missing")
                                                all_uploads_successful = False
                                            
                                            # Check if upload succeeded
                                            if all_uploads_successful:
                                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] ===== Upload Summary =====")
                                                terminal_output.append(f"[SUCCESS] Job uploaded successfully!")
                                                
                                                job_yaml = job_item.get("yaml_filename")
                                                file_group = job_item.get("file_group", [])
                                                yaml_desc = job_yaml if job_yaml else "stock/default config"
                                                terminal_output.append(f"[SUCCESS] {len(file_group)} file(s), YAML: {yaml_desc}")
                                                terminal_output.append(f"[SUCCESS]   Config: axolotl_config.yaml")
                                                terminal_output.append(f"[SUCCESS]   Data: training_data.jsonl")
                                                
                                                terminal_output.append(f"[SUCCESS] ========================================")
                                                terminal_output.append(f"[INFO] Phase 2 complete! Ready to proceed to training.")
                                                active_job["files_uploaded"] = True
                                                # Keep status as 'validated' - don't change to 'running' until training actually starts
                                                # Phase determination logic will advance to Phase 3 when status is 'validated' and files_uploaded is True
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
                                    
                                    # Final save and rerun - ensure terminal is updated
                                    st.session_state[terminal_output_key] = list(terminal_output)  # Ensure it's a fresh copy
                                    st.rerun()
                                except Exception as e:
                                    error_msg = str(e)
                                    # Ensure terminal_output is available in exception handler
                                    if terminal_output_key not in st.session_state:
                                        st.session_state[terminal_output_key] = []
                                    terminal_output = list(st.session_state[terminal_output_key])
                                    terminal_output.append(f"[ERROR] {error_msg}")
                                    st.session_state[terminal_output_key] = list(terminal_output)  # Save fresh copy
                                    st.error(f"Error: {error_msg}")
                                    st.rerun()  # Rerun even on error to show the error message
                        else:
                            # Files uploaded - show status
                            st.info("✅ Files uploaded")
                    
                    with col2:
                        confirm_key_phase2 = f"confirm_redo_phase_2_{active_job.get('instance_id')}"
                        if confirm_key_phase2 not in st.session_state:
                            st.session_state[confirm_key_phase2] = False
                        
                        if st.session_state[confirm_key_phase2]:
                            if st.button("✅ Confirm Redo Phase 2", key="confirm_redo_phase_2_btn", type="primary"):
                                try:
                                    # Clear terminal before redoing phase
                                    st.session_state[terminal_output_key] = []
                                    terminal_output = []
                                    
                                    instance_id = active_job.get("instance_id")
                                    if instance_id:
                                        terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Redoing Phase 2 - cleaning up uploaded files...")
                                        
                                        # Clear package info to force regeneration (will detect latest versions for incremental training)
                                        active_job["all_package_infos"] = []
                                        active_job["package_info"] = None
                                        training_manager._save_job(active_job)
                                        terminal_output.append(f"[INFO] Cleared package info - will regenerate to detect latest versions")
                                        
                                        # Get SSH info - prefer saved SSH details from job over API
                                        ssh_host = active_job.get("ssh_host")
                                        # Check for SSH port override first (user-specified port takes precedence)
                                        ssh_port_override = active_job.get("ssh_port_override")
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                        else:
                                            ssh_port = active_job.get("ssh_port", 22)

                                        # If not in job, get from API
                                        if not ssh_host:
                                            job_status = training_manager.get_job_status(instance_id)
                                            ssh_host = job_status.get("ssh_host")
                                            api_ssh_port = job_status.get("ssh_port", 22)

                                            # Use override port if set, otherwise use API port
                                            if ssh_port_override:
                                                ssh_port = ssh_port_override
                                            else:
                                                ssh_port = api_ssh_port
                                            
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
                                                cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
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
                                                mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
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
                                        st.session_state[confirm_key_phase2] = False
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
                            if st.button("❌ Cancel", key="cancel_redo_phase_2"):
                                st.session_state[confirm_key_phase2] = False
                                st.rerun()
                        else:
                            if st.button("🔄 Redo Phase", key="retry_phase_2", type="secondary", help="Reset Phase 2, delete uploaded files, and clear terminal output"):
                                st.session_state[confirm_key_phase2] = True
                                st.rerun()
                    
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
                    
                    # Initialize Phase 3: Ensure files are active on instance (single job, no index needed)
                    phase3_init_key = f"phase3_init_{active_job.get('instance_id')}"
                    if phase3_init_key not in st.session_state:
                        try:
                            import subprocess
                            instance_id = active_job.get("instance_id")
                            # Get SSH info - prefer saved SSH details from job over API
                            ssh_host = active_job.get("ssh_host")
                            # Check for SSH port override first (user-specified port takes precedence)
                            ssh_port_override = active_job.get("ssh_port_override")
                            if ssh_port_override:
                                ssh_port = ssh_port_override
                            else:
                                ssh_port = active_job.get("ssh_port", 22)
                            
                            # If not in job, get from API (proactively retrieve SSH info)
                            if not ssh_host:
                                if not terminal_output:
                                    terminal_output = []
                                terminal_output.append(f"[INFO] SSH info not in job - retrieving from API...")
                                try:
                                    job_status = training_manager.get_job_status(instance_id)
                                    ssh_host = job_status.get("ssh_host")
                                    api_ssh_port = job_status.get("ssh_port", 22)
                                    
                                    # Use override port if set, otherwise use API port
                                    if ssh_port_override:
                                        ssh_port = ssh_port_override
                                    else:
                                        ssh_port = api_ssh_port
                                    
                                    # Save to job for future use
                                    if ssh_host:
                                        active_job["ssh_host"] = ssh_host
                                        active_job["ssh_port"] = ssh_port
                                        training_manager._save_job(active_job)
                                        port_source = "override" if ssh_port_override else "API"
                                        terminal_output.append(f"[SUCCESS] Retrieved SSH info from API: {ssh_host}:{ssh_port} ({port_source})")
                                    else:
                                        terminal_output.append(f"[WARNING] SSH host not available from API - instance may still be initializing")
                                        terminal_output.append(f"[INFO] The 'Start Training' button will appear once SSH info is available")
                                except Exception as e:
                                    terminal_output.append(f"[WARNING] Could not retrieve SSH info from API: {str(e)}")
                            
                            if ssh_host:
                                if not terminal_output:
                                    terminal_output = []
                                
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Phase 3...")
                                
                                # Clean up old numbered files (single job uses active files, no numbered files needed)
                                terminal_output.append(f"[SSH] Cleaning up old numbered training files (keeping active files)...")
                                cleanup_cmd = [
                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                    f"root@{ssh_host}",
                                    f"cd /workspace/data && "
                                    f"echo 'Cleaning up old numbered training files...' && "
                                    f"# Verify active files exist before cleaning "
                                    f"if [ ! -f axolotl_config.yaml ] || [ ! -f training_data.jsonl ]; then "
                                    f"  echo 'WARNING: Active files missing - skipping cleanup to preserve any files'; "
                                    f"else "
                                    f"  # Remove ONLY numbered files (pattern requires underscore+number, won't match active files) "
                                    f"  rm -f axolotl_config_0.yaml training_data_0.jsonl 2>/dev/null || true && "
                                    f"  for f in axolotl_config_[0-9]*.yaml training_data_[0-9]*.jsonl; do "
                                    f"    if [ -f \"$f\" ]; then "
                                    f"      rm -f \"$f\" && echo \"Removed old numbered file: $f\"; "
                                    f"    fi; "
                                    f"  done; "
                                    f"  echo 'Removed numbered files (kept active files)'; "
                                    f"fi && "
                                    f"echo 'Cleanup complete' && "
                                    f"ls -la /workspace/data/ 2>/dev/null | grep -E '(training_data|axolotl_config)' || echo 'No training files found'"
                                ]
                                cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                if cleanup_result.returncode == 0:
                                    terminal_output.append(f"[SSH] Old files cleaned up successfully")
                                    if cleanup_result.stdout:
                                        for line in cleanup_result.stdout.strip().split("\n"):
                                            if line.strip() and "Cleaning up" not in line and "complete" not in line and "No training files" not in line:
                                                terminal_output.append(f"[SSH]   {line}")
                                else:
                                    terminal_output.append(f"[WARNING] Cleanup had issues, but continuing...")
                                
                                terminal_output.append(f"[SSH] Verifying training files are ready...")
                                
                                # Check for active files (single job, no suffix needed)
                                expected_config_name = "axolotl_config.yaml"
                                expected_data_name = "training_data.jsonl"
                                
                                check_files_cmd = [
                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                    f"root@{ssh_host}",
                                    f"cd /workspace/data && "
                                    f"if [ -f {expected_config_name} ] && [ -f {expected_data_name} ]; then "
                                    f"  config_size=$(stat -c%s {expected_config_name} 2>/dev/null || echo 0) "
                                    f"  data_size=$(stat -c%s {expected_data_name} 2>/dev/null || echo 0) "
                                    f"  if [ $config_size -gt 0 ] && [ $data_size -gt 0 ]; then "
                                    f"    echo 'FILES_READY'; "
                                    f"    ls -lh {expected_config_name} {expected_data_name}; "
                                    f"  else "
                                    f"    echo 'FILES_EMPTY'; "
                                    f"  fi "
                                    f"else "
                                    f"  echo 'FILES_MISSING'; "
                                    f"fi"
                                ]
                                check_result = subprocess.run(check_files_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
                                if "FILES_MISSING" in check_result.stdout:
                                    terminal_output.append(f"[ERROR] Training files ({expected_config_name}, {expected_data_name}) are missing!")
                                    st.error("❌ Training files are missing. Please re-upload files in Phase 2.")
                                    st.stop()
                                elif "FILES_EMPTY" in check_result.stdout:
                                    terminal_output.append(f"[ERROR] Training files exist but are empty!")
                                    st.error("❌ Training files are empty. Please re-upload files in Phase 2.")
                                    st.stop()
                                else:
                                    terminal_output.append(f"[SUCCESS] Training files are verified and ready")
                                    if check_result.stdout:
                                        for line in check_result.stdout.strip().split("\n"):
                                            if line.strip() and "FILES" not in line:
                                                terminal_output.append(f"[SSH]   {line}")
                                    
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
                                        f"    # First, verify the dataset file exists and config points to it\n"
                                        f"    expected_dataset = '/workspace/data/training_data.jsonl'\n"
                                        f"    if not os.path.exists(expected_dataset):\n"
                                        f"        print(f'ERROR: Dataset file not found: {{expected_dataset}}')\n"
                                        f"        sys.exit(1)\n"
                                        f"    \n"
                                        f"    # Check if datasets config points to the correct file\n"
                                        f"    datasets = config.get('datasets', [])\n"
                                        f"    dataset_found = False\n"
                                        f"    for dataset in datasets:\n"
                                        f"        if isinstance(dataset, dict):\n"
                                        f"            dataset_path = dataset.get('path', '')\n"
                                        f"            if dataset_path == expected_dataset or dataset_path.endswith('training_data.jsonl'):\n"
                                        f"                dataset_found = True\n"
                                        f"                print(f'Dataset path found in config: {{dataset_path}}')\n"
                                        f"                break\n"
                                        f"    \n"
                                        f"    if not dataset_found and datasets:\n"
                                        f"        print(f'WARNING: Dataset path in config may not match expected path: {{expected_dataset}}')\n"
                                        f"        print(f'Current datasets config: {{datasets}}')\n"
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
                                        f"    # Ensure adapter: 'lora' is set if lora_* parameters exist (required for LoRA mode)\n"
                                        f"    # BUT only if there's no existing adapter path (for incremental training)\n"
                                        f"    # The path issue is prevented by NOT setting lora_model_dir to output_dir\n"
                                        f"    has_lora_params = config.get('lora_r') is not None and config.get('lora_alpha') is not None\n"
                                        f"    current_adapter = config.get('adapter')\n"
                                        f"    # Only set adapter: 'lora' if no adapter is set AND it's not a path (incremental training)\n"
                                        f"    if has_lora_params and (not current_adapter or current_adapter is None):\n"
                                        f"        # Don't override if adapter is already set to a path (incremental training)\n"
                                        f"        if not (isinstance(current_adapter, str) and current_adapter.startswith('/')):\n"
                                        f"            config['adapter'] = 'lora'\n"
                                        f"            fixed = True\n"
                                        f"            print('Added adapter: lora (required for LoRA mode)')\n"
                                        f"        else:\n"
                                        f"            print(f'Keeping existing adapter path for incremental training: {{current_adapter}}')\n"
                                        f"    \n"
                                        f"    # CRITICAL: Remove lora_model_dir if it's set to output_dir (causes Axolotl to try loading from there)\n"
                                        f"    # lora_model_dir should only be set to a valid adapter path (for incremental training)\n"
                                        f"    output_dir = config.get('output_dir', '')\n"
                                        f"    lora_model_dir = config.get('lora_model_dir', '')\n"
                                        f"    adapter_val = config.get('adapter', '')\n"
                                        f"    \n"
                                        f"    # If lora_model_dir is set to output_dir, always remove it (unless there's a valid adapter path)\n"
                                        f"    if lora_model_dir == output_dir:\n"
                                        f"        # Only keep it if adapter is set to a valid path (incremental training)\n"
                                        f"        if not adapter_val or not isinstance(adapter_val, str) or not adapter_val.startswith('/'):\n"
                                        f"            del config['lora_model_dir']\n"
                                        f"            fixed = True\n"
                                        f"            print('Removed lora_model_dir (was pointing to output_dir - causes adapter loading errors)')\n"
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
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        fix_config_remote_cmd
                                    ]
                                    fix_config_result = subprocess.run(fix_config_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                    
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
                                    
                                    # Update package_info (single job, no index needed)
                                    all_package_infos = active_job.get("all_package_infos", [])
                                    if all_package_infos and len(all_package_infos) > 0:
                                        active_job["package_info"] = all_package_infos[0]  # Only one job
                                        # Update YAML config in package_info for display
                                        job_queue = active_job.get("job_queue", [])
                                        if job_queue and len(job_queue) > 0:
                                            current_job = job_queue[0]  # Only one job
                                            if current_job.get("yaml_path"):
                                                from pathlib import Path
                                                yaml_filename = Path(current_job.get("yaml_path")).name
                                                active_job["package_info"]["yaml_config"] = yaml_filename
                                            else:
                                                active_job["package_info"]["yaml_config"] = None
                                        training_manager._save_job(active_job)
                                
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
                    phase3_yaml_debug_key = f"phase3_yaml_debug_{active_job.get('instance_id')}"
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
                            
                            job_queue = active_job.get("job_queue", [])
                            if job_queue and len(job_queue) > 0:
                                current_job = job_queue[0]  # Only one job
                                expected_yaml_from_queue = current_job.get("yaml_filename")
                                expected_yaml_path = current_job.get("yaml_path")
                                terminal_output.append(f"[YAML DEBUG] Expected YAML from queue: {expected_yaml_from_queue if expected_yaml_from_queue else 'None'}")
                                terminal_output.append(f"[YAML DEBUG] Expected YAML path: {expected_yaml_path if expected_yaml_path else 'None'}")
                            else:
                                terminal_output.append(f"[YAML DEBUG] Checking package_info")
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
                            # Check for SSH port override first (user-specified port takes precedence)
                            ssh_port_override = active_job.get("ssh_port_override")
                            if ssh_port_override:
                                ssh_port = ssh_port_override
                            else:
                                ssh_port = active_job.get("ssh_port", 22)
                            
                            # If not in job, get from API
                            if not ssh_host:
                                job_status = training_manager.get_job_status(instance_id)
                                ssh_host = job_status.get("ssh_host")
                                api_ssh_port = job_status.get("ssh_port", 22)
                                
                                # Use override port if set, otherwise use API port
                                if ssh_port_override:
                                    ssh_port = ssh_port_override
                                else:
                                    ssh_port = api_ssh_port
                                
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
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        f"test -f /workspace/data/{yaml_to_check} && echo 'found' || echo 'not_found'"
                                    ]
                                    yaml_check_result = subprocess.run(check_yaml_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
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
                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                            
                            # Run comprehensive LoRA verification
                            terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] === LoRA Configuration Verification ===")
                            
                            if ssh_host:
                                lora_verify_cmd = [
                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                    f"root@{ssh_host}",
                                    "cd /workspace/data && python3 << 'PYTHON_EOF'\n"
                                    "import yaml\n"
                                    "import os\n"
                                    "import json\n"
                                    "try:\n"
                                    "    with open('axolotl_config.yaml', 'r') as f:\n"
                                    "        config = yaml.safe_load(f) or {}\n"
                                    "    \n"
                                    "    print('LORA_VERIFICATION:')\n"
                                    "    \n"
                                    "    # Check LoRA parameters\n"
                                    "    lora_r = config.get('lora_r')\n"
                                    "    lora_alpha = config.get('lora_alpha')\n"
                                    "    lora_dropout = config.get('lora_dropout')\n"
                                    "    lora_target_modules = config.get('lora_target_modules', [])\n"
                                    "    adapter_val = config.get('adapter')\n"
                                    "    base_model = config.get('base_model', '')\n"
                                    "    \n"
                                    "    print(f'adapter={adapter_val}')\n"
                                    "    print(f'base_model={base_model}')\n"
                                    "    print(f'lora_r={lora_r}')\n"
                                    "    print(f'lora_alpha={lora_alpha}')\n"
                                    "    print(f'lora_dropout={lora_dropout}')\n"
                                    "    print(f'lora_target_modules_count={len(lora_target_modules) if lora_target_modules else 0}')\n"
                                    "    \n"
                                    "    # Check if adapter: 'lora' is set (required for new LoRA training)\n"
                                    "    is_new_lora = (adapter_val == 'lora' and lora_r is not None)\n"
                                    "    print(f'is_new_lora={is_new_lora}')\n"
                                    "    \n"
                                    "    # Check if adapter path exists (for incremental training)\n"
                                    "    is_incremental = False\n"
                                    "    adapter_exists = False\n"
                                    "    adapter_config_exists = False\n"
                                    "    base_model_match = False\n"
                                    "    \n"
                                    "    if adapter_val and isinstance(adapter_val, str) and adapter_val.startswith('/'):\n"
                                    "        is_incremental = True\n"
                                    "        adapter_exists = os.path.exists(adapter_val)\n"
                                    "        adapter_config_path = os.path.join(adapter_val, 'adapter_config.json')\n"
                                    "        adapter_config_exists = os.path.exists(adapter_config_path)\n"
                                    "        \n"
                                    "        print(f'adapter_path={adapter_val}')\n"
                                    "        print(f'adapter_exists={adapter_exists}')\n"
                                    "        print(f'adapter_config_exists={adapter_config_exists}')\n"
                                    "        \n"
                                    "        # Verify base model matches\n"
                                    "        if adapter_config_exists:\n"
                                    "            try:\n"
                                    "                with open(adapter_config_path, 'r') as f:\n"
                                    "                    adapter_config = json.load(f)\n"
                                    "                    adapter_base = adapter_config.get('base_model_name', '')\n"
                                    "                    base_model_match = (adapter_base == base_model) if adapter_base and base_model else False\n"
                                    "                    print(f'adapter_base_model={adapter_base}')\n"
                                    "                    print(f'base_model_match={base_model_match}')\n"
                                    "            except Exception as e:\n"
                                    "                print(f'adapter_config_read_error={e}')\n"
                                    "    \n"
                                    "    # Determine LoRA status\n"
                                    "    lora_enabled = is_new_lora or (is_incremental and adapter_config_exists)\n"
                                    "    print(f'lora_enabled={lora_enabled}')\n"
                                    "    \n"
                                    "    # Check lora_model_dir (should NOT be set to output_dir)\n"
                                    "    lora_model_dir = config.get('lora_model_dir', '')\n"
                                    "    output_dir = config.get('output_dir', '')\n"
                                    "    lora_model_dir_issue = (lora_model_dir == output_dir)\n"
                                    "    print(f'lora_model_dir={lora_model_dir}')\n"
                                    "    print(f'lora_model_dir_issue={lora_model_dir_issue}')\n"
                                    "    \n"
                                    "except Exception as e:\n"
                                    "    print(f'ERROR: {str(e)}')\n"
                                    "PYTHON_EOF"
                                ]
                                lora_verify_result = subprocess.run(lora_verify_cmd, capture_output=True, text=True, timeout=20, input="yes\n")
                                if lora_verify_result.returncode == 0:
                                    verify_output = lora_verify_result.stdout.strip()
                                    if "LORA_VERIFICATION:" in verify_output:
                                        for line in verify_output.split("\n"):
                                            if "=" in line and "LORA_VERIFICATION" not in line:
                                                if "lora_enabled=True" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ✓ LoRA is enabled")
                                                elif "lora_enabled=False" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ✗ LoRA is NOT enabled")
                                                elif "is_new_lora=True" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ✓ New LoRA training detected (adapter: lora)")
                                                elif "is_incremental=True" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ✓ Incremental training detected")
                                                elif "adapter_config_exists=True" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ✓ Adapter config found (adapter is valid)")
                                                elif "base_model_match=True" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ✓ Base model matches between adapter and config")
                                                elif "base_model_match=False" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ⚠ Base model mismatch detected")
                                                elif "lora_model_dir_issue=True" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ⚠ lora_model_dir points to output_dir (may cause issues)")
                                                elif "adapter=lora" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ✓ Config has adapter: lora (NEW training - no prior weights)")
                                                elif "adapter=" in line and "/" in line:
                                                    adapter_path = line.split("=")[1].strip()
                                                    terminal_output.append(f"[LoRA VERIFY] ✓ Config has adapter path: {adapter_path} (INCREMENTAL training - prior weights attached)")
                                                elif "is_incremental=True" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ✓ Incremental training mode detected (prior weights will be loaded)")
                                                elif "is_incremental=False" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ⚠ New training mode (no prior weights - training from base model)")
                                                elif "adapter_exists=True" in line:
                                                    terminal_output.append(f"[LoRA VERIFY] ✓ Adapter directory exists on remote")
                                                elif "adapter_exists=False" in line and "is_incremental=True" in verify_output:
                                                    terminal_output.append(f"[LoRA VERIFY] ⚠ WARNING: Incremental training configured but adapter directory NOT FOUND")
                                else:
                                    terminal_output.append(f"[LoRA VERIFY] Error running verification: {lora_verify_result.stderr[:200]}")
                            
                            terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] === End LoRA Verification ===")
                            
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
                    
                    # Terminal output area (scrollable) - CREATE PLACEHOLDER FIRST
                    st.markdown("#### Terminal Output")
                    terminal_placeholder = st.empty()
                    
                    # Action buttons - BUTTONS MUST BE DEFINED BEFORE TERMINAL DISPLAY
                    # This ensures button handlers run and update session state BEFORE terminal reads
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                    with col1:
                        # Check if training is already running - check both status AND actual processes
                        training_status = active_job.get("training_status", {})
                        status_indicates_running = training_status.get("status") in ["training", "preprocessing", "completed"]
                        
                        # Also check if processes are actually running via SSH (more reliable)
                        processes_actually_running = False
                        ssh_host = active_job.get("ssh_host")
                        ssh_port = active_job.get("ssh_port", 22)
                        ssh_port_override = active_job.get("ssh_port_override")
                        if ssh_port_override:
                            ssh_port = ssh_port_override
                        
                        if ssh_host:
                            try:
                                check_process_cmd = [
                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                    f"root@{ssh_host}",
                                    "ps aux | grep -E '(accelerate|axolotl.cli.train)' | grep -v grep | wc -l"
                                ]
                                process_check_result = subprocess.run(check_process_cmd, capture_output=True, text=True, timeout=10, input="yes\n")
                                if process_check_result.returncode == 0:
                                    process_count = int(process_check_result.stdout.strip())
                                    processes_actually_running = process_count > 0
                            except:
                                # If SSH check fails, fall back to status check
                                pass
                        
                        # Training is running if either status says so OR processes are actually running
                        training_is_running = status_indicates_running or processes_actually_running
                        
                        # Show "Start Training" button if training hasn't started
                        if not training_is_running:
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
                                    
                                    # Get SSH info - check for port override first
                                    ssh_host = active_job.get("ssh_host")
                                    ssh_port_override = active_job.get("ssh_port_override")
                                    
                                    # Use port override if provided, otherwise use saved port
                                    if ssh_port_override:
                                        ssh_port = ssh_port_override
                                        terminal_output.append(f"[INFO] Using SSH port override: {ssh_port}")
                                    else:
                                        ssh_port = active_job.get("ssh_port", 22)
                                    
                                    if not ssh_host:
                                        job_status = training_manager.get_job_status(instance_id)
                                        ssh_host = job_status.get("ssh_host")
                                        api_ssh_port = job_status.get("ssh_port", 22)
                                        
                                        # Use port override if provided, otherwise use API port
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                            terminal_output.append(f"[INFO] Using SSH port override: {ssh_port} (instead of API port: {api_ssh_port})")
                                        else:
                                            ssh_port = api_ssh_port
                                        
                                        if ssh_host:
                                            active_job["ssh_host"] = ssh_host
                                            active_job["ssh_port"] = ssh_port
                                            training_manager._save_job(active_job)
                                    
                                    if ssh_host:
                                        # Find and embed most recent version weights before training
                                        terminal_output.append(f"[LOCAL] Looking for most recent version weights to embed...")
                                        from utils.model_manager import ModelManager
                                        model_manager = ModelManager()
                                        latest_version = model_manager.get_most_recent_version(model_name)
                                        
                                        if latest_version:
                                            weights_path = model_manager.get_version_weights_path(model_name, latest_version)
                                            if weights_path and weights_path.exists():
                                                terminal_output.append(f"[PRIOR WEIGHTS] ✓ Found version {latest_version} weights at: {weights_path}")
                                                terminal_output.append(f"[PRIOR WEIGHTS] Uploading and embedding previous version weights...")
                                                
                                                # Upload weights to instance and update config
                                                # First, upload the adapter directory
                                                import subprocess
                                                upload_weights_cmd = [
                                                    "scp", "-r", "-P", str(ssh_port), 
                                                    "-o", "StrictHostKeyChecking=no", 
                                                    "-o", "ConnectTimeout=30",
                                                    str(weights_path),
                                                    f"root@{ssh_host}:/workspace/previous_adapter"
                                                ]
                                                upload_result = subprocess.run(upload_weights_cmd, capture_output=True, text=True, timeout=300, input="yes\n")
                                                if upload_result.returncode == 0:
                                                    terminal_output.append(f"[PRIOR WEIGHTS] ✓ Uploaded previous version weights to /workspace/previous_adapter")
                                                    if upload_result.stdout:
                                                        terminal_output.append(f"[PRIOR WEIGHTS] Upload output: {upload_result.stdout[:200]}")
                                                    
                                                    # Update config file to use the adapter
                                                    update_config_cmd = [
                                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                        f"root@{ssh_host}",
                                                        f"cd /workspace/data && "
                                                        f"python3 << 'PYTHON_EOF'\n"
                                                        f"import yaml\n"
                                                        f"import os\n"
                                                        f"try:\n"
                                                        f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                                        f"        config = yaml.safe_load(f) or {{}}\n"
                                                        f"    \n"
                                                        f"    # Set adapter path to uploaded weights\n"
                                                        f"    config['adapter'] = '/workspace/previous_adapter'\n"
                                                        f"    \n"
                                                        f"    with open('axolotl_config.yaml', 'w') as f:\n"
                                                        f"        yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                        f"    \n"
                                                        f"    print('SUCCESS: Updated config with adapter path: /workspace/previous_adapter')\n"
                                                        f"except Exception as e:\n"
                                                        f"    print(f'ERROR: Failed to update config: {{str(e)}}')\n"
                                                        f"    import sys\n"
                                                        f"    sys.exit(1)\n"
                                                        f"PYTHON_EOF"
                                                    ]
                                                    update_result = subprocess.run(update_config_cmd, capture_output=True, text=True, timeout=30)
                                                    if update_result.returncode == 0:
                                                        terminal_output.append(f"[SUCCESS] Embedded version {latest_version} weights in config")
                                                        if update_result.stdout:
                                                            for line in update_result.stdout.strip().split("\n"):
                                                                if line.strip():
                                                                    terminal_output.append(f"[SSH]   {line}")
                                                    else:
                                                        terminal_output.append(f"[WARNING] Failed to update config with adapter path: {update_result.stderr[:200]}")
                                                else:
                                                    terminal_output.append(f"[WARNING] Failed to upload previous version weights: {upload_result.stderr[:200]}")
                                                    terminal_output.append(f"[INFO] Will train from base model instead")
                                            else:
                                                terminal_output.append(f"[INFO] Version {latest_version} exists but weights not found. Training from base model.")
                                        else:
                                            terminal_output.append(f"[INFO] No previous versions found. Training from base model.")
                                        
                                        # CRITICAL: Ensure files are activated before starting training
                                        # Files should already be axolotl_config.yaml and training_data.jsonl (no suffix needed for single job)
                                        terminal_output.append(f"[SSH] Ensuring training files are ready before starting training...")
                                        
                                        # Check for active files (single job, no suffix)
                                        check_active_cmd = [
                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                            f"root@{ssh_host}",
                                            f"cd /workspace/data && "
                                            f"if [ -f axolotl_config.yaml ] && [ -f training_data.jsonl ]; then "
                                            f"  # Check file sizes to ensure they're not empty "
                                            f"  config_size=$(stat -c%s axolotl_config.yaml 2>/dev/null || echo 0) "
                                            f"  data_size=$(stat -c%s training_data.jsonl 2>/dev/null || echo 0) "
                                            f"  if [ $config_size -gt 0 ] && [ $data_size -gt 0 ]; then "
                                            f"    echo 'ACTIVE_EXISTS'; "
                                            f"    ls -lh axolotl_config.yaml training_data.jsonl; "
                                            f"  else "
                                            f"    echo 'ACTIVE_EMPTY'; "
                                            f"    echo 'Config size: '$config_size', Data size: '$data_size; "
                                            f"  fi "
                                            f"else "
                                            f"  echo 'ACTIVE_MISSING'; "
                                            f"  ls -la axolotl_config.yaml training_data.jsonl 2>&1 || echo 'Files not found'; "
                                            f"fi"
                                        ]
                                        check_result = subprocess.run(check_active_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
                                        if "ACTIVE_MISSING" in check_result.stdout:
                                            terminal_output.append(f"[ERROR] Training files (axolotl_config.yaml, training_data.jsonl) are missing!")
                                            if check_result.stdout:
                                                for line in check_result.stdout.strip().split("\n"):
                                                    if line.strip() and "ACTIVE" not in line:
                                                        terminal_output.append(f"[SSH]   {line}")
                                            terminal_output.append(f"[ERROR] Cannot start training without active files.")
                                            st.error("❌ Training files are missing. Please re-upload files in Phase 2.")
                                            st.stop()
                                        elif "ACTIVE_EMPTY" in check_result.stdout:
                                            terminal_output.append(f"[ERROR] Job 1 files exist but are empty!")
                                            if check_result.stdout:
                                                for line in check_result.stdout.strip().split("\n"):
                                                    if line.strip() and "ACTIVE" not in line:
                                                        terminal_output.append(f"[SSH]   {line}")
                                            terminal_output.append(f"[ERROR] Cannot start training with empty files. Please re-upload files in Phase 2.")
                                            st.error("❌ Training files are empty. Please re-upload files in Phase 2.")
                                            st.stop()
                                        else:
                                            terminal_output.append(f"[SUCCESS] Training files are ready (axolotl_config.yaml, training_data.jsonl)")
                                            if check_result.stdout:
                                                for line in check_result.stdout.strip().split("\n"):
                                                    if line.strip() and "ACTIVE" not in line:
                                                        terminal_output.append(f"[SSH]   {line}")
                                        
                                        st.session_state[terminal_output_key] = terminal_output
                                        terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                        
                                        # Check if files exist
                                        import subprocess
                                        check_files_cmd = [
                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                            f"root@{ssh_host}",
                                            "test -f /workspace/data/axolotl_config.yaml && test -f /workspace/data/training_data.jsonl && echo 'files_exist' || echo 'files_missing'"
                                        ]
                                        files_check = subprocess.run(check_files_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
                                        
                                        if "files_exist" in files_check.stdout:
                                            terminal_output.append(f"[SSH] Training files found - setting up environment...")
                                            
                                            # Check if there's already a training process running
                                            # Check for existing processes
                                            check_processes_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | head -5 || echo 'no_processes'"
                                            ]
                                            check_processes_result = subprocess.run(check_processes_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
                                            
                                            has_existing_process = "no_processes" not in check_processes_result.stdout and check_processes_result.stdout.strip()
                                            
                                            # ALWAYS kill any existing processes before starting training
                                            # This ensures a clean state even if process detection missed something
                                            if has_existing_process:
                                                terminal_output.append(f"[WARNING] Found existing training process. This will be stopped to start fresh training.")
                                            else:
                                                terminal_output.append(f"[SSH] No existing training processes detected - ensuring clean state...")
                                            
                                            # Always kill processes to ensure clean state
                                            kill_all_training_processes(ssh_host, ssh_port, terminal_output)
                                            
                                            # Clean output directory
                                            terminal_output.append(f"[SSH] Cleaning output directory...")
                                            cleanup_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "rm -rf /workspace/output/training/* && echo 'Output directory cleaned'"
                                            ]
                                            cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                            if cleanup_result.returncode == 0:
                                                terminal_output.append(f"[SSH] Output directory cleaned")
                                            
                                            # FINAL VERIFICATION: Check files exist and are valid right before starting training
                                            terminal_output.append(f"[SSH] Final verification: Checking files exist and are valid before starting training...")
                                            final_verify_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                f"cd /workspace/data && "
                                                f"echo '=== FINAL FILE VERIFICATION ===' && "
                                                f"if [ ! -f axolotl_config.yaml ]; then "
                                                f"  echo 'ERROR: axolotl_config.yaml missing'; exit 1; "
                                                f"fi && "
                                                f"if [ ! -f training_data.jsonl ]; then "
                                                f"  echo 'ERROR: training_data.jsonl missing'; exit 1; "
                                                f"fi && "
                                                f"config_size=$(stat -c%s axolotl_config.yaml 2>/dev/null || echo 0) && "
                                                f"data_size=$(stat -c%s training_data.jsonl 2>/dev/null || echo 0) && "
                                                f"if [ $config_size -eq 0 ]; then "
                                                f"  echo 'ERROR: axolotl_config.yaml is empty'; exit 1; "
                                                f"fi && "
                                                f"if [ $data_size -eq 0 ]; then "
                                                f"  echo 'ERROR: training_data.jsonl is empty'; exit 1; "
                                                f"fi && "
                                                f"echo 'Files verified:' && "
                                                f"ls -lh axolotl_config.yaml training_data.jsonl && "
                                                f"echo 'File sizes: config=$config_size bytes, data=$data_size bytes' && "
                                                f"echo '=== VERIFICATION PASSED ==='"
                                            ]
                                            final_verify_result = subprocess.run(final_verify_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
                                            if final_verify_result.returncode != 0:
                                                terminal_output.append(f"[ERROR] Final verification FAILED!")
                                                if final_verify_result.stdout:
                                                    for line in final_verify_result.stdout.strip().split("\n"):
                                                        if line.strip():
                                                            terminal_output.append(f"[SSH]   {line}")
                                                if final_verify_result.stderr:
                                                    stderr_filtered = filter_malloc_warnings(final_verify_result.stderr)
                                                    for line in stderr_filtered.strip().split("\n"):
                                                        if line.strip():
                                                            terminal_output.append(f"[SSH]   stderr: {line}")
                                                terminal_output.append(f"[ERROR] Cannot start training - files are missing or invalid!")
                                                st.error("❌ Training files are missing or invalid. Please re-upload files in Phase 2.")
                                                st.session_state[terminal_output_key] = terminal_output
                                                st.stop()
                                            else:
                                                terminal_output.append(f"[SUCCESS] Final verification passed - files are ready")
                                                if final_verify_result.stdout:
                                                    for line in final_verify_result.stdout.strip().split("\n"):
                                                        if line.strip() and "VERIFICATION" not in line:
                                                            terminal_output.append(f"[SSH]   {line}")
                                            
                                            # Verify config file points to correct dataset path
                                            terminal_output.append(f"[SSH] Verifying config file points to correct dataset path...")
                                            verify_config_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                f"cd /workspace/data && "
                                                f"python3 << 'PYTHON_EOF'\n"
                                                f"import yaml\n"
                                                f"import os\n"
                                                f"import sys\n"
                                                f"try:\n"
                                                f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                                f"        config = yaml.safe_load(f) or {{}}\n"
                                                f"    \n"
                                                f"    expected_dataset = '/workspace/data/training_data.jsonl'\n"
                                                f"    \n"
                                                f"    # Check if file exists\n"
                                                f"    if not os.path.exists(expected_dataset):\n"
                                                f"        print(f'ERROR: Dataset file does not exist: {{expected_dataset}}')\n"
                                                f"        sys.exit(1)\n"
                                                f"    \n"
                                                f"    # Check file size\n"
                                                f"    file_size = os.path.getsize(expected_dataset)\n"
                                                f"    if file_size == 0:\n"
                                                f"        print(f'ERROR: Dataset file is empty: {{expected_dataset}}')\n"
                                                f"        sys.exit(1)\n"
                                                f"    \n"
                                                f"    # Check if datasets config points to the correct file\n"
                                                f"    datasets = config.get('datasets', [])\n"
                                                f"    dataset_found = False\n"
                                                f"    for dataset in datasets:\n"
                                                f"        if isinstance(dataset, dict):\n"
                                                f"            dataset_path = dataset.get('path', '')\n"
                                                f"            if dataset_path == expected_dataset:\n"
                                                f"                dataset_found = True\n"
                                                f"                print(f'✓ Dataset path in config matches: {{dataset_path}}')\n"
                                                f"                break\n"
                                                f"            elif dataset_path and dataset_path.endswith('training_data.jsonl'):\n"
                                                f"                print(f'WARNING: Dataset path in config is {{dataset_path}}, expected {{expected_dataset}}')\n"
                                                f"    \n"
                                                f"    if not dataset_found and datasets:\n"
                                                f"        print(f'WARNING: Dataset path in config may not match expected path')\n"
                                                f"        print(f'Expected: {{expected_dataset}}')\n"
                                                f"        print(f'Config datasets: {{datasets}}')\n"
                                                f"    \n"
                                                f"    print(f'✓ Config verification passed')\n"
                                                f"    print(f'✓ Dataset file exists: {{expected_dataset}}')\n"
                                                f"    print(f'✓ Dataset file size: {{file_size}} bytes')\n"
                                                f"except Exception as e:\n"
                                                f"    print(f'ERROR: Config verification failed: {{str(e)}}')\n"
                                                f"    sys.exit(1)\n"
                                                f"PYTHON_EOF"
                                            ]
                                            verify_config_result = subprocess.run(verify_config_cmd, capture_output=True, text=True, timeout=15, input="yes\n")
                                            if verify_config_result.returncode != 0:
                                                terminal_output.append(f"[ERROR] Config verification FAILED!")
                                                if verify_config_result.stdout:
                                                    for line in verify_config_result.stdout.strip().split("\n"):
                                                        if line.strip():
                                                            terminal_output.append(f"[SSH]   {line}")
                                                if verify_config_result.stderr:
                                                    stderr_filtered = filter_malloc_warnings(verify_config_result.stderr)
                                                    for line in stderr_filtered.strip().split("\n"):
                                                        if line.strip():
                                                            terminal_output.append(f"[SSH]   stderr: {line}")
                                                terminal_output.append(f"[ERROR] Cannot start training - config or dataset file is invalid!")
                                                st.error("❌ Config or dataset file verification failed. Please re-upload files in Phase 2.")
                                                st.session_state[terminal_output_key] = terminal_output
                                                st.stop()
                                            else:
                                                terminal_output.append(f"[SUCCESS] Config verification passed")
                                                if verify_config_result.stdout:
                                                    for line in verify_config_result.stdout.strip().split("\n"):
                                                        if line.strip() and "ERROR" not in line:
                                                            terminal_output.append(f"[SSH]   {line}")
                                            
                                            # CRITICAL: Ensure sample_packing is disabled before starting training
                                            # This prevents multipack sampler IndexError issues
                                            terminal_output.append(f"[SSH] Ensuring sample_packing is disabled...")
                                            fix_sample_packing_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                f"cd /workspace/data && "
                                                f"python3 << 'PYTHON_EOF'\n"
                                                f"import yaml\n"
                                                f"import sys\n"
                                                f"try:\n"
                                                f"    with open('axolotl_config.yaml', 'r') as f:\n"
                                                f"        config = yaml.safe_load(f) or {{}}\n"
                                                f"    \n"
                                                f"    # Always disable sample_packing to prevent multipack sampler errors\n"
                                                f"    if config.get('sample_packing', True):  # Default to True if not set\n"
                                                f"        config['sample_packing'] = False\n"
                                                f"        with open('axolotl_config.yaml', 'w') as f:\n"
                                                f"            yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                f"        print('SUCCESS: Disabled sample_packing')\n"
                                                f"    else:\n"
                                                f"        print('OK: sample_packing already disabled')\n"
                                                f"except Exception as e:\n"
                                                f"    print(f'ERROR: {{str(e)}}')\n"
                                                f"    sys.exit(1)\n"
                                                f"PYTHON_EOF"
                                            ]
                                            fix_sample_packing_result = subprocess.run(fix_sample_packing_cmd, capture_output=True, text=True, timeout=30)
                                            if fix_sample_packing_result.returncode == 0:
                                                if "SUCCESS" in fix_sample_packing_result.stdout:
                                                    terminal_output.append(f"[SUCCESS] sample_packing disabled in config")
                                                elif "OK" in fix_sample_packing_result.stdout:
                                                    terminal_output.append(f"[INFO] sample_packing already disabled")
                                                else:
                                                    terminal_output.append(f"[INFO] Config check: {fix_sample_packing_result.stdout.strip()}")
                                            else:
                                                terminal_output.append(f"[WARNING] Could not verify sample_packing setting: {fix_sample_packing_result.stderr[:200]}")
                                                # Continue anyway - the error detection will catch it if it fails
                                            
                                            st.session_state[terminal_output_key] = terminal_output
                                            
                                            # Use the same method as Redo Phase 3 - simpler and more reliable
                                            # Use a here-document to pass the command more reliably through SSH
                                            # This avoids issues with quote escaping and command length
                                            
                                            terminal_output.append(f"[SSH] Starting training...")
                                            terminal_output.append(f"[DEBUG] Using SSH port: {ssh_port} for training command")

                                            # Build HF token export section - same as Redo Phase 3
                                            token_exports = ""
                                            if hf_token and isinstance(hf_token, str) and hf_token.strip():
                                                hf_token_clean = hf_token.strip().replace('\n', '').replace('\r', '').replace(';', '').replace('`', '')
                                                if hf_token_clean:
                                                    hf_token_escaped = hf_token_clean.replace("'", "'\"'\"'")
                                                    token_exports = f"export HF_TOKEN='{hf_token_escaped}'\nexport HUGGING_FACE_HUB_TOKEN='{hf_token_escaped}'\n"
                                                    terminal_output.append(f"[INFO] HF token prepared for export (length: {len(hf_token_clean)})")
                                            else:
                                                terminal_output.append(f"[WARNING] No HF token found - gated models (e.g., Gemma) may fail to load")
                                            
                                            # Use the same here-document approach as Redo Phase 3
                                            ssh_command = f"""bash << 'REMOTE_SCRIPT'
set -e
{token_exports}# Ensure output directory exists before training starts
mkdir -p /workspace/output/training
cd /workspace/axolotl
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
                                            
                                            # Log the axolotl command that will be executed
                                            terminal_output.append(f"[DEBUG] Axolotl training command that will be executed:")
                                            terminal_output.append(f"[DEBUG]   accelerate launch -m axolotl.cli.train /workspace/data/axolotl_config.yaml")
                                            
                                            training_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                                f"root@{ssh_host}",
                                                ssh_command
                                            ]
                                            terminal_output.append(f"[DEBUG] Executing training command via SSH on port: {ssh_port}")
                                            terminal_output.append(f"[DEBUG] SSH command: ssh -p {ssh_port} root@{ssh_host}")
                                            
                                            try:
                                                training_result = subprocess.run(training_cmd, capture_output=True, text=True, timeout=5)
                                                terminal_output.append(f"[DEBUG] Training command return code: {training_result.returncode}")
                                                
                                                if training_result.returncode == 0:
                                                    terminal_output.append(f"[SSH] Training command executed")
                                                if training_result.stdout:
                                                    stdout_filtered = filter_malloc_warnings(training_result.stdout)
                                                    for line in stdout_filtered.strip().split("\n"):
                                                        if line.strip():
                                                                terminal_output.append(f"[SSH] {line[:200]}")
                                                else:
                                                    terminal_output.append(f"[SSH] Training command sent (return code: {training_result.returncode}, checking process status...)")
                                                    if training_result.stdout:
                                                        stdout_filtered = filter_malloc_warnings(training_result.stdout)
                                                        for line in stdout_filtered.strip().split("\n"):
                                                            if line.strip():
                                                                terminal_output.append(f"[STDOUT] {line[:200]}")
                                                if training_result.stderr:
                                                    stderr_filtered = filter_malloc_warnings(training_result.stderr)
                                                    for line in stderr_filtered.strip().split("\n"):
                                                        if line.strip():
                                                                terminal_output.append(f"[STDERR] {line[:200]}")
                                            except subprocess.TimeoutExpired:
                                                # Timeout is expected - the process is running in background
                                                terminal_output.append(f"[SSH] Training command sent (timeout expected for background process)")
                                            except Exception as e:
                                                terminal_output.append(f"[ERROR] Exception while executing training command: {str(e)}")
                                                import traceback
                                                terminal_output.append(f"[ERROR] Traceback: {traceback.format_exc()[:300]}")
                                            
                                            # Wait a moment and then check if the process actually started
                                                import time
                                            time.sleep(3)
                                                
                                            # Quick check to see if log file is being created
                                            try:
                                                quick_log_check_cmd = [
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                    f"root@{ssh_host}",
                                                    "test -f /workspace/output/training/training.log && (wc -l /workspace/output/training/training.log && tail -20 /workspace/output/training/training.log) || echo 'log_not_created'"
                                                ]
                                                quick_log_check = subprocess.run(quick_log_check_cmd, capture_output=True, text=True, timeout=10)
                                                if "log_not_created" not in quick_log_check.stdout:
                                                    log_output = quick_log_check.stdout.strip()
                                                    lines = log_output.split('\n')
                                                    if len(lines) > 0:
                                                        log_lines = lines[0]
                                                        terminal_output.append(f"[SSH] Log file created: {log_lines}")
                                                        if len(lines) > 1:
                                                            log_content = '\n'.join(lines[1:])
                                                            if log_content.strip():
                                                                terminal_output.append(f"[SSH] Recent log content (showing more lines):")
                                                                for line in log_content.strip().split('\n'):
                                                                    if line.strip():
                                                                        terminal_output.append(f"[SSH]   {line[:200]}")
                                                else:
                                                    terminal_output.append(f"[WARNING] Log file not created yet - training may not have started")
                                            except subprocess.TimeoutExpired:
                                                terminal_output.append(f"[WARNING] Log check timed out")
                                            except Exception as e:
                                                terminal_output.append(f"[WARNING] Could not check log file: {str(e)}")
                                            
                                            # Always check if process started, regardless of command result
                                            terminal_output.append(f"[SSH] Verifying training process started...")
                                            
                                            # Wait a moment and verify training process is running
                                            try:
                                                verify_cmd = [
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                        terminal_output.append(f"[SUCCESS] Training started successfully!")
                                                        
                                                        # Update job status
                                                        active_job["training_status"] = {"status": "training"}
                                                        training_manager._save_job(active_job)
                                                    else:
                                                        terminal_output.append(f"[WARNING] Training command executed but process not found yet")
                                                        
                                                        # Check training log for errors
                                                        check_log_cmd = [
                                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                            f"root@{ssh_host}",
                                                            "tail -50 /workspace/output/training/training.log 2>/dev/null || echo 'no_log'"
                                                        ]
                                                        log_check = subprocess.run(check_log_cmd, capture_output=True, text=True, timeout=15)
                                                        if "no_log" not in log_check.stdout:
                                                            log_output = filter_malloc_warnings(log_check.stdout)
                                                            terminal_output.append(f"[WARNING] Last log output (showing more lines for debugging):")
                                                            for line in log_output.strip().split("\n"):
                                                                if line.strip():
                                                                    terminal_output.append(f"[LOG] {line[:200]}")
                                                        else:
                                                            terminal_output.append(f"[WARNING] Training log file not found at /workspace/output/training/training.log")
                                                        
                                                    terminal_output.append(f"[INFO] Training may still be starting. Use 'Check Training Status' to verify.")
                                            except subprocess.TimeoutExpired:
                                                terminal_output.append(f"[WARNING] Process verification timed out - training may still be starting")
                                                terminal_output.append(f"[INFO] Check training status again in a few moments")
                                            except Exception as e:
                                                terminal_output.append(f"[WARNING] Could not verify training process: {str(e)}")
                                            terminal_output.append(f"[INFO] Training command was sent - check status to verify training started")
                                            
                                            # Training verification is already handled above - no duplicate code needed
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
                    
                    with col2:
                        # Always show "Check Training Status" button - needed for queue monitoring
                        status_msg = training_status.get("status", "unknown")
                        if status_msg == "training":
                            button_label = "🔄 Check Training Status"
                        elif status_msg == "completed":
                            button_label = "✅ Check Training Status"
                        else:
                            button_label = "🔍 Check Training Status"
                        
                        # Check if button was clicked or manually triggered (e.g., by force advance)
                        button_clicked = st.button(button_label, key="check_training_status")
                        trigger_flag = st.session_state.get("trigger_status_check", False)
                        should_check = button_clicked or trigger_flag
                        
                        if should_check:
                            # Clear the trigger flag
                            if "trigger_status_check" in st.session_state:
                                del st.session_state["trigger_status_check"]
                            
                            try:
                                instance_id = active_job.get("instance_id")
                                
                                if not instance_id:
                                    st.error("No instance ID found in job.")
                                    return
                                
                                # Ensure terminal_output is initialized from session state
                                if terminal_output_key not in st.session_state:
                                    st.session_state[terminal_output_key] = []
                                # Get terminal_output from session state and make a copy to modify
                                terminal_output = list(st.session_state[terminal_output_key])
                                
                                terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ Button clicked - checking training status...")
                                # Save immediately - use list() to create a fresh copy
                                # CRITICAL: Force a new list object so Streamlit detects the change
                                # Create a completely new list to break any reference issues
                                fresh_output = [line for line in terminal_output]  # List comprehension creates new list
                                
                                # CRITICAL: Increment version counter FIRST before saving
                                # This ensures the version changes, which triggers the dependency
                                terminal_version_key = f"{terminal_output_key}_version"
                                current_version = st.session_state.get(terminal_version_key, 0)
                                new_version = current_version + 1
                                st.session_state[terminal_version_key] = new_version
                                
                                # Now save the terminal output
                                st.session_state[terminal_output_key] = fresh_output
                                
                                # Force Streamlit to recognize the change by updating a timestamp
                                st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                # Debug: Verify it was saved
                                if st.session_state.get("debug_terminal", False):
                                    terminal_output.append(f"[DEBUG] Terminal output saved. Key: {terminal_output_key}, Length: {len(terminal_output)}")
                                    st.session_state[terminal_output_key] = list(terminal_output)
                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                
                                # Initialize SSH check results (will be updated if SSH is available)
                                ssh_files_exist = None
                                ssh_training_running = None
                                training_logs_content = None
                                debug_log_content = None
                                training_error = None
                                
                                # Get SSH info - prefer saved SSH details from job over API
                                # Always reload active_job first to get latest saved SSH info
                                # Reload from active_jobs list (which should have latest saved data after rerun)
                                for idx, job in enumerate(active_jobs):
                                    if job.get("instance_id") == instance_id:
                                        active_job = job
                                        break
                                
                                ssh_host = active_job.get("ssh_host") if active_job else None
                                # Check for SSH port override first (user-specified port takes precedence)
                                ssh_port_override = active_job.get("ssh_port_override") if active_job else None
                                if ssh_port_override:
                                    ssh_port = ssh_port_override
                                else:
                                    ssh_port = active_job.get("ssh_port", 22) if active_job else 22
                                
                                # If not in job, get from API
                                if not ssh_host:
                                    try:
                                        job_status = training_manager.get_job_status(instance_id)
                                        ssh_host = job_status.get("ssh_host")
                                        api_ssh_port = job_status.get("ssh_port", 22)
                                        
                                        # Use override port if set, otherwise use API port
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                        else:
                                            ssh_port = api_ssh_port
                                        
                                        # Save to job for future use
                                        if ssh_host and active_job:
                                            active_job["ssh_host"] = ssh_host
                                            active_job["ssh_port"] = ssh_port
                                            training_manager._save_job(active_job)
                                            # Update active_jobs list too
                                            for idx, job in enumerate(active_jobs):
                                                if job.get("instance_id") == instance_id:
                                                    active_jobs[idx] = active_job
                                                    break
                                    except Exception as e:
                                        terminal_output.append(f"[WARNING] Could not retrieve SSH info from API: {str(e)[:200]}")
                                
                                if ssh_host:
                                    terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                    st.session_state[terminal_output_key] = list(terminal_output)  # Save as fresh copy
                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                    
                                    # First test SSH connection with a simple command
                                    terminal_output.append(f"[SSH] Testing SSH connection...")
                                    test_connection_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                        f"root@{ssh_host}",
                                        "echo 'SSH connection successful'"
                                    ]
                                    try:
                                        test_result = subprocess.run(test_connection_cmd, capture_output=True, text=True, timeout=15)
                                        if test_result.returncode == 0:
                                            terminal_output.append(f"[SSH] ✓ SSH connection successful")
                                        else:
                                            terminal_output.append(f"[SSH] ⚠️ SSH connection test failed (returncode: {test_result.returncode})")
                                            if test_result.stderr:
                                                terminal_output.append(f"[SSH] Error: {test_result.stderr[:300]}")
                                            terminal_output.append(f"[SSH] Attempting to continue anyway...")
                                    except subprocess.TimeoutExpired:
                                        terminal_output.append(f"[SSH] ⚠️ SSH connection test timed out")
                                        terminal_output.append(f"[SSH] This may indicate network issues or the instance is not ready")
                                        terminal_output.append(f"[SSH] Attempting to continue anyway...")
                                    except Exception as e:
                                        terminal_output.append(f"[SSH] ⚠️ SSH connection test error: {str(e)[:200]}")
                                        terminal_output.append(f"[SSH] Attempting to continue anyway...")
                                    
                                    st.session_state[terminal_output_key] = list(terminal_output)
                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                    
                                    # Check for output directory and its contents
                                    terminal_output.append(f"[SSH] Checking output directory...")
                                    terminal_output.append(f"[DEBUG] SSH command: ssh -p {ssh_port} root@{ssh_host} 'ls -la /workspace/output/training'")
                                    check_output_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                        f"root@{ssh_host}",
                                        """bash -c '
                                        echo "=== /workspace/output/training ==="
                                        ls -la /workspace/output/training 2>/dev/null || echo "Directory does not exist"
                                        echo ""
                                        echo "=== Checking for checkpoints ==="
                                        find /workspace/output/training -type d -name "checkpoint-*" 2>/dev/null | head -5 || echo "No checkpoints found"
                                        echo ""
                                        echo "=== Checking for adapter directory ==="
                                        ls -la /workspace/output/training/adapter 2>/dev/null || echo "No adapter directory"
                                        echo ""
                                        echo "=== Checking for any files in output/training ==="
                                        find /workspace/output/training -type f 2>/dev/null | head -10 || echo "No files found"
                                        echo ""
                                        echo "=== Checking Axolotl output locations ==="
                                        find /workspace/axolotl -name "checkpoint-*" -type d 2>/dev/null | head -3 || echo "No checkpoints in axolotl dir"
                                        find /workspace/axolotl -name "adapter*" -type d 2>/dev/null | head -3 || echo "No adapters in axolotl dir"
                                        '"""
                                    ]
                                    try:
                                        output_result = subprocess.run(check_output_cmd, capture_output=True, text=True, timeout=15)
                                        terminal_output.append(f"[DEBUG] Output directory check returncode: {output_result.returncode}")
                                        if output_result.stdout:
                                            terminal_output.append(f"[DEBUG] Output directory stdout: {output_result.stdout[:300]}")
                                        if output_result.stderr:
                                            terminal_output.append(f"[DEBUG] Output directory stderr: {output_result.stderr[:200]}")
                                    except subprocess.TimeoutExpired as e:
                                        terminal_output.append(f"[SSH] ⚠️ Output directory check timed out")
                                        terminal_output.append(f"[SSH] SSH connection may be slow or unstable")
                                        terminal_output.append(f"[DEBUG] Timeout exception: {str(e)}")
                                        st.session_state[terminal_output_key] = list(terminal_output)
                                        st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                        output_result = type('obj', (object,), {'stdout': 'no_output', 'returncode': 1})()
                                    except Exception as e:
                                        terminal_output.append(f"[SSH] ⚠️ Output directory check failed: {str(e)[:200]}")
                                        terminal_output.append(f"[DEBUG] Exception type: {type(e).__name__}")
                                        st.session_state[terminal_output_key] = list(terminal_output)
                                        st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                        output_result = type('obj', (object,), {'stdout': 'no_output', 'returncode': 1})()
                                    if "no_output" not in output_result.stdout and output_result.stdout.strip():
                                        terminal_output.append(f"[SSH] Output directory check results:")
                                        stdout_filtered = filter_malloc_warnings(output_result.stdout)
                                        # Show the full output for diagnostics
                                        for line in stdout_filtered.strip().split("\n"):
                                            if line.strip():
                                                terminal_output.append(f"[SSH] {line}")
                                        
                                        # Check if directory is truly empty (only . and ..)
                                        if "total 0" in output_result.stdout or ("checkpoint" not in output_result.stdout.lower() and "adapter" not in output_result.stdout.lower()):
                                            terminal_output.append(f"[INFO] Output directory is empty - this is normal if training hasn't saved checkpoints yet")
                                            terminal_output.append(f"[INFO] Axolotl saves checkpoints at intervals (check save_steps in config), not continuously")
                                            terminal_output.append(f"[INFO] Checkpoints and adapters will appear when Axolotl saves them (typically every N steps or at epoch end)")
                                    else:
                                        terminal_output.append(f"[SSH] Output directory not found or empty")
                                        terminal_output.append(f"[INFO] This is normal if training hasn't saved checkpoints yet")
                                    st.session_state[terminal_output_key] = list(terminal_output)  # Save after output check
                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                    
                                    # Check if training files exist
                                    terminal_output.append(f"[SSH] Checking if training files are present...")
                                    terminal_output.append(f"[DEBUG] SSH command: ssh -p {ssh_port} root@{ssh_host} 'ls -la /workspace/data/'")
                                    check_files_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        "ls -la /workspace/data/ 2>/dev/null | grep -E '(training_data|axolotl_config)' || echo 'files_missing'"
                                    ]
                                    try:
                                        files_result = subprocess.run(check_files_cmd, capture_output=True, text=True, timeout=15)
                                        terminal_output.append(f"[DEBUG] Training files check returncode: {files_result.returncode}")
                                        if files_result.stdout:
                                            terminal_output.append(f"[DEBUG] Training files stdout: {files_result.stdout[:300]}")
                                        if files_result.stderr:
                                            terminal_output.append(f"[DEBUG] Training files stderr: {files_result.stderr[:200]}")
                                    except Exception as e:
                                        terminal_output.append(f"[ERROR] Training files check failed: {str(e)}")
                                        files_result = type('obj', (object,), {'stdout': 'files_missing', 'returncode': 1})()
                                    # Capture whether files exist for diagnostics
                                    ssh_files_exist = "files_missing" not in files_result.stdout
                                    if not ssh_files_exist:
                                        terminal_output.append(f"[WARNING] Training files not found in /workspace/data/")
                                    else:
                                        stdout_filtered = filter_malloc_warnings(files_result.stdout)
                                        terminal_output.append(f"[SSH] Training files found:")
                                        
                                        # Parse files and categorize them
                                        all_files = []
                                        expected_files = []
                                        unexpected_files = []
                                        
                                        # Determine expected files based on job queue
                                        job_queue = active_job.get("job_queue", [])
                                        if job_queue:
                                            # Single job - only expect active files (no suffix)
                                            expected_files.extend(["axolotl_config.yaml", "training_data.jsonl"])
                                        else:
                                            # Single job - should be no suffix
                                            expected_files.extend(["axolotl_config.yaml", "training_data.jsonl"])
                                        
                                        # Parse file listing
                                        for line in stdout_filtered.strip().split("\n"):
                                            if line.strip() and "files_missing" not in line:
                                                # Extract filename from ls output (last field)
                                                parts = line.strip().split()
                                                if len(parts) >= 9:
                                                    filename = parts[-1]
                                                    all_files.append(filename)
                                                    
                                                    # Check if it's expected or unexpected
                                                    if filename in expected_files:
                                                        terminal_output.append(f"[SSH]   {line} (expected)")
                                                    elif filename.endswith("_0.yaml") or filename.endswith("_0.jsonl"):
                                                        # Old naming scheme - warn about it
                                                        terminal_output.append(f"[SSH]   {line} (⚠️ old naming - should be removed)")
                                                        unexpected_files.append(filename)
                                                    else:
                                                        terminal_output.append(f"[SSH]   {line}")
                                        
                                    st.session_state[terminal_output_key] = list(terminal_output)  # Save after file check
                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                    
                                    if unexpected_files:
                                        terminal_output.append(f"[WARNING] Found files with old naming scheme (_0 suffix). These should be removed.")
                                        # Automatically remove _0 files (old naming scheme)
                                        terminal_output.append(f"[SSH] Removing old _0 files...")
                                        remove_old_files_cmd = [
                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                            f"root@{ssh_host}",
                                            "cd /workspace/data && rm -f axolotl_config_0.yaml training_data_0.jsonl && echo 'Removed _0 files' || echo 'No _0 files to remove'"
                                        ]
                                        remove_result = subprocess.run(remove_old_files_cmd, capture_output=True, text=True, timeout=15)
                                        if remove_result.returncode == 0:
                                            terminal_output.append(f"[SUCCESS] Removed old _0 files")
                                        else:
                                            terminal_output.append(f"[WARNING] Could not remove _0 files: {remove_result.stderr[:200]}")
                                    
                                    # Check for training processes (ignore onstart script for existing instances)
                                    terminal_output.append(f"[SSH] Checking for training processes...")
                                    terminal_output.append(f"[DEBUG] SSH command: ssh -p {ssh_port} root@{ssh_host} 'ps aux | grep -E ...'")
                                    check_training_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | head -5 || echo 'no_training'"
                                    ]
                                    try:
                                        training_process_result = subprocess.run(check_training_cmd, capture_output=True, text=True, timeout=15)
                                        terminal_output.append(f"[DEBUG] Process check returncode: {training_process_result.returncode}")
                                        if training_process_result.stdout:
                                            terminal_output.append(f"[DEBUG] Process check stdout length: {len(training_process_result.stdout)}")
                                            terminal_output.append(f"[DEBUG] Process check stdout preview: {training_process_result.stdout[:200]}")
                                        if training_process_result.stderr:
                                            terminal_output.append(f"[DEBUG] Process check stderr: {training_process_result.stderr[:200]}")
                                    except subprocess.TimeoutExpired as e:
                                        terminal_output.append(f"[WARNING] Process check timed out after 15 seconds")
                                        terminal_output.append(f"[DEBUG] Timeout exception: {str(e)}")
                                        st.session_state[terminal_output_key] = terminal_output
                                        training_process_result = type('obj', (object,), {'stdout': 'no_training', 'returncode': 1})()
                                    except Exception as e:
                                        terminal_output.append(f"[ERROR] Process check failed with exception: {str(e)}")
                                        terminal_output.append(f"[DEBUG] Exception type: {type(e).__name__}")
                                        st.session_state[terminal_output_key] = terminal_output
                                        training_process_result = type('obj', (object,), {'stdout': 'no_training', 'returncode': 1})()
                                    
                                    # Check if process is running - need actual output, not just absence of "no_training"
                                    stdout_text = training_process_result.stdout or ""
                                    # Ensure we return a boolean, not an empty string
                                    ssh_training_running = bool(
                                        stdout_text.strip() and 
                                        "no_training" not in stdout_text and
                                        len(stdout_text.strip()) > 0
                                    )
                                    terminal_output.append(f"[DEBUG] ssh_training_running determined as: {ssh_training_running} (type: {type(ssh_training_running).__name__})")
                                    terminal_output.append(f"[DEBUG] stdout_text length: {len(stdout_text)}, content preview: {repr(stdout_text[:100])}")
                                    
                                    if ssh_training_running:
                                        # Show raw output first for debugging
                                        raw_output = training_process_result.stdout
                                        stdout_filtered = filter_malloc_warnings(raw_output)
                                        
                                        # Count processes and check if they're legitimate workers or duplicates
                                        process_lines = [line for line in raw_output.strip().split("\n") 
                                                        if line.strip() and "no_training" not in line and "grep" not in line.lower()]
                                        
                                        # Extract PIDs to analyze process relationships
                                        pids = []
                                        for line in process_lines:
                                            parts = line.split()
                                            if len(parts) > 1:
                                                try:
                                                    pids.append(int(parts[1]))  # PID is second column in ps aux
                                                except:
                                                    pass
                                        
                                        # Analyze if multiple processes are legitimate workers or duplicates
                                        if len(pids) > 1:
                                            terminal_output.append(f"[SSH] ⚠️ Detected {len(pids)} training-related processes")
                                            # Check if processes share a common parent (legitimate workers from accelerate)
                                            try:
                                                parent_pids = {}
                                                for pid in pids:
                                                    get_parent_cmd = [
                                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                        f"root@{ssh_host}",
                                                        f"ps -o ppid= -p {pid} 2>/dev/null | tr -d ' '"
                                                    ]
                                                    parent_result = subprocess.run(get_parent_cmd, capture_output=True, text=True, timeout=5)
                                                    if parent_result.returncode == 0 and parent_result.stdout.strip():
                                                        try:
                                                            parent_pid = int(parent_result.stdout.strip())
                                                            parent_pids[pid] = parent_pid
                                                        except:
                                                            pass
                                                
                                                # Check if processes are related (shared parents indicate workers)
                                                unique_parents = set(parent_pids.values())
                                                if len(unique_parents) <= 2:  # Allow for accelerate parent + main process
                                                    terminal_output.append(f"[SSH] ✓ Processes appear to be legitimate worker processes (accelerate workers)")
                                                else:
                                                    terminal_output.append(f"[SSH] ⚠️ WARNING: Multiple independent training processes detected!")
                                                    terminal_output.append(f"[SSH] ⚠️ This may cause OOM errors. Use 'Redo Phase' to kill all processes.")
                                            except Exception as e:
                                                # If analysis fails, just show warning
                                                terminal_output.append(f"[SSH] ⚠️ Multiple processes detected - unable to verify if they're workers or duplicates")
                                        
                                        terminal_output.append(f"[SSH] ✓ Training process detected:")
                                        
                                        # Show raw output if filtered output is empty or different
                                        if not stdout_filtered.strip() or stdout_filtered.strip() == raw_output.strip():
                                            # No filtering happened or output is the same
                                            display_lines = process_lines
                                        else:
                                            # Output was filtered, show both
                                            terminal_output.append(f"[DEBUG] Raw process output (before filtering):")
                                            for line in raw_output.strip().split("\n")[:3]:
                                                if line.strip() and "no_training" not in line:
                                                    terminal_output.append(f"[DEBUG]   {line[:200]}")
                                            display_lines = stdout_filtered.strip().split("\n")
                                        
                                        process_lines_shown = False
                                        for line in display_lines[:5]:
                                            if line.strip() and "no_training" not in line and "grep" not in line.lower():
                                                terminal_output.append(f"[SSH]   {line[:150]}")
                                                process_lines_shown = True
                                        
                                        if not process_lines_shown:
                                            terminal_output.append(f"[SSH]   (Process found but no process lines to display)")
                                            terminal_output.append(f"[SSH]   Raw output length: {len(raw_output)} chars")
                                            if raw_output.strip():
                                                terminal_output.append(f"[SSH]   First 200 chars of raw output: {raw_output[:200]}")
                                        
                                        st.session_state[terminal_output_key] = terminal_output
                                        
                                        # Get more details about the process - show full command line
                                        terminal_output.append(f"[SSH] Getting detailed process information...")
                                        st.session_state[terminal_output_key] = terminal_output
                                        
                                        check_process_cmd_full = [
                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                            f"root@{ssh_host}",
                                            "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | head -1"
                                        ]
                                        try:
                                            process_cmd_full_result = subprocess.run(check_process_cmd_full, capture_output=True, text=True, timeout=15)
                                        except subprocess.TimeoutExpired:
                                            terminal_output.append(f"[WARNING] Process details check timed out")
                                            st.session_state[terminal_output_key] = terminal_output
                                            process_cmd_full_result = type('obj', (object,), {'stdout': '', 'returncode': 1})()
                                        
                                        if process_cmd_full_result.stdout and process_cmd_full_result.stdout.strip():
                                            raw_cmd_output = process_cmd_full_result.stdout
                                            cmd_filtered = filter_malloc_warnings(raw_cmd_output)
                                            
                                            # Use filtered output, or raw if filtering removed everything
                                            cmd_output = cmd_filtered if cmd_filtered.strip() else raw_cmd_output
                                            
                                            terminal_output.append(f"[SSH] Process details:")
                                            # Parse the ps output to show PID, status, and command
                                            parts = cmd_output.strip().split(None, 10)
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
                                                terminal_output.append(f"[SSH]   Could not parse process details (unexpected format)")
                                                terminal_output.append(f"[DEBUG]   Raw output: {cmd_output[:300]}")
                                                terminal_output.append(f"[DEBUG]   Parts count: {len(parts)}")
                                            st.session_state[terminal_output_key] = terminal_output
                                        else:
                                            terminal_output.append(f"[SSH]   Could not get process details")
                                            if process_cmd_full_result.stdout:
                                                terminal_output.append(f"[DEBUG]   stdout: {process_cmd_full_result.stdout[:200]}")
                                            if hasattr(process_cmd_full_result, 'stderr') and process_cmd_full_result.stderr:
                                                terminal_output.append(f"[DEBUG]   stderr: {process_cmd_full_result.stderr[:200]}")
                                            terminal_output.append(f"[DEBUG]   returncode: {getattr(process_cmd_full_result, 'returncode', 'unknown')}")
                                            st.session_state[terminal_output_key] = terminal_output
                                    else:
                                        # No process found - show what we got for debugging
                                        terminal_output.append(f"[SSH] No training process found")
                                        if training_process_result.stdout:
                                            terminal_output.append(f"[DEBUG] Process check stdout: {training_process_result.stdout[:200]}")
                                        if hasattr(training_process_result, 'stderr') and training_process_result.stderr:
                                            stderr_text = training_process_result.stderr
                                            if "Welcome to vast.ai" not in stderr_text:  # Don't show the welcome message
                                                terminal_output.append(f"[DEBUG] Process check stderr: {stderr_text[:200]}")
                                        terminal_output.append(f"[DEBUG] Process check returncode: {getattr(training_process_result, 'returncode', 'unknown')}")
                                    st.session_state[terminal_output_key] = list(terminal_output)  # Save after process check
                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                    
                                    # Check training log (not onstart log - we're using existing instances)
                                    terminal_output.append(f"[SSH] Checking training logs...")
                                    st.session_state[terminal_output_key] = terminal_output
                                    
                                    # First check if log file exists and its size
                                    check_log_exists_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                        f"root@{ssh_host}",
                                        "ls -lh /workspace/output/training/training.log 2>/dev/null | awk '{print $5, $9}' || echo 'log_not_found'"
                                    ]
                                    try:
                                        log_exists_result = subprocess.run(check_log_exists_cmd, capture_output=True, text=True, timeout=15)
                                    except subprocess.TimeoutExpired:
                                        terminal_output.append(f"[WARNING] Log file check timed out")
                                        st.session_state[terminal_output_key] = terminal_output
                                        log_exists_result = type('obj', (object,), {'stdout': 'log_not_found', 'returncode': 1})()
                                    
                                    if "log_not_found" not in log_exists_result.stdout and log_exists_result.stdout.strip():
                                        terminal_output.append(f"[SSH] Training log file: {log_exists_result.stdout.strip()}")
                                        st.session_state[terminal_output_key] = terminal_output
                                    
                                    terminal_output.append(f"[SSH] Checking training logs...")
                                    terminal_output.append(f"[DEBUG] SSH command: ssh -p {ssh_port} root@{ssh_host} 'tail -100 /workspace/output/training/training.log'")
                                    check_training_log_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                        f"root@{ssh_host}",
                                        "tail -100 /workspace/output/training/training.log 2>/dev/null || echo 'no_training_log'"
                                    ]
                                    try:
                                        training_log_result = subprocess.run(check_training_log_cmd, capture_output=True, text=True, timeout=15)
                                        terminal_output.append(f"[DEBUG] Training log check returncode: {training_log_result.returncode}")
                                        if training_log_result.stdout:
                                            terminal_output.append(f"[DEBUG] Training log check stdout length: {len(training_log_result.stdout)}")
                                            terminal_output.append(f"[DEBUG] Training log check stdout preview: {training_log_result.stdout[:300]}")
                                        if training_log_result.stderr:
                                            terminal_output.append(f"[DEBUG] Training log check stderr: {training_log_result.stderr[:200]}")
                                    except subprocess.TimeoutExpired:
                                        terminal_output.append(f"[WARNING] Training log check timed out")
                                        terminal_output.append(f"[DEBUG] Timeout after 15 seconds")
                                        st.session_state[terminal_output_key] = terminal_output
                                        training_log_result = type('obj', (object,), {'stdout': 'no_training_log', 'returncode': 1})()
                                    except Exception as e:
                                        terminal_output.append(f"[ERROR] Training log check failed: {str(e)}")
                                        terminal_output.append(f"[DEBUG] Exception type: {type(e).__name__}")
                                        st.session_state[terminal_output_key] = terminal_output
                                        training_log_result = type('obj', (object,), {'stdout': 'no_training_log', 'returncode': 1})()
                                    training_logs_content = None
                                    if "no_training_log" not in training_log_result.stdout and training_log_result.stdout.strip():
                                        stdout_filtered = filter_malloc_warnings(training_log_result.stdout)
                                        training_logs_content = stdout_filtered
                                        
                                        # Also try to get the full log to extract stats more accurately
                                        full_log_cmd = [
                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                            f"root@{ssh_host}",
                                            "cat /workspace/output/training/training.log 2>/dev/null | head -500 || echo 'no_full_log'"
                                        ]
                                        try:
                                            full_log_result = subprocess.run(full_log_cmd, capture_output=True, text=True, timeout=15)
                                        except subprocess.TimeoutExpired:
                                            terminal_output.append(f"[WARNING] Full log retrieval timed out - using partial log")
                                            st.session_state[terminal_output_key] = terminal_output
                                            full_log_result = type('obj', (object,), {'stdout': 'no_full_log', 'returncode': 1})()
                                        if "no_full_log" not in full_log_result.stdout and full_log_result.stdout.strip():
                                            full_log_content = filter_malloc_warnings(full_log_result.stdout)
                                            # Use full log for stats extraction (more accurate)
                                            training_logs_content = full_log_content
                                        
                                        # Extract dataset statistics from training logs
                                        training_stats = extract_dataset_stats(training_logs_content)
                                        
                                        # If we still don't have final_count, try to check the prepared dataset directly
                                        if training_stats.get('original_count') and not training_stats.get('final_count'):
                                            terminal_output.append(f"[DATASET STATS] Attempting to get final count from prepared dataset...")
                                            st.session_state[terminal_output_key] = terminal_output
                                            
                                            # Try multiple methods to count the prepared dataset
                                            check_dataset_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                                f"root@{ssh_host}",
                                                "find /workspace/axolotl/prepared_data -name 'train.jsonl' -type f 2>/dev/null | head -1 | xargs -I {} sh -c 'if [ -f \"{}\" ]; then wc -l \"{}\" | awk \"{print \\$1}\"; else echo \"0\"; fi' || echo 'cannot_count'"
                                            ]
                                            try:
                                                dataset_count_result = subprocess.run(check_dataset_cmd, capture_output=True, text=True, timeout=15)
                                            except subprocess.TimeoutExpired:
                                                terminal_output.append(f"[WARNING] Dataset count check timed out")
                                                st.session_state[terminal_output_key] = terminal_output
                                                dataset_count_result = type('obj', (object,), {'stdout': 'cannot_count', 'returncode': 1})()
                                            
                                            if dataset_count_result.stdout and dataset_count_result.stdout.strip().isdigit():
                                                try:
                                                    final_count = int(dataset_count_result.stdout.strip())
                                                    if final_count > 0:
                                                        training_stats['final_count'] = final_count
                                                        terminal_output.append(f"[DATASET STATS] Found final count from prepared dataset: {final_count}")
                                                        st.session_state[terminal_output_key] = terminal_output
                                                except:
                                                    pass
                                            elif "cannot_count" not in dataset_count_result.stdout:
                                                terminal_output.append(f"[DEBUG] Dataset count result: {dataset_count_result.stdout[:200]}")
                                                st.session_state[terminal_output_key] = terminal_output
                                        if training_stats:
                                            terminal_output.append(f"[DATASET STATS] ========================================")
                                            if training_stats.get("original_count"):
                                                terminal_output.append(f"[DATASET STATS] Original samples: {training_stats['original_count']}")
                                            if training_stats.get("final_count"):
                                                terminal_output.append(f"[DATASET STATS] Final training samples: {training_stats['final_count']}")
                                                # Calculate retention
                                                if training_stats.get("original_count"):
                                                    retention_pct = (training_stats['final_count'] / training_stats['original_count'] * 100)
                                                    terminal_output.append(f"[DATASET STATS] Sample retention: {retention_pct:.1f}%")
                                            else:
                                                terminal_output.append(f"[DATASET STATS] Final count: Not available (checking prepared dataset...)")
                                            if training_stats.get("dropped_long"):
                                                terminal_output.append(f"[DATASET STATS] Dropped (too long): {training_stats['dropped_long']}")
                                            if training_stats.get("dropped_zero_tokens"):
                                                terminal_output.append(f"[DATASET STATS] Dropped (zero tokens): {training_stats['dropped_zero_tokens']}")
                                            if training_stats.get("total_dropped"):
                                                dropped_pct = (training_stats['total_dropped'] / training_stats['original_count'] * 100) if training_stats.get('original_count') else 0
                                                terminal_output.append(f"[DATASET STATS] Total dropped: {training_stats['total_dropped']} ({dropped_pct:.1f}%)")
                                            # Show train_on_inputs status if we can check the config
                                            terminal_output.append(f"[DATASET STATS] train_on_inputs: True (set to maximize retention)")
                                            terminal_output.append(f"[DATASET STATS] ========================================")
                                            st.session_state[terminal_output_key] = terminal_output
                                        
                                        terminal_output.append(f"[TRAINING LOG] Last 20 lines:")
                                        for line in stdout_filtered.strip().split("\n")[-20:]:
                                            if line.strip():
                                                terminal_output.append(f"[TRAINING] {line[:200]}")
                                    else:
                                        terminal_output.append(f"[TRAINING LOG] No training log found yet - training may not have started")
                                        
                                        # Check if training was ever started - look for any evidence of training attempts
                                        if not ssh_training_running:
                                            terminal_output.append(f"[DIAGNOSTICS] No training process found. Checking if training was ever started...")
                                            st.session_state[terminal_output_key] = terminal_output
                                            
                                            # Check if there's any history of training commands or if the log directory was ever written to
                                            check_training_history_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "ls -la /workspace/output/training/ 2>/dev/null | head -10 || echo 'no_output_dir'"
                                            ]
                                            try:
                                                history_result = subprocess.run(check_training_history_cmd, capture_output=True, text=True, timeout=15)
                                                if history_result.stdout and "no_output_dir" not in history_result.stdout:
                                                    terminal_output.append(f"[DIAGNOSTICS] Output directory contents: {history_result.stdout.strip()[:300]}")
                                                else:
                                                    terminal_output.append(f"[DIAGNOSTICS] Output directory is empty or doesn't exist")
                                            except Exception as e:
                                                terminal_output.append(f"[DIAGNOSTICS] Could not check training history: {str(e)}")
                                            
                                            # Check if axolotl is properly installed and accessible
                                            check_axolotl_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "cd /workspace/axolotl && (/opt/conda/bin/python -c 'import axolotl; import accelerate' 2>&1 || python3 -c 'import axolotl; import accelerate' 2>&1 || python -c 'import axolotl; import accelerate' 2>&1) || echo 'axolotl_not_available'"
                                            ]
                                            try:
                                                axolotl_check = subprocess.run(check_axolotl_cmd, capture_output=True, text=True, timeout=15)
                                                if "axolotl_not_available" in axolotl_check.stdout:
                                                    terminal_output.append(f"[DIAGNOSTICS] ⚠️ WARNING: axolotl or accelerate may not be properly installed")
                                                elif axolotl_check.returncode == 0:
                                                    terminal_output.append(f"[DIAGNOSTICS] ✓ axolotl and accelerate are available")
                                                else:
                                                    terminal_output.append(f"[DIAGNOSTICS] axolotl check output: {axolotl_check.stdout[:200]}")
                                            except Exception as e:
                                                terminal_output.append(f"[DIAGNOSTICS] Could not check axolotl installation: {str(e)}")
                                            
                                            # Check if config file is valid
                                            check_config_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "test -f /workspace/data/axolotl_config.yaml && (head -5 /workspace/data/axolotl_config.yaml || echo 'config_read_error') || echo 'config_not_found'"
                                            ]
                                            try:
                                                config_check = subprocess.run(check_config_cmd, capture_output=True, text=True, timeout=15)
                                                if "config_not_found" in config_check.stdout:
                                                    terminal_output.append(f"[DIAGNOSTICS] ⚠️ WARNING: Config file not found at /workspace/data/axolotl_config.yaml")
                                                elif "config_read_error" in config_check.stdout:
                                                    terminal_output.append(f"[DIAGNOSTICS] ⚠️ WARNING: Could not read config file")
                                                else:
                                                    terminal_output.append(f"[DIAGNOSTICS] ✓ Config file exists and is readable")
                                            except Exception as e:
                                                terminal_output.append(f"[DIAGNOSTICS] Could not check config file: {str(e)}")
                                            
                                            terminal_output.append(f"[INFO] Training has not started. Use 'Start Training' or 'Restart Training' button to begin.")
                                            st.session_state[terminal_output_key] = terminal_output
                                        
                                        # If process is running but no logs, check for errors in other locations
                                        if ssh_training_running:
                                            terminal_output.append(f"[DIAGNOSTICS] Process detected but no logs - checking for errors...")
                                            
                                            # Check if the log file is being created but is empty (process might be starting)
                                            check_log_size_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                            # This is a non-critical diagnostic check, so handle timeouts gracefully
                                            try:
                                                check_tmp_output_cmd = [
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                    f"root@{ssh_host}",
                                                    "ls -lht /tmp/*.log /tmp/*.out /tmp/*.err 2>/dev/null | head -3 || echo 'no_tmp_output'"
                                                ]
                                                tmp_output_result = subprocess.run(check_tmp_output_cmd, capture_output=True, text=True, timeout=10)
                                                if "no_tmp_output" not in tmp_output_result.stdout and tmp_output_result.stdout.strip():
                                                    tmp_filtered = filter_malloc_warnings(tmp_output_result.stdout)
                                                    terminal_output.append(f"[DIAGNOSTICS] Found output files in /tmp: {tmp_filtered.strip()}")
                                            except subprocess.TimeoutExpired:
                                                # Timeout is non-critical - just skip this diagnostic check
                                                pass
                                            except Exception as e:
                                                # Other errors are also non-critical
                                                pass
                                            
                                            # Check if we can see the process's file descriptors to see if it's writing
                                            check_process_fds_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "ps aux | grep -E '(accelerate|axolotl|train)' | grep -v grep | awk '{print $2}' | head -1 | xargs -I {} sh -c 'if [ -n \"{}\" ]; then ls -l /proc/{}/fd/ 2>/dev/null | grep -E \"(training.log|stdout|stderr)\" || echo \"no_fds\"; else echo \"no_pid\"; fi'"
                                            ]
                                            process_fds_result = subprocess.run(check_process_fds_cmd, capture_output=True, text=True, timeout=15)
                                            if process_fds_result.stdout and "no_fds" not in process_fds_result.stdout and "no_pid" not in process_fds_result.stdout:
                                                fds_filtered = filter_malloc_warnings(process_fds_result.stdout)
                                                terminal_output.append(f"[DIAGNOSTICS] Process file descriptors: {fds_filtered.strip()}")
                                                
                                                # If log file is deleted but process has it open, try to read from the file descriptor
                                                if "(deleted)" in process_fds_result.stdout:
                                                    terminal_output.append(f"[DIAGNOSTICS] ⚠️ Log file was deleted but process still has it open")
                                                    terminal_output.append(f"[DIAGNOSTICS] The log file /workspace/output/training/training.log was deleted after the process started")
                                                    terminal_output.append(f"[DIAGNOSTICS] The process (PID) still has the file descriptor open, so we can read from it")
                                                    terminal_output.append(f"[DIAGNOSTICS] Location: Reading from /proc/PID/fd/1 (process stdout file descriptor)")
                                                    terminal_output.append(f"[DIAGNOSTICS] To recreate the log file, you would need to restart training with proper logging")
                                                    
                                                    # Get the main training process PID (the one with accelerate launch)
                                                    get_main_pid_cmd = [
                                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                        f"root@{ssh_host}",
                                                        "ps aux | grep -E 'accelerate.*launch.*axolotl' | grep -v grep | awk '{print $2}' | head -1"
                                                    ]
                                                    try:
                                                        pid_result = subprocess.run(get_main_pid_cmd, capture_output=True, text=True, timeout=10)
                                                        if pid_result.stdout and pid_result.stdout.strip().isdigit():
                                                            main_pid = pid_result.stdout.strip()
                                                            terminal_output.append(f"[DIAGNOSTICS] Main process PID: {main_pid}")
                                                            terminal_output.append(f"[DIAGNOSTICS] Reading logs from /proc/{main_pid}/fd/1 (process stdout)")
                                                            
                                                            # Try to read from stdout (fd 1) and stderr (fd 2)
                                                            read_fd_cmd = [
                                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                                f"root@{ssh_host}",
                                                                f"if [ -r /proc/{main_pid}/fd/1 ]; then tail -100 /proc/{main_pid}/fd/1 2>/dev/null || echo 'cannot_read_fd1'; else echo 'fd1_not_readable'; fi"
                                                            ]
                                                            fd_read_result = subprocess.run(read_fd_cmd, capture_output=True, text=True, timeout=15)
                                                            if fd_read_result.stdout and "cannot_read_fd1" not in fd_read_result.stdout and "fd1_not_readable" not in fd_read_result.stdout and fd_read_result.stdout.strip():
                                                                fd_content = filter_malloc_warnings(fd_read_result.stdout)
                                                                terminal_output.append(f"[TRAINING LOG] Reading from process stdout (PID {main_pid}):")
                                                                terminal_output.append(f"[INFO] Note: This is reading from the process file descriptor, not a file on disk")
                                                                # Show last 50 lines
                                                                fd_lines = fd_content.strip().split("\n")
                                                                for line in fd_lines[-50:]:
                                                                    if line.strip():
                                                                        terminal_output.append(f"[LOG] {line[:300]}")
                                                                st.session_state[terminal_output_key] = terminal_output
                                                            else:
                                                                terminal_output.append(f"[DIAGNOSTICS] Could not read from file descriptor")
                                                    except Exception as e:
                                                        terminal_output.append(f"[DIAGNOSTICS] Could not read from process file descriptor: {str(e)}")
                                            
                                            # Also check Axolotl's default output locations
                                            check_axolotl_output_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "find /workspace -name '*.log' -type f -path '*/axolotl/*' -o -path '*/output/*' 2>/dev/null | head -5 | xargs -I {} sh -c 'if [ -f \"{}\" ]; then echo \"{}: $(tail -20 \"{}\" | wc -l) lines\"; fi' || echo 'no_axolotl_logs'"
                                            ]
                                            try:
                                                axolotl_output_result = subprocess.run(check_axolotl_output_cmd, capture_output=True, text=True, timeout=15)
                                                if axolotl_output_result.stdout and "no_axolotl_logs" not in axolotl_output_result.stdout and axolotl_output_result.stdout.strip():
                                                    terminal_output.append(f"[DIAGNOSTICS] Found Axolotl output files: {axolotl_output_result.stdout.strip()}")
                                            except:
                                                pass
                                    
                                    # Check debug.log if it exists
                                    terminal_output.append(f"[SSH] Checking debug.log for errors...")
                                    terminal_output.append(f"[DEBUG] SSH command: ssh -p {ssh_port} root@{ssh_host} 'if [ -f /workspace/output/training/debug.log ]; then tail -30 /workspace/output/training/debug.log; else echo no_debug_log; fi'")
                                    debug_log_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        "if [ -f /workspace/output/training/debug.log ]; then tail -30 /workspace/output/training/debug.log; else echo 'no_debug_log'; fi"
                                    ]
                                    try:
                                        debug_result = subprocess.run(debug_log_cmd, capture_output=True, text=True, timeout=15)
                                        terminal_output.append(f"[DEBUG] Debug log check returncode: {debug_result.returncode}")
                                        if debug_result.stdout:
                                            terminal_output.append(f"[DEBUG] Debug log check stdout length: {len(debug_result.stdout)}")
                                        if debug_result.stderr:
                                            terminal_output.append(f"[DEBUG] Debug log check stderr: {debug_result.stderr[:200]}")
                                    except Exception as e:
                                        terminal_output.append(f"[ERROR] Debug log check failed: {str(e)}")
                                        debug_result = type('obj', (object,), {'stdout': 'no_debug_log', 'returncode': 1})()
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
                                    st.session_state[terminal_output_key] = terminal_output
                                    log_command = (
                                        "if [ -f /workspace/output/training/training.log ]; then "
                                        "tail -50 /workspace/output/training/training.log; "
                                        "elif [ -f /workspace/axolotl/training.log ]; then "
                                        "tail -50 /workspace/axolotl/training.log; "
                                        "elif [ -f /workspace/output/training/debug.log ]; then "
                                        "tail -50 /workspace/output/training/debug.log; "
                                        "else echo 'no_logs'; fi"
                                    )
                                    terminal_output.append(f"[DEBUG] SSH command: ssh -p {ssh_port} root@{ssh_host} 'checking multiple log locations...'")
                                    log_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                        f"root@{ssh_host}",
                                        log_command
                                    ]
                                    terminal_output.append(f"[DEBUG] Log retrieval command: {log_command[:200]}...")
                                    try:
                                        terminal_output.append(f"[SSH] Attempting to retrieve logs (timeout: 15s)...")
                                        st.session_state[terminal_output_key] = terminal_output
                                        log_result = subprocess.run(log_cmd, capture_output=True, text=True, timeout=15)
                                        terminal_output.append(f"[DEBUG] Log retrieval returncode: {log_result.returncode}")
                                        if log_result.stdout:
                                            terminal_output.append(f"[DEBUG] Log retrieval stdout length: {len(log_result.stdout)}")
                                            terminal_output.append(f"[DEBUG] Log retrieval stdout preview: {log_result.stdout[:300]}")
                                        if log_result.stderr:
                                            terminal_output.append(f"[DEBUG] Log retrieval stderr: {log_result.stderr[:200]}")
                                    except subprocess.TimeoutExpired:
                                        terminal_output.append(f"[WARNING] Log retrieval timed out after 15 seconds")
                                        terminal_output.append(f"[DEBUG] Timeout exception occurred")
                                        terminal_output.append(f"[INFO] Log file may be very large or SSH connection is slow")
                                        terminal_output.append(f"[INFO] Training may still be initializing - logs will appear once training starts")
                                        st.session_state[terminal_output_key] = terminal_output
                                        log_result = type('obj', (object,), {'stdout': 'no_logs', 'returncode': 1})()
                                    except Exception as e:
                                        terminal_output.append(f"[ERROR] Log retrieval failed: {str(e)}")
                                        terminal_output.append(f"[DEBUG] Exception type: {type(e).__name__}")
                                        st.session_state[terminal_output_key] = terminal_output
                                        log_result = type('obj', (object,), {'stdout': 'no_logs', 'returncode': 1})()
                                    if "no_logs" not in log_result.stdout and log_result.stdout.strip():
                                        stdout_filtered = filter_malloc_warnings(log_result.stdout)
                                        # Get more lines to find the actual error (not just the wrapper)
                                        log_lines = stdout_filtered.strip().split("\n")
                                        # Append to existing training_logs_content if it exists, otherwise create new
                                        if training_logs_content:
                                            training_logs_content = training_logs_content + "\n" + "\n".join(log_lines)
                                        else:
                                            training_logs_content = "\n".join(log_lines)
                                    else:
                                        # If no logs found in files, try reading from process file descriptors
                                        terminal_output.append(f"[INFO] No training logs found in files - attempting to read from process file descriptors...")
                                        st.session_state[terminal_output_key] = terminal_output
                                        try:
                                            # Get the main training process PID
                                            get_pid_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "ps aux | grep -E 'accelerate.*launch.*axolotl' | grep -v grep | awk '{print $2}' | head -1"
                                            ]
                                            pid_result = subprocess.run(get_pid_cmd, capture_output=True, text=True, timeout=10)
                                            if pid_result.stdout and pid_result.stdout.strip().isdigit():
                                                main_pid = pid_result.stdout.strip()
                                                # Try to read from stdout (fd 1)
                                                read_fd_cmd = [
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                    f"root@{ssh_host}",
                                                    f"if [ -r /proc/{main_pid}/fd/1 ]; then tail -100 /proc/{main_pid}/fd/1 2>/dev/null || echo 'cannot_read'; else echo 'not_readable'; fi"
                                                ]
                                                fd_result = subprocess.run(read_fd_cmd, capture_output=True, text=True, timeout=15)
                                                if fd_result.stdout and "cannot_read" not in fd_result.stdout and "not_readable" not in fd_result.stdout and fd_result.stdout.strip():
                                                    fd_content = filter_malloc_warnings(fd_result.stdout)
                                                    training_logs_content = fd_content
                                                    terminal_output.append(f"[SUCCESS] Retrieved logs from process stdout (PID {main_pid})")
                                                    st.session_state[terminal_output_key] = terminal_output
                                                else:
                                                    terminal_output.append(f"[INFO] Could not read from process file descriptor")
                                            else:
                                                terminal_output.append(f"[INFO] Could not find main training process PID")
                                        except Exception as e:
                                            terminal_output.append(f"[INFO] Could not read from process file descriptor: {str(e)}")
                                        
                                        # Extract dataset statistics from logs (if we have content)
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
                                        
                                        # Verify LoRA mode is active
                                        lora_verification_key = f"lora_verified_{instance_id}"
                                        if lora_verification_key not in st.session_state:
                                            import re
                                            lora_indicators = []
                                            
                                            # Check logs for LoRA indicators
                                            if training_logs_content:
                                                log_lower = training_logs_content.lower()
                                                
                                                # Check for trainable parameters (LoRA shows much smaller trainable vs total)
                                                trainable_params_match = re.search(r'trainable params[:\s]+([\d,]+)', log_lower, re.IGNORECASE)
                                                all_params_match = re.search(r'all params[:\s]+([\d,]+)', log_lower, re.IGNORECASE)
                                                
                                                if trainable_params_match and all_params_match:
                                                    try:
                                                        trainable = int(trainable_params_match.group(1).replace(',', ''))
                                                        all_params = int(all_params_match.group(1).replace(',', ''))
                                                        if all_params > 0:
                                                            trainable_pct = (trainable / all_params) * 100
                                                            if trainable_pct < 5:  # LoRA typically trains <5% of parameters
                                                                lora_indicators.append(f"✓ Trainable params: {trainable:,} ({trainable_pct:.2f}% of {all_params:,}) - indicates LoRA mode")
                                                            else:
                                                                lora_indicators.append(f"⚠ Trainable params: {trainable:,} ({trainable_pct:.2f}% of {all_params:,}) - may not be LoRA")
                                                    except:
                                                        pass
                                                
                                                # Check for PEFT/LoRA mentions
                                                if any(term in log_lower for term in ['peft', 'lora', 'low-rank', 'adapter']):
                                                    lora_indicators.append("✓ Found PEFT/LoRA mentions in logs")
                                                
                                                # Check for LoRA module names
                                                if re.search(r'lora\.(a|b)', log_lower):
                                                    lora_indicators.append("✓ Found LoRA module references (lora.a/lora.b)")
                                            
                                            # Check config file for LoRA parameters
                                            if ssh_host:
                                                try:
                                                    check_config_cmd = [
                                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                        f"root@{ssh_host}",
                                                        "cd /workspace/data && python3 << 'PYTHON_EOF'\n"
                                                        "import yaml\n"
                                                        "import os\n"
                                                        "import json\n"
                                                        "try:\n"
                                                        "    with open('axolotl_config.yaml', 'r') as f:\n"
                                                        "        config = yaml.safe_load(f) or {}\n"
                                                        "    \n"
                                                        "    # Check for LoRA parameters\n"
                                                        "    has_lora_r = config.get('lora_r') is not None\n"
                                                        "    has_lora_alpha = config.get('lora_alpha') is not None\n"
                                                        "    adapter_val = config.get('adapter')\n"
                                                        "    has_valid_adapter_path = adapter_val and isinstance(adapter_val, str) and adapter_val.startswith('/')\n"
                                                        "    merge_lora = config.get('merge_lora', True)\n"
                                                        "    save_merged_lora = config.get('save_merged_lora', True)\n"
                                                        "    base_model = config.get('base_model', '')\n"
                                                        "    \n"
                                                        "    # Check if adapter path exists (for incremental training)\n"
                                                        "    adapter_exists = False\n"
                                                        "    adapter_config_exists = False\n"
                                                        "    if has_valid_adapter_path:\n"
                                                        "        adapter_config_path = os.path.join(adapter_val, 'adapter_config.json')\n"
                                                        "        adapter_exists = os.path.exists(adapter_val)\n"
                                                        "        adapter_config_exists = os.path.exists(adapter_config_path)\n"
                                                        "        \n"
                                                        "        # If adapter config exists, check base model match\n"
                                                        "        base_model_match = False\n"
                                                        "        if adapter_config_exists:\n"
                                                        "            try:\n"
                                                        "                with open(adapter_config_path, 'r') as f:\n"
                                                        "                    adapter_config = json.load(f)\n"
                                                        "                    adapter_base_model = adapter_config.get('base_model_name', '')\n"
                                                        "                    base_model_match = (adapter_base_model == base_model) if adapter_base_model and base_model else False\n"
                                                        "            except:\n"
                                                        "                pass\n"
                                                        "    \n"
                                                        "    # LoRA mode is enabled if adapter: 'lora' is set OR if there's a valid adapter path\n"
                                                        "    is_lora_mode = (adapter_val == 'lora' and has_lora_r and has_lora_alpha) or (has_valid_adapter_path and adapter_config_exists)\n"
                                                        "    \n"
                                                        "    print('LORA_CONFIG_CHECK:')\n"
                                                        "    print(f'base_model={base_model}')\n"
                                                        "    print(f'lora_r={config.get(\"lora_r\")}')\n"
                                                        "    print(f'lora_alpha={config.get(\"lora_alpha\")}')\n"
                                                        "    print(f'adapter={adapter_val}')\n"
                                                        "    print(f'adapter_exists={adapter_exists}')\n"
                                                        "    print(f'adapter_config_exists={adapter_config_exists}')\n"
                                                        "    if has_valid_adapter_path and adapter_config_exists:\n"
                                                        "        try:\n"
                                                        "            with open(os.path.join(adapter_val, 'adapter_config.json'), 'r') as f:\n"
                                                        "                adapter_config = json.load(f)\n"
                                                        "                adapter_base = adapter_config.get('base_model_name', 'unknown')\n"
                                                        "                print(f'adapter_base_model={adapter_base}')\n"
                                                        "                print(f'base_model_match={adapter_base == base_model}')\n"
                                                        "        except:\n"
                                                        "            pass\n"
                                                        "    print(f'merge_lora={merge_lora}')\n"
                                                        "    print(f'save_merged_lora={save_merged_lora}')\n"
                                                        "    print(f'has_lora_params={has_lora_r and has_lora_alpha}')\n"
                                                        "    print(f'is_lora_mode={is_lora_mode}')\n"
                                                        "except Exception as e:\n"
                                                        "    print(f'ERROR: {str(e)}')\n"
                                                        "PYTHON_EOF"
                                                    ]
                                                    config_check_result = subprocess.run(check_config_cmd, capture_output=True, text=True, timeout=15)
                                                    if config_check_result.returncode == 0:
                                                        config_output = config_check_result.stdout.strip()
                                                        if "LORA_CONFIG_CHECK:" in config_output:
                                                            for line in config_output.split("\n"):
                                                                if "=" in line and "LORA_CONFIG_CHECK" not in line:
                                                                    if "is_lora_mode=True" in line:
                                                                        lora_indicators.append("✓ Config confirms LoRA mode is enabled")
                                                                    elif "lora_r=" in line and "None" not in line:
                                                                        lora_r_val = line.split("=")[1].strip()
                                                                        lora_indicators.append(f"✓ Config has lora_r={lora_r_val}")
                                                                    elif "adapter=" in line and "None" not in line:
                                                                        adapter_val = line.split("=")[1].strip()
                                                                        if adapter_val == "lora":
                                                                            lora_indicators.append("✓ Config has adapter: lora (LoRA mode enabled)")
                                                                        elif adapter_val.startswith("/"):
                                                                            lora_indicators.append(f"✓ Config has adapter path: {adapter_val} (incremental training)")
                                                                    elif "adapter_exists=True" in line:
                                                                        lora_indicators.append("✓ Adapter directory exists on remote")
                                                                    elif "adapter_config_exists=True" in line:
                                                                        lora_indicators.append("✓ Adapter config file found (adapter is valid)")
                                                                    elif "base_model_match=True" in line:
                                                                        lora_indicators.append("✓ Base model matches between adapter and config")
                                                                    elif "base_model_match=False" in line and "adapter_base_model=" in config_output:
                                                                        # Extract both models for comparison
                                                                        adapter_base = None
                                                                        config_base = None
                                                                        for l in config_output.split("\n"):
                                                                            if "adapter_base_model=" in l:
                                                                                adapter_base = l.split("=")[1].strip()
                                                                            elif "base_model=" in l and "adapter_base_model" not in l:
                                                                                config_base = l.split("=")[1].strip()
                                                                        if adapter_base and config_base:
                                                                            lora_indicators.append(f"⚠ Base model mismatch: adapter={adapter_base}, config={config_base}")
                                                                    elif "merge_lora=False" in line:
                                                                        lora_indicators.append("✓ Config has merge_lora: false (adapters will be saved separately)")
                                                except:
                                                    pass
                                            
                                            # Display LoRA verification results
                                            if lora_indicators:
                                                terminal_output.append(f"[LoRA VERIFICATION] ========================================")
                                                for indicator in lora_indicators:
                                                    terminal_output.append(f"[LoRA VERIFICATION] {indicator}")
                                                terminal_output.append(f"[LoRA VERIFICATION] ========================================")
                                                st.session_state[lora_verification_key] = True
                                            elif training_logs_content and len(training_logs_content) > 1000:
                                                # Only show warning if we have substantial logs but no LoRA indicators
                                                terminal_output.append(f"[LoRA VERIFICATION] ⚠ No clear LoRA indicators found in logs yet")
                                                terminal_output.append(f"[LoRA VERIFICATION] This may be normal if training just started")
                                        st.session_state[terminal_output_key] = list(terminal_output)  # Save after dataset stats
                                        st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                        
                                        # Show new log lines in terminal (avoid duplicates by tracking last seen line)
                                        # Use training_logs_content if log_lines is not available
                                        if training_logs_content:
                                            # Convert training_logs_content to log_lines if we don't have it
                                            if 'log_lines' not in locals():
                                                log_lines = training_logs_content.strip().split("\n")
                                            
                                            last_log_key = f"last_training_log_line_{instance_id}"
                                            last_seen_line = st.session_state.get(last_log_key, "")
                                            
                                            # Find where we left off (only add new lines that aren't already displayed)
                                            new_lines_start_idx = 0
                                            if last_seen_line:
                                                try:
                                                    # Find the index of the last seen line
                                                    for i, line in enumerate(log_lines):
                                                        if line.strip() == last_seen_line.strip():
                                                            new_lines_start_idx = i + 1
                                                            break
                                                except:
                                                    new_lines_start_idx = 0
                                            
                                            # Get new lines (or all lines if this is first time)
                                            new_lines = log_lines[new_lines_start_idx:] if new_lines_start_idx > 0 else log_lines[-50:]  # Show last 50 on first load
                                            
                                            if new_lines:
                                                terminal_output.append(f"[TRAINING LOGS] {len(new_lines)} new line(s) (showing last 50):")
                                                # Show last 50 lines to avoid overwhelming the terminal
                                                display_lines = new_lines[-50:] if len(new_lines) > 50 else new_lines
                                                for line in display_lines:
                                                    if line.strip():
                                                        terminal_output.append(f"[TRAINING] {line}")
                                                # Update last seen line
                                                if log_lines:
                                                    st.session_state[last_log_key] = log_lines[-1]
                                            else:
                                                terminal_output.append(f"[TRAINING LOGS] No new log lines since last check")
                                            
                                            st.session_state[terminal_output_key] = terminal_output
                                        else:
                                            terminal_output.append(f"[INFO] No training logs found yet")
                                            st.session_state[terminal_output_key] = terminal_output
                                
                                # Check for errors in training logs and debug logs before checking status
                                training_error = None
                                # Combine onstart logs, training logs, and debug logs for error detection and stats
                                # training_logs_content is set earlier in the code (around line 2990 or 3169)
                                # Make sure we have the latest training_logs_content
                                all_logs_content = ""
                                # Use training_logs_content if it was set (from either the earlier check or the later retrieval)
                                if training_logs_content:
                                    all_logs_content = training_logs_content
                                # Also check if we got logs from the later log retrieval (this overwrites training_logs_content)
                                # So if training_logs_content exists, it should already have the latest content
                                if debug_log_content:
                                    if all_logs_content:
                                        all_logs_content = all_logs_content + "\n" + debug_log_content
                                    else:
                                        all_logs_content = debug_log_content
                                
                                # Debug: Log what we have for completion detection
                                terminal_output.append(f"[DEBUG] all_logs_content length: {len(all_logs_content) if all_logs_content else 0}")
                                if all_logs_content:
                                    # Check if completion indicators are present
                                    has_completed = "training completed" in all_logs_content.lower()
                                    has_saved = "model successfully saved" in all_logs_content.lower()
                                    terminal_output.append(f"[DEBUG] Completion indicators: has_completed={has_completed}, has_saved={has_saved}")
                                    
                                    # If both completion indicators are present, force status to completed
                                    if has_completed and has_saved:
                                        terminal_output.append(f"[DEBUG] Both completion indicators detected - forcing status to 'completed'")
                                        status_val = "completed"
                                        training_status["status"] = "completed"
                                        active_job["training_status"] = training_status
                                        training_manager._save_job(active_job)
                                        terminal_output.append(f"[DEBUG] Status forced to 'completed' based on log indicators")
                                        # Note: Transition to Phase 4 will happen in the completion check below
                                st.session_state[terminal_output_key] = terminal_output
                                
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
                                            "IndexError",
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
                                            
                                            # Check for specific error types and provide guidance
                                            import re
                                            is_oom_error = False
                                            if "OutOfMemoryError" in training_error or "out of memory" in training_error.lower() or "CUDA out of memory" in training_error:
                                                is_oom_error = True
                                                terminal_output.append(f"[ERROR] ⚠️ CUDA Out of Memory Error detected!")
                                                
                                                # Extract memory information from error
                                                memory_info = {}
                                                # Try to extract: "Tried to allocate X GiB"
                                                alloc_match = re.search(r'tried to allocate ([\d.]+)\s*(GiB|MiB)', training_error, re.IGNORECASE)
                                                if alloc_match:
                                                    memory_info['requested'] = f"{alloc_match.group(1)} {alloc_match.group(2)}"
                                                
                                                # Try to extract: "GPU 0 has a total capacity of X GiB"
                                                total_match = re.search(r'total capacity of ([\d.]+)\s*(GiB|MiB)', training_error, re.IGNORECASE)
                                                if total_match:
                                                    memory_info['total'] = f"{total_match.group(1)} {total_match.group(2)}"
                                                
                                                # Try to extract: "X GiB is free"
                                                free_match = re.search(r'([\d.]+)\s*(GiB|MiB)\s+is free', training_error, re.IGNORECASE)
                                                if free_match:
                                                    memory_info['free'] = f"{free_match.group(1)} {free_match.group(2)}"
                                                
                                                # Try to extract process memory usage
                                                process_matches = re.findall(r'Process \d+ has ([\d.]+)\s*(GiB|MiB)', training_error, re.IGNORECASE)
                                                if process_matches:
                                                    memory_info['other_processes'] = [f"{m[0]} {m[1]}" for m in process_matches]
                                                
                                                # Provide detailed information
                                                if memory_info:
                                                    terminal_output.append(f"[INFO] Memory Status:")
                                                    if 'requested' in memory_info:
                                                        terminal_output.append(f"  • Requested: {memory_info['requested']}")
                                                    if 'total' in memory_info:
                                                        terminal_output.append(f"  • Total GPU: {memory_info['total']}")
                                                    if 'free' in memory_info:
                                                        terminal_output.append(f"  • Free: {memory_info['free']}")
                                                    if 'other_processes' in memory_info:
                                                        terminal_output.append(f"  • Other processes using: {', '.join(memory_info['other_processes'])}")
                                                
                                                terminal_output.append(f"[ACTION] Solutions:")
                                                terminal_output.append(f"  1. Reduce batch_size in training config (try 1 or 2)")
                                                terminal_output.append(f"  2. Reduce gradient_accumulation_steps")
                                                terminal_output.append(f"  3. Enable gradient_checkpointing if not already enabled")
                                                terminal_output.append(f"  4. Use a smaller model or enable LoRA with lower rank")
                                                terminal_output.append(f"  5. Kill other GPU processes if they're not needed")
                                                terminal_output.append(f"  6. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
                                                
                                            else:
                                                terminal_output.append(f"[ERROR] Error detected in training logs")
                                            
                                            st.session_state[terminal_output_key] = list(terminal_output)  # Save after error detection
                                            st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                else:
                                    # No SSH host available - try to get it from API
                                    terminal_output.append(f"[INFO] SSH host not available - attempting to retrieve from API...")
                                    try:
                                        updated_job_status = training_manager.get_job_status(instance_id)
                                        if updated_job_status and not updated_job_status.get("error"):
                                            ssh_host_from_api = updated_job_status.get("ssh_host")
                                            api_ssh_port = updated_job_status.get("ssh_port", 22)
                                            
                                            # Check for SSH port override (user-specified port takes precedence)
                                            ssh_port_override = active_job.get("ssh_port_override")
                                            if ssh_port_override:
                                                ssh_port_from_api = ssh_port_override
                                            else:
                                                ssh_port_from_api = api_ssh_port
                                            
                                            if ssh_host_from_api:
                                                active_job["ssh_host"] = ssh_host_from_api
                                                # Only update port if no override is set (override takes precedence)
                                                if not ssh_port_override:
                                                    active_job["ssh_port"] = api_ssh_port
                                                training_manager._save_job(active_job)
                                                # Update active_jobs list too
                                                for idx, job in enumerate(active_jobs):
                                                    if job.get("instance_id") == instance_id:
                                                        active_jobs[idx] = active_job
                                                        break
                                                port_source = "override" if ssh_port_override else "API"
                                                terminal_output.append(f"[SUCCESS] Retrieved SSH info from API: {ssh_host_from_api}:{ssh_port_from_api} ({port_source})")
                                                terminal_output.append(f"[INFO] SSH info saved to job. Refreshing...")
                                                st.session_state[terminal_output_key] = terminal_output
                                                # Update the local variables immediately before rerun
                                                ssh_host = ssh_host_from_api
                                                ssh_port = ssh_port_from_api
                                                st.rerun()  # Rerun to use the new SSH info
                                            else:
                                                terminal_output.append(f"[WARNING] SSH host still not available from API - instance may still be initializing")
                                                terminal_output.append(f"[INFO] You may need to wait for the instance to fully start, or check the instance status in Vast.ai")
                                    except Exception as e:
                                        terminal_output.append(f"[WARNING] Could not retrieve SSH info from API: {str(e)}")
                                    st.session_state[terminal_output_key] = terminal_output
                                
                                # Reload active_job from active_jobs list to get any SSH info that was just saved
                                # This ensures we have the latest saved SSH info after a rerun
                                for idx, job in enumerate(active_jobs):
                                    if job.get("instance_id") == instance_id:
                                        # Reload active_job from the list (may have been updated)
                                        active_job = job
                                        # Update SSH variables from reloaded job
                                        ssh_host = active_job.get("ssh_host")
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                        else:
                                            ssh_port = active_job.get("ssh_port", 22)
                                        break
                                
                                # Get fresh job status from manager FIRST (this updates the job with latest info from API)
                                # This ensures we have the most up-to-date status before checking training
                                updated_job_status = training_manager.get_job_status(instance_id)
                                if updated_job_status and not updated_job_status.get("error"):
                                    # Update active_job with fresh data (including status, SSH info, etc.)
                                    active_job.update(updated_job_status)
                                    # Also update the active_jobs list so it's reflected in the UI
                                    for idx, job in enumerate(active_jobs):
                                        if job.get("instance_id") == instance_id:
                                            active_jobs[idx] = active_job
                                            break
                                    # Update SSH variables from updated job status (if SSH info is in the update)
                                    if updated_job_status.get("ssh_host"):
                                        ssh_host = updated_job_status.get("ssh_host")
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                        else:
                                            ssh_port = updated_job_status.get("ssh_port", 22)
                                
                                terminal_output.append(f"[DEBUG] Calling check_training_status for instance {instance_id}")
                                terminal_output.append(f"[DEBUG] Using SSH: {ssh_host}:{ssh_port}")
                                
                                training_status = training_manager.check_training_status(instance_id)
                                
                                # Log the raw response for debugging
                                terminal_output.append(f"[DEBUG] check_training_status returned: status={training_status.get('status') if training_status else 'None'}")
                                if training_status:
                                    terminal_output.append(f"[DEBUG] Training status details:")
                                    terminal_output.append(f"[DEBUG]   - training_running: {training_status.get('training_running')}")
                                    terminal_output.append(f"[DEBUG]   - training_started: {training_status.get('training_started')}")
                                    terminal_output.append(f"[DEBUG]   - training_files_exist: {training_status.get('training_files_exist')}")
                                    terminal_output.append(f"[DEBUG]   - has_output: {training_status.get('has_output')}")
                                    terminal_output.append(f"[DEBUG]   - logs_length: {training_status.get('logs_length', 0)}")
                                    if training_status.get('error'):
                                        terminal_output.append(f"[DEBUG]   - error: {training_status.get('error')}")
                                    if training_status.get('failure_reason'):
                                        terminal_output.append(f"[DEBUG]   - failure_reason: {training_status.get('failure_reason')[:200]}")
                                
                                # Ensure training_status is a dict
                                if training_status is None:
                                    training_status = {}
                                    terminal_output.append(f"[WARNING] check_training_status returned None - using empty dict")
                                    
                                    # Override status if we found errors in logs
                                    if training_error and training_status.get("status") != "completed":
                                        training_status["status"] = "failed"
                                        # Extract a concise error message
                                        error_lines = training_error.split("\n")
                                        for line in reversed(error_lines):
                                            if any(keyword in line for keyword in ["Error", "Exception", "AttributeError", "ModuleNotFoundError", "ImportError", "OutOfMemoryError"]):
                                                training_status["failure_reason"] = line.strip()
                                                break
                                        if "failure_reason" not in training_status:
                                            # Check if it's an OOM error
                                            if "OutOfMemoryError" in training_error or "out of memory" in training_error.lower() or "CUDA out of memory" in training_error:
                                                training_status["failure_reason"] = "CUDA Out of Memory Error - GPU memory exhausted"
                                            else:
                                                training_status["failure_reason"] = "Training error detected in logs (see training logs above)"
                                    
                                    # If SSH check shows training is running but status is unknown, set it to training
                                    if ssh_training_running is not None and ssh_training_running:
                                        if training_status.get("status") == "unknown" or not training_status.get("status"):
                                            training_status["status"] = "training"
                                            terminal_output.append(f"[INFO] Training process detected via SSH - setting status to 'training'")
                                    
                                    active_job["training_status"] = training_status
                                    training_manager._save_job(active_job)
                                    
                                    # Reload job again to get any status updates from get_job_status
                                    updated_job_status = training_manager.get_job_status(instance_id)
                                    if updated_job_status and not updated_job_status.get("error"):
                                        # Update active_job with fresh data
                                        active_job.update(updated_job_status)
                                    
                                    status_val = training_status.get("status", "unknown")
                                    terminal_output.append(f"[DEBUG] Initial status_val from training_status: {status_val}")
                                    st.session_state[terminal_output_key] = list(terminal_output)  # Save after status determination
                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                    
                                    # Check for completion indicators in logs BEFORE checking for preprocessing
                                    # This ensures completion takes priority over preprocessing detection
                                    if all_logs_content:
                                        # Strong completion indicators (definitive)
                                        strong_completion_indicators = [
                                            "training completed",
                                            "model successfully saved",
                                            "saving trained model"
                                        ]
                                        
                                        # Check for strong indicators
                                        has_strong_indicator = any(indicator in all_logs_content.lower() for indicator in strong_completion_indicators)
                                        # Check for both completion AND saved indicators (most definitive)
                                        has_both_indicators = "training completed" in all_logs_content.lower() and "model successfully saved" in all_logs_content.lower()
                                        # Also check that no process is running (process has exited)
                                        process_exited = not ssh_training_running if ssh_training_running is not None else False
                                        
                                        terminal_output.append(f"[DEBUG] Completion check: has_strong_indicator={has_strong_indicator}, has_both_indicators={has_both_indicators}, process_exited={process_exited}, status_val={status_val}")
                                        terminal_output.append(f"[DEBUG] all_logs_content length: {len(all_logs_content) if all_logs_content else 0}")
                                        terminal_output.append(f"[DEBUG] ssh_training_running: {ssh_training_running}")
                                        st.session_state[terminal_output_key] = terminal_output
                                        
                                        # If we have both indicators, consider it completed even if process hasn't exited yet
                                        # (process might still be cleaning up or the check might be delayed)
                                        # Priority: has_both_indicators > (has_strong_indicator and process_exited)
                                        if has_both_indicators:
                                            is_completed = True
                                            terminal_output.append(f"[DEBUG] Both completion indicators found - treating as completed regardless of process status")
                                        else:
                                            is_completed = has_strong_indicator and process_exited
                                        
                                        terminal_output.append(f"[DEBUG] is_completed={is_completed}, status_val={status_val}")
                                        st.session_state[terminal_output_key] = terminal_output
                                        
                                        if is_completed and status_val != "completed":
                                            # Override status to completed if we see completion indicators and process has exited
                                            terminal_output.append(f"[INFO] ✓ Detected training completion from logs (process has exited)")
                                            terminal_output.append(f"[INFO] Strong indicator found: {[ind for ind in strong_completion_indicators if ind in all_logs_content.lower()]}")
                                            st.session_state[terminal_output_key] = terminal_output
                                            status_val = "completed"
                                            training_status["status"] = "completed"
                                            active_job["training_status"] = training_status
                                            training_manager._save_job(active_job)
                                            terminal_output.append(f"[INFO] ✓ Status updated to 'completed' and saved to job")
                                            terminal_output.append(f"[INFO] status_val is now: {status_val}")
                                            st.session_state[terminal_output_key] = terminal_output
                                        elif status_val == "completed":
                                            terminal_output.append(f"[DEBUG] Status already set to 'completed' - proceeding to queue transition check...")
                                            st.session_state[terminal_output_key] = terminal_output
                                        else:
                                            terminal_output.append(f"[DEBUG] Completion not detected: is_completed={is_completed}, status_val={status_val}")
                                            st.session_state[terminal_output_key] = terminal_output
                                    else:
                                        terminal_output.append(f"[DEBUG] Completion check skipped: all_logs_content is empty")
                                        st.session_state[terminal_output_key] = terminal_output
                                    
                                    # Save status before showing it - ensure we save after completion detection
                                    st.session_state[terminal_output_key] = terminal_output
                                    
                                    terminal_output.append(f"[INFO] Training status: {status_val}")
                                    terminal_output.append(f"[DEBUG] About to check if status_val == 'completed'...")
                                    st.session_state[terminal_output_key] = terminal_output
                                    
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
                                                # Check for active training step progress first (most specific)
                                                import re
                                                training_step_patterns = [
                                                    r'\|\s+\d+/\d+\s+\|',  # Progress bar format: "| 1/393 |" or "| 2/393 |"
                                                    r'\d+/\d+\s+\[',  # Progress bar format: "1/393 [" or "2/393 ["
                                                    r'step\s+\d+/\d+',  # "step 1/393", "step 2/393"
                                                    r'step\s+\d+',  # "step 1", "step 2"
                                                    r'epoch\s+\d+.*loss',  # "epoch 1 loss: 0.5"
                                                    r'loss.*step',  # "loss: 0.5 at step 1"
                                                    r'train.*loss.*\d+\.\d+',  # "train_loss: 0.5234"
                                                    r'eval.*loss.*\d+\.\d+',  # "eval_loss: 0.5234"
                                                ]
                                                is_training_active = False
                                                if all_logs_content:
                                                    for pattern in training_step_patterns:
                                                        if re.search(pattern, all_logs_content, re.IGNORECASE):
                                                            is_training_active = True
                                                            break
                                                
                                                # Check for preprocessing indicators
                                                preprocessing_indicators = ["tokenizing", "preprocessing", "dropping", "saving the dataset", "sample packing"]
                                                is_preprocessing = any(indicator in all_logs_content.lower() if all_logs_content else False for indicator in preprocessing_indicators)
                                                
                                                # Check if preprocessing is complete (look for 100% completion)
                                                preprocessing_complete = False
                                                if all_logs_content:
                                                    # Check if we see 100% completion for preprocessing steps
                                                    preprocessing_complete_patterns = [
                                                        r'dropping.*100%',
                                                        r'saving the dataset.*100%',
                                                        r'maximum number of steps set at',  # This appears after preprocessing
                                                    ]
                                                    for pattern in preprocessing_complete_patterns:
                                                        if re.search(pattern, all_logs_content, re.IGNORECASE):
                                                            preprocessing_complete = True
                                                            break
                                                
                                                # Check for model loading indicators (after preprocessing, before training starts)
                                                model_loading_indicators = ["loading checkpoint", "loading model", "loading tokenizer", "memory usage after model load"]
                                                is_model_loading = any(indicator in all_logs_content.lower() if all_logs_content else False for indicator in model_loading_indicators)
                                                
                                                # Check if model loading is complete (checkpoint shards loaded)
                                                model_loading_complete = False
                                                if all_logs_content:
                                                    # Look for "Loading checkpoint shards: 100%" which indicates model loading is done
                                                    if re.search(r'loading checkpoint shards.*100%', all_logs_content, re.IGNORECASE):
                                                        model_loading_complete = True
                                                
                                                if is_training_active:
                                                    # Extract step information if available - try multiple formats
                                                    step_match = None
                                                    # Try progress bar format first: "| 1/393 |" or "1/393 ["
                                                    step_match = re.search(r'\|\s+(\d+)/(\d+)\s+\|', all_logs_content)
                                                    if not step_match:
                                                        step_match = re.search(r'(\d+)/(\d+)\s+\[', all_logs_content)
                                                    # Try standard step format: "step 1/393"
                                                    if not step_match:
                                                        step_match = re.search(r'step\s+(\d+)/(\d+)', all_logs_content, re.IGNORECASE)
                                                    
                                                    if step_match:
                                                        current_step = int(step_match.group(1))
                                                        total_steps = int(step_match.group(2))
                                                        
                                                        # Extract epoch information if available
                                                        epoch_match = re.search(r"'epoch':\s*([\d.]+)", all_logs_content)
                                                        current_epoch = None
                                                        if epoch_match:
                                                            current_epoch = float(epoch_match.group(1))
                                                        
                                                        # Calculate time estimates
                                                        # Try to extract time per step from progress bar: "33.88s/it"
                                                        time_per_step_match = re.search(r'\[.*?([\d.]+)s/it\]', all_logs_content)
                                                        time_per_step = None
                                                        if time_per_step_match:
                                                            time_per_step = float(time_per_step_match.group(1))
                                                        
                                                        # Build status message
                                                        status_msg = f"[INFO] ✓ Training is active - Step {current_step}/{total_steps}"
                                                        if current_epoch is not None:
                                                            # Calculate estimated total epochs
                                                            if current_step > 0:
                                                                steps_per_epoch = current_step / current_epoch
                                                                estimated_total_epochs = total_steps / steps_per_epoch
                                                                status_msg += f" (Epoch {current_epoch:.2f}, ~{estimated_total_epochs:.1f} epochs total)"
                                                        
                                                        # Add time estimate
                                                        if time_per_step:
                                                            remaining_steps = total_steps - current_step
                                                            remaining_seconds = remaining_steps * time_per_step
                                                            remaining_hours = remaining_seconds / 3600
                                                            status_msg += f" | Est. remaining: {remaining_hours:.1f} hours"
                                                        
                                                        terminal_output.append(status_msg)
                                                    else:
                                                        terminal_output.append(f"[INFO] ✓ Training is active - training steps detected in logs")
                                                    terminal_output.append(f"[INFO] Check the training logs above to see current progress and loss values.")
                                                    # Mark that we detected active training
                                                    status_val = "training"
                                                    training_status["status"] = "training"
                                                    active_job["training_status"] = training_status
                                                    training_manager._save_job(active_job)
                                                elif model_loading_complete or (is_model_loading and preprocessing_complete):
                                                    # Model loading is complete or in progress after preprocessing - prioritize this
                                                    if model_loading_complete:
                                                        terminal_output.append(f"[INFO] ✓ Model loading completed - training should start soon.")
                                                        terminal_output.append(f"[INFO] Check the training logs above to see when training steps begin.")
                                                    else:
                                                        terminal_output.append(f"[INFO] ✓ Model is loading - training will begin once initialization is complete.")
                                                        terminal_output.append(f"[INFO] Check the training logs above to see when model loading completes.")
                                                    status_val = "training"
                                                    training_status["status"] = "training"
                                                    active_job["training_status"] = training_status
                                                    training_manager._save_job(active_job)
                                                elif is_preprocessing and not preprocessing_complete:
                                                    terminal_output.append(f"[INFO] ✓ Training is active - preprocessing data (tokenizing, preparing dataset).")
                                                    terminal_output.append(f"[INFO] This is normal and can take several minutes depending on dataset size.")
                                                    terminal_output.append(f"[INFO] Check the logs above to see preprocessing progress.")
                                                    # Mark that we detected preprocessing so we don't show "unclear" message later
                                                    status_val = "training"  # Override status to training when preprocessing is detected
                                                    # Update training_status to reflect this
                                                    training_status["status"] = "training"
                                                    active_job["training_status"] = training_status
                                                    training_manager._save_job(active_job)
                                                elif is_model_loading:
                                                    terminal_output.append(f"[INFO] ✓ Model loading completed - training should start soon.")
                                                    terminal_output.append(f"[INFO] Training will begin once model initialization is complete.")
                                                    terminal_output.append(f"[INFO] Check the training logs above to see when training steps begin.")
                                                    status_val = "training"
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
                                    
                                    # Check status_val after all the completion detection logic
                                    terminal_output.append(f"[DEBUG] ========================================")
                                    terminal_output.append(f"[DEBUG] Final status_val: '{status_val}'")
                                    terminal_output.append(f"[DEBUG] Will check queue transition: {status_val == 'completed'}")
                                    terminal_output.append(f"[DEBUG] status_val type: {type(status_val).__name__}")
                                    terminal_output.append(f"[DEBUG] status_val == 'completed': {status_val == 'completed'}")
                                    # Force save before checking
                                    st.session_state[terminal_output_key] = terminal_output
                                    
                                    # CRITICAL: Check if status_val is actually "completed" and trigger queue transition
                                    if str(status_val) == "completed":
                                        terminal_output.append(f"[QUEUE] ========================================")
                                        terminal_output.append(f"[QUEUE] ✓ Status is 'completed' - ENTERING QUEUE TRANSITION")
                                        st.session_state[terminal_output_key] = terminal_output
                                        terminal_output.append(f"[DEBUG] ✓ Status is 'completed' - entering queue transition block")
                                        st.session_state[terminal_output_key] = terminal_output
                                        terminal_output.append(f"[SUCCESS] ✓ Training completed!")
                                        terminal_output.append(f"[QUEUE] Checking if queue transition is needed...")
                                        st.session_state[terminal_output_key] = terminal_output
                                        
                                        # Verify adapter files are saved
                                        terminal_output.append(f"[VERIFY] Checking if LoRA adapter files are saved...")
                                        st.session_state[terminal_output_key] = terminal_output
                                        
                                        # Get SSH info for adapter verification
                                        ssh_host = active_job.get("ssh_host")
                                        # Check for SSH port override first (user-specified port takes precedence)
                                        ssh_port_override = active_job.get("ssh_port_override")
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                        else:
                                            ssh_port = active_job.get("ssh_port", 22)
                                        
                                        if not ssh_host:
                                            job_status = training_manager.get_job_status(instance_id)
                                            ssh_host = job_status.get("ssh_host")
                                            api_ssh_port = job_status.get("ssh_port", 22)
                                            
                                            # Use override port if set, otherwise use API port
                                            if ssh_port_override:
                                                ssh_port = ssh_port_override
                                            else:
                                                ssh_port = api_ssh_port
                                        
                                        if ssh_host:
                                            check_adapter_files_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                """bash -c '
                                                    adapter_found=false
                                                    if [ -f /workspace/output/training/adapter/adapter_config.json ]; then
                                                        echo "adapter_location:/workspace/output/training/adapter"
                                                        ls -lh /workspace/output/training/adapter/ | head -10
                                                        adapter_found=true
                                                    elif [ -f /workspace/output/training/adapter_config.json ]; then
                                                        echo "adapter_location:/workspace/output/training"
                                                        ls -lh /workspace/output/training/adapter* 2>/dev/null | head -10
                                                        adapter_found=true
                                                    fi
                                                    if [ "$adapter_found" = false ]; then
                                                        echo "adapter_not_found"
                                                        echo "Checking for any adapter files..."
                                                        find /workspace/output -name "adapter_config.json" -o -name "adapter_model.bin" -o -name "adapter_model.safetensors" 2>/dev/null | head -10
                                                    fi
                                                '"""
                                            ]
                                            adapter_check_result = subprocess.run(check_adapter_files_cmd, capture_output=True, text=True, timeout=15)
                                            if "adapter_location:" in adapter_check_result.stdout:
                                                adapter_loc = [line for line in adapter_check_result.stdout.split('\n') if 'adapter_location:' in line][0].split('adapter_location:')[1].strip()
                                                terminal_output.append(f"[VERIFY] ✓ LoRA adapter files found at: {adapter_loc}")
                                                # Show file listing
                                                for line in adapter_check_result.stdout.split('\n'):
                                                    if line.strip() and 'adapter_location:' not in line:
                                                        terminal_output.append(f"[VERIFY]   {line}")
                                            elif "adapter_not_found" in adapter_check_result.stdout:
                                                terminal_output.append(f"[WARNING] ⚠ LoRA adapter files not found in expected locations!")
                                                terminal_output.append(f"[WARNING]   Checked: /workspace/output/training/adapter/ and /workspace/output/training/")
                                                # Show what was found
                                                for line in adapter_check_result.stdout.split('\n'):
                                                    if line.strip() and 'adapter_not_found' not in line and 'Checking' not in line:
                                                        terminal_output.append(f"[VERIFY]   {line}")
                                            else:
                                                terminal_output.append(f"[VERIFY] Could not verify adapter files (check timed out or failed)")
                                        
                                        st.session_state[terminal_output_key] = terminal_output
                                        
                                        # Training completed - advance to Phase 4 (single job mode, no queue transitions)
                                        terminal_output.append(f"[SUCCESS] ========================================")
                                        terminal_output.append(f"[SUCCESS] Training completed!")
                                        terminal_output.append(f"[SUCCESS] ========================================")
                                        terminal_output.append(f"[INFO] Advancing to Phase 4: Finalize (downloading weights)")
                                        terminal_output.append(f"[INFO] No queue transitions - single job mode")
                                        
                                        # Auto-advance to Phase 4 (no queue/job serialization)
                                        # Save terminal output before transition
                                        st.session_state[terminal_output_key] = terminal_output
                                        # Advance to Phase 4
                                        st.session_state[phase_key] = 4
                                        # Trigger rerun to show Phase 4
                                        st.rerun()
                                        
                                        # NOTE: Queue transition code removed - we no longer serialize jobs
                                        # This code should not be reached, but keeping for safety
                                        if False:
                                            terminal_output.append(f"[QUEUE] ========================================")
                                            terminal_output.append(f"[QUEUE] ✓ Starting queue transition from job {current_job_index + 1} to job {current_job_index + 2}...")
                                            st.session_state[terminal_output_key] = terminal_output
                                            
                                            # More jobs in queue - merge adapter from completed job into next job
                                            next_job_index = current_job_index + 1
                                            next_job = job_queue[next_job_index]
                                            
                                            terminal_output.append(f"[QUEUE] Job {current_job_index + 1} complete. Starting job {next_job_index + 1}/{len(job_queue)}...")
                                            terminal_output.append(f"[MERGE] Merging adapter from job {current_job_index + 1} into job {next_job_index + 1}...")
                                            st.session_state[terminal_output_key] = terminal_output
                                            
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
                                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                        f"root@{ssh_host}",
                                                        f"ls -la /workspace/output/training/ 2>/dev/null | head -5 && "
                                                        f"rm -rf /workspace/output/training/* && "
                                                        f"mkdir -p /workspace/output/training && "
                                                        f"ls -la /workspace/output/training/ && "
                                                        f"echo 'Output directory cleaned'"
                                                    ]
                                                    cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                                    
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
                                                            f"    # Job 1 (index 0) = no suffix, Job 2+ (index 1+) = _{index} suffix\n"
                                                            f"    config_filename = 'axolotl_config.yaml' if {next_job_index} == 0 else f'axolotl_config_{next_job_index}.yaml'\n"
                                                            f"    with open(config_filename, 'r') as f:\n"
                                                            f"        config = yaml.safe_load(f) or {{}}\n"
                                                            f"    old_adapter = config.get('adapter', 'none')\n"
                                                            f"    config['adapter'] = '/workspace/data/previous_adapter_{current_job_index}'\n"
                                                            f"    with open(config_filename, 'w') as f:\n"
                                                            f"        yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                            f"    print(f'Config updated: adapter set to /workspace/data/previous_adapter_{current_job_index}')\n"
                                                            f"    print(f'Previous adapter value: {{old_adapter}}')\n"
                                                            f"except Exception as e:\n"
                                                            f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                                            f"    sys.exit(1)\n"
                                                            f"PYTHON_EOF"
                                                        )
                                                        update_config_cmd = [
                                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                    
                                                    # Step 5: Clean up old active files and activate next job's files
                                                    # Naming scheme: Job 1 (index 0) = no suffix, Job 2+ (index 1+) = _{index} suffix
                                                    terminal_output.append(f"[SSH] ========================================")
                                                    terminal_output.append(f"[SSH] Step 5: Cleaning up old active files and activating job {next_job_index + 1} files...")
                                                    terminal_output.append(f"[SSH]   Current active files (job {current_job_index + 1}): axolotl_config.yaml, training_data.jsonl")
                                                    
                                                    if next_job_index == 0:
                                                        # Job 1: Files should already be axolotl_config.yaml and training_data.jsonl (no suffix)
                                                        terminal_output.append(f"[SSH]   Job 1: Files should already be active (no suffix needed)")
                                                        terminal_output.append(f"[SSH]   Just removing previous job files and verifying job 1 files exist...")
                                                        rename_cmd = [
                                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                            f"root@{ssh_host}",
                                                            f"cd /workspace/data && "
                                                            f"echo 'Step 1: Removing previous job active files...' && "
                                                            f"rm -f axolotl_config.yaml training_data.jsonl && "
                                                            f"echo 'Step 2: Verifying job 1 files exist (they should have been uploaded without suffix)...' && "
                                                            f"if [ -f axolotl_config.yaml ] && [ -f training_data.jsonl ]; then "
                                                            f"  ls -la axolotl_config.yaml training_data.jsonl && "
                                                            f"  echo 'Job 1 files are ready'; "
                                                            f"else "
                                                            f"  echo 'ERROR: Job 1 files missing!'; "
                                                            f"  exit 1; "
                                                            f"fi"
                                                        ]
                                                    else:
                                                        # Job 2+: Files are uploaded with _{next_job_index} suffix, need to activate them
                                                        terminal_output.append(f"[SSH]   Files to activate (job {next_job_index + 1}): axolotl_config_{next_job_index}.yaml → axolotl_config.yaml")
                                                        terminal_output.append(f"[SSH]   Files to activate (job {next_job_index + 1}): training_data_{next_job_index}.jsonl → training_data.jsonl")
                                                        rename_cmd = [
                                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                            f"root@{ssh_host}",
                                                            f"cd /workspace/data && "
                                                            f"echo '=== File Rename Operation ===' && "
                                                            f"echo 'Step 1: Removing old active files (job {current_job_index + 1})...' && "
                                                            f"ls -la axolotl_config.yaml training_data.jsonl 2>/dev/null | head -5 && "
                                                            f"rm -f axolotl_config.yaml training_data.jsonl && "
                                                            f"echo 'Step 2: Activating job {next_job_index + 1} files (removing _{next_job_index} suffix)...' && "
                                                            f"ls -la axolotl_config_{next_job_index}.yaml training_data_{next_job_index}.jsonl 2>/dev/null | head -5 && "
                                                            f"mv axolotl_config_{next_job_index}.yaml axolotl_config.yaml && "
                                                            f"mv training_data_{next_job_index}.jsonl training_data.jsonl && "
                                                            f"echo 'Step 3: Verifying new active files...' && "
                                                            f"ls -la axolotl_config.yaml training_data.jsonl 2>/dev/null | head -5 && "
                                                            f"echo 'Files renamed successfully'"
                                                        ]
                                                    st.session_state[terminal_output_key] = terminal_output
                                                    rename_result = subprocess.run(rename_cmd, capture_output=True, text=True, timeout=30)
                                                    
                                                    # Log the rename operation output
                                                    if rename_result.stdout:
                                                        stdout_filtered = filter_malloc_warnings(rename_result.stdout)
                                                        for line in stdout_filtered.strip().split("\n"):
                                                            if line.strip() and "Files renamed successfully" not in line:
                                                                terminal_output.append(f"[SSH]   {line}")
                                                    if rename_result.stderr:
                                                        stderr_filtered = filter_malloc_warnings(rename_result.stderr)
                                                        if stderr_filtered.strip():
                                                            terminal_output.append(f"[SSH]   stderr: {stderr_filtered.strip()}")
                                                    st.session_state[terminal_output_key] = terminal_output
                                                    
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
                                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                            kill_all_training_processes(ssh_host, ssh_port, terminal_output)
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
                                                            "mkdir -p /workspace/output/training && "
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
                                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                        terminal_output.append(f"[QUEUE] ========================================")
                                                        terminal_output.append(f"[QUEUE] Queue transition complete!")
                                                        terminal_output.append(f"[QUEUE] Job {next_job_index + 1}/{len(job_queue)} is now running.")
                                                        terminal_output.append(f"[QUEUE] Use 'Check Training Status' to monitor progress.")
                                                        terminal_output.append(f"[QUEUE] ========================================")
                                                        st.session_state[terminal_output_key] = terminal_output
                                                        st.rerun()
                                                        
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
                                                terminal_output.append(f"[ERROR] Exception details: {str(e)}")
                                                import traceback
                                                terminal_output.append(f"[ERROR] Traceback: {traceback.format_exc()[:500]}")
                                                st.session_state[terminal_output_key] = terminal_output
                                                st.rerun()
                                    elif status_val == "training":
                                        terminal_output.append(f"[INFO] Training is actively running...")
                                        
                                        # Check for errors in logs even if training appears to be running
                                        # (training might be about to fail, or error occurred during preprocessing)
                                        if training_error and ssh_host:
                                            # Check if it's a multipack sampler error (this can happen during preprocessing)
                                            is_multipack_error = False
                                            if all_logs_content:
                                                has_index_error = "IndexError" in all_logs_content or "list index out of range" in all_logs_content.lower()
                                                has_batches_ref = "batches[-1]" in all_logs_content or "generate_batches" in all_logs_content
                                                has_multipack = "multipack" in all_logs_content.lower() or "samplers/multipack" in all_logs_content.lower() or "/multipack.py" in all_logs_content or "multipack.py" in all_logs_content
                                                
                                                if has_index_error and has_batches_ref and has_multipack:
                                                    is_multipack_error = True
                                                    terminal_output.append(f"[WARNING] Detected multipack sampler error in logs - training will likely fail soon")
                                                    terminal_output.append(f"[ACTION] Attempting to fix config proactively...")
                                                    st.session_state[terminal_output_key] = list(terminal_output)
                                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                                    
                                                    # Trigger the multipack fix (we'll reuse the same fix logic)
                                                    # Import the fix code inline to avoid duplication
                                                    try:
                                                        # Use the same fix logic as in the "failed" branch
                                                        terminal_output.append(f"[ACTION] Detected multipack sampler IndexError - checking if data survived filtering...")
                                                        st.session_state[terminal_output_key] = list(terminal_output)
                                                        st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                                        
                                                        # Check train_on_inputs and data survival, then fix
                                                        # (This will be handled by the same code block below)
                                                        # For now, just trigger the fix directly
                                                        fix_multipack_remote_cmd = (
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
                                                            f"    # ALWAYS ensure train_on_inputs is True to maximize sample retention\n"
                                                            f"    if not config.get('train_on_inputs', True):\n"
                                                            f"        config['train_on_inputs'] = True\n"
                                                            f"        fixed = True\n"
                                                            f"        print('Set train_on_inputs=True to maximize sample retention')\n"
                                                            f"    \n"
                                                            f"    # Disable multipack sampler (safest fix)\n"
                                                            f"    if config.get('sample_packing', True):\n"
                                                            f"        config['sample_packing'] = False\n"
                                                            f"        fixed = True\n"
                                                            f"        print('Disabled sample_packing (multipack sampler)')\n"
                                                            f"    \n"
                                                            f"    if fixed:\n"
                                                            f"        with open('axolotl_config.yaml', 'w') as f:\n"
                                                            f"            yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                            f"        print('Config fixed successfully for multipack sampler error')\n"
                                                            f"    else:\n"
                                                            f"        print('Config already adjusted')\n"
                                                            f"except Exception as e:\n"
                                                            f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                                            f"    sys.exit(1)\n"
                                                            f"PYTHON_EOF"
                                                        )
                                                        fix_multipack_cmd = [
                                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                            f"root@{ssh_host}",
                                                            fix_multipack_remote_cmd
                                                        ]
                                                        fix_multipack_result = subprocess.run(fix_multipack_cmd, capture_output=True, text=True, timeout=30)
                                                        if fix_multipack_result.returncode == 0 and "fixed successfully" in fix_multipack_result.stdout:
                                                            terminal_output.append(f"[SUCCESS] Config file fixed for multipack sampler error!")
                                                            terminal_output.append(f"[INFO] Changes: Disabled sample_packing, set train_on_inputs=True")
                                                            terminal_output.append(f"[INFO] ⚠️ Training is still running but will likely fail. Stop training and restart with 'Redo Phase' to use the fixed config.")
                                                            st.session_state[terminal_output_key] = list(terminal_output)
                                                            st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                                        elif fix_multipack_result.returncode == 0:
                                                            terminal_output.append(f"[INFO] Config check completed: {fix_multipack_result.stdout.strip()}")
                                                            st.session_state[terminal_output_key] = list(terminal_output)
                                                            st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                                        else:
                                                            error_msg = fix_multipack_result.stderr or fix_multipack_result.stdout
                                                            terminal_output.append(f"[WARNING] Could not fix config: {error_msg[:200]}")
                                                            st.session_state[terminal_output_key] = list(terminal_output)
                                                            st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                                    except Exception as e:
                                                        terminal_output.append(f"[WARNING] Error attempting to fix config: {str(e)}")
                                                        st.session_state[terminal_output_key] = list(terminal_output)
                                                        st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
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
                                                    f"    fixed = False\n"
                                                    f"    \n"
                                                    f"    # Ensure adapter: 'lora' is set if lora_* parameters exist (required for LoRA mode)\n"
                                                    f"    # BUT preserve adapter paths for incremental training (V2+)\n"
                                                    f"    # The path issue is prevented by NOT setting lora_model_dir to output_dir\n"
                                                    f"    has_lora_params = config.get('lora_r') is not None and config.get('lora_alpha') is not None\n"
                                                    f"    if 'adapter' in config:\n"
                                                    f"        adapter_val = config['adapter']\n"
                                                    f"        # If adapter is set to an invalid string (not 'lora' and not a path), remove it\n"
                                                    f"        if isinstance(adapter_val, str) and adapter_val != 'lora' and not adapter_val.startswith('/'):\n"
                                                    f"            del config['adapter']\n"
                                                    f"            fixed = True\n"
                                                    f"            print('Removed invalid adapter field')\n"
                                                    f"        elif adapter_val == 'lora':\n"
                                                    f"            # Keep adapter: 'lora' - it's required for LoRA mode\n"
                                                    f"            print('Adapter: lora already set (LoRA mode enabled)')\n"
                                                    f"        elif isinstance(adapter_val, str) and adapter_val.startswith('/'):\n"
                                                    f"            # Keep adapter path for incremental training (V2+)\n"
                                                    f"            print(f'Keeping adapter path for incremental training: {{adapter_val}}')\n"
                                                    f"    elif has_lora_params:\n"
                                                    f"        # Add adapter: 'lora' if lora_* parameters exist but adapter is missing\n"
                                                    f"        # Only for new training (not incremental)\n"
                                                    f"        config['adapter'] = 'lora'\n"
                                                    f"        fixed = True\n"
                                                    f"        print('Added adapter: lora (required for LoRA mode)')\n"
                                                    f"    \n"
                                                    f"    # CRITICAL: Remove lora_model_dir if it's set to output_dir (causes Axolotl to try loading from there)\n"
                                                    f"    # lora_model_dir should only be set to a valid adapter path (for incremental training)\n"
                                                    f"    output_dir = config.get('output_dir', '')\n"
                                                    f"    lora_model_dir = config.get('lora_model_dir', '')\n"
                                                    f"    adapter_val = config.get('adapter', '')\n"
                                                    f"    \n"
                                                    f"    # If lora_model_dir is set to output_dir, always remove it (unless there's a valid adapter path)\n"
                                                    f"    if lora_model_dir == output_dir:\n"
                                                    f"        # Only keep it if adapter is set to a valid path (incremental training)\n"
                                                    f"        if not adapter_val or not isinstance(adapter_val, str) or not adapter_val.startswith('/'):\n"
                                                    f"            del config['lora_model_dir']\n"
                                                    f"            fixed = True\n"
                                                    f"            print('Removed lora_model_dir (was pointing to output_dir - causes adapter loading errors)')\n"
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
                                                    fix_adapter_cmd = [
                                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                        
                                        # Check if it's a multipack sampler IndexError (batch configuration issue)
                                        # Check for IndexError with multipack sampler (can be in file path or error message)
                                        # Make the condition more lenient - check if it's an IndexError with batches reference
                                        # First check in all_logs_content (has more context including file paths)
                                        is_multipack_error = False
                                        if all_logs_content:
                                            has_index_error = "IndexError" in all_logs_content or "list index out of range" in all_logs_content.lower()
                                            has_batches_ref = "batches[-1]" in all_logs_content or "generate_batches" in all_logs_content
                                            has_multipack = "multipack" in all_logs_content.lower() or "samplers/multipack" in all_logs_content.lower() or "/multipack.py" in all_logs_content or "multipack.py" in all_logs_content
                                            
                                            if has_index_error and has_batches_ref and has_multipack:
                                                is_multipack_error = True
                                                terminal_output.append(f"[DEBUG] Detected multipack error in all_logs_content")
                                        
                                        # Also check in training_error (extracted error)
                                        if not is_multipack_error and training_error:
                                            has_index_error = "IndexError" in training_error or "list index out of range" in training_error.lower()
                                            has_batches_ref = "batches[-1]" in training_error or "generate_batches" in training_error
                                            has_multipack = "multipack" in training_error.lower() or "samplers/multipack" in training_error.lower() or "/multipack.py" in training_error or "multipack.py" in training_error
                                            
                                            if has_index_error and has_batches_ref and has_multipack:
                                                is_multipack_error = True
                                                terminal_output.append(f"[DEBUG] Detected multipack error in training_error")
                                        
                                        if is_multipack_error:
                                            terminal_output.append(f"[ACTION] Detected multipack sampler IndexError - checking if data survived filtering...")
                                            st.session_state[terminal_output_key] = terminal_output  # Save immediately so user sees the message
                                            
                                            # First, check if train_on_inputs is set correctly (this is critical for sample retention)
                                            check_train_on_inputs_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "cd /workspace/data && python3 << 'PYTHON_EOF'\n"
                                                "import yaml\n"
                                                "try:\n"
                                                "    with open('axolotl_config.yaml', 'r') as f:\n"
                                                "        config = yaml.safe_load(f) or {}\n"
                                                "    train_on_inputs = config.get('train_on_inputs', None)\n"
                                                "    print(f'{train_on_inputs}')\n"
                                                "except Exception as e:\n"
                                                "    print(f'error: {e}')\n"
                                                "PYTHON_EOF"
                                            ]
                                            train_on_inputs_set = None
                                            try:
                                                train_on_inputs_result = subprocess.run(check_train_on_inputs_cmd, capture_output=True, text=True, timeout=10)
                                                if train_on_inputs_result.returncode == 0:
                                                    output = train_on_inputs_result.stdout.strip()
                                                    if output == "True":
                                                        train_on_inputs_set = True
                                                        terminal_output.append(f"[DIAGNOSTIC] ✓ train_on_inputs is set to True (maximizes sample retention)")
                                                    elif output == "False":
                                                        train_on_inputs_set = False
                                                        terminal_output.append(f"[DIAGNOSTIC] ✗ WARNING: train_on_inputs is False - this may cause samples to be dropped!")
                                                        terminal_output.append(f"[DIAGNOSTIC]   Setting train_on_inputs=True is critical to prevent filtering out samples.")
                                                    elif output == "None":
                                                        train_on_inputs_set = None
                                                        terminal_output.append(f"[DIAGNOSTIC] ⚠ train_on_inputs is not set in config - will be set to True")
                                                    else:
                                                        terminal_output.append(f"[DIAGNOSTIC] Could not determine train_on_inputs setting: {output}")
                                            except Exception as e:
                                                        terminal_output.append(f"[DIAGNOSTIC] Could not check train_on_inputs setting: {str(e)}")
                                            st.session_state[terminal_output_key] = terminal_output  # Save after train_on_inputs check
                                            
                                            # Initialize variables for data survival check
                                            data_survived = None  # None = unknown, True = yes, False = no
                                            survived_count = 0
                                            
                                            # First, check if data actually survived filtering
                                            check_survived_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "python3 << 'PYTHON_EOF'\n"
                                                "import os\n"
                                                "import json\n"
                                                "# Check multiple possible locations for prepared dataset\n"
                                                "locations = [\n"
                                                "    '/workspace/axolotl/prepared_data/train.jsonl',\n"
                                                "    '/workspace/data/prepared_data/train.jsonl',\n"
                                                "    '/workspace/output/prepared_data/train.jsonl',\n"
                                                "    '/workspace/axolotl/data/train.jsonl',\n"
                                                "]\n"
                                                "count = 0\n"
                                                "found_path = None\n"
                                                "for path in locations:\n"
                                                "    if os.path.exists(path):\n"
                                                "        found_path = path\n"
                                                "        with open(path, 'r') as f:\n"
                                                "            count = sum(1 for line in f if line.strip())\n"
                                                "        break\n"
                                                "# Also try to find any train.jsonl file\n"
                                                "if count == 0:\n"
                                                "    import subprocess\n"
                                                "    result = subprocess.run(['find', '/workspace', '-name', 'train.jsonl', '-type', 'f', '2>/dev/null'], capture_output=True, text=True, timeout=5)\n"
                                                "    if result.returncode == 0 and result.stdout.strip():\n"
                                                "        found_path = result.stdout.strip().split('\\n')[0]\n"
                                                "        with open(found_path, 'r') as f:\n"
                                                "            count = sum(1 for line in f if line.strip())\n"
                                                "print(f'{count}|{found_path or \"not_found\"}')\n"
                                                "PYTHON_EOF"
                                            ]
                                            try:
                                                survived_result = subprocess.run(check_survived_cmd, capture_output=True, text=True, timeout=15)
                                                if survived_result.returncode == 0 and '|' in survived_result.stdout:
                                                    parts = survived_result.stdout.strip().split('|')
                                                    if len(parts) == 2:
                                                        survived_count = int(parts[0]) if parts[0].isdigit() else 0
                                                        dataset_path = parts[1] if parts[1] != 'not_found' else None
                                                        data_survived = survived_count > 0
                                                        
                                                        if survived_count > 0:
                                                            terminal_output.append(f"[DIAGNOSTIC] ✓ Data survived filtering: {survived_count} samples found in prepared dataset")
                                                            if dataset_path:
                                                                terminal_output.append(f"[DIAGNOSTIC]   Dataset location: {dataset_path}")
                                                            terminal_output.append(f"[DIAGNOSTIC] The error is likely a batch configuration issue, not data filtering.")
                                                            terminal_output.append(f"[DIAGNOSTIC] The multipack sampler cannot create batches with current batch_size/sequence_len settings.")
                                                        else:
                                                            terminal_output.append(f"[DIAGNOSTIC] ✗ WARNING: No data survived filtering!")
                                                            terminal_output.append(f"[DIAGNOSTIC] The prepared dataset is empty or not found.")
                                                            terminal_output.append(f"[DIAGNOSTIC] This means all {training_stats.get('original_count', 'original')} samples were filtered out.")
                                                            terminal_output.append(f"[DIAGNOSTIC] Possible causes:")
                                                            terminal_output.append(f"[DIAGNOSTIC]   - All sequences are too long for sequence_len setting")
                                                            terminal_output.append(f"[DIAGNOSTIC]   - All samples have zero trainable tokens (check train_on_inputs setting)")
                                                            terminal_output.append(f"[DIAGNOSTIC]   - Dataset format issues")
                                            except Exception as e:
                                                terminal_output.append(f"[DIAGNOSTIC] Could not check if data survived: {str(e)}")
                                                data_survived = None  # Unknown
                                            st.session_state[terminal_output_key] = terminal_output  # Save after data survival check
                                            
                                            terminal_output.append(f"[ACTION] Attempting to fix config by adjusting batch configuration...")
                                            st.session_state[terminal_output_key] = terminal_output  # Save before fix attempt
                                            if data_survived is False:
                                                terminal_output.append(f"[INFO] Since no data survived, will try to fix filtering settings (increase sequence_len, enable train_on_inputs).")
                                            elif data_survived is True:
                                                terminal_output.append(f"[INFO] Data survived filtering - will fix batch configuration (disable sample_packing, adjust batch size).")
                                            else:
                                                terminal_output.append(f"[INFO] Could not determine if data survived - will try both batch config and filtering fixes.")
                                            try:
                                                
                                                fix_multipack_remote_cmd = (
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
                                                    f"    # ALWAYS ensure train_on_inputs is True to maximize sample retention\n"
                                                    f"    # This is critical - prevents dropping samples where input has no trainable tokens\n"
                                                    f"    if not config.get('train_on_inputs', True):\n"
                                                    f"        config['train_on_inputs'] = True\n"
                                                    f"        fixed = True\n"
                                                    f"        print('Set train_on_inputs=True to maximize sample retention')\n"
                                                )
                                                
                                                # If no data survived (or unknown), try to fix filtering issues
                                                if data_survived is False:
                                                    fix_multipack_remote_cmd += (
                                                        f"    \n"
                                                        f"    # No data survived - fix filtering issues\n"
                                                        f"    # Option 1: Increase sequence length to retain more samples\n"
                                                        f"    current_seq_len = config.get('sequence_len', 2048)\n"
                                                        f"    if current_seq_len < 4096:\n"
                                                        f"        config['sequence_len'] = 4096\n"
                                                        f"        fixed = True\n"
                                                        f"        print(f'Increased sequence_len from {{current_seq_len}} to 4096 to retain more samples')\n"
                                                        f"    \n"
                                                        f"    # Option 2: Disable sample packing for small/empty datasets\n"
                                                        f"    if config.get('sample_packing', True):\n"
                                                        f"        config['sample_packing'] = False\n"
                                                        f"        fixed = True\n"
                                                        f"        print('Disabled sample_packing (no data survived filtering)')\n"
                                                    )
                                                elif data_survived is True:
                                                    # Data survived - fix batch configuration
                                                    fix_multipack_remote_cmd += (
                                                        f"    \n"
                                                        f"    # Data survived - fix batch configuration\n"
                                                        f"    # Option 1: Disable multipack sampler (safest fix)\n"
                                                        f"    if config.get('sample_packing', True):\n"
                                                        f"        config['sample_packing'] = False\n"
                                                        f"        fixed = True\n"
                                                        f"        print('Disabled sample_packing (multipack sampler)')\n"
                                                        f"    \n"
                                                        f"    # Option 2: Reduce batch size if it's too large\n"
                                                        f"    current_batch = config.get('micro_batch_size', 4)\n"
                                                        f"    if current_batch > 2:\n"
                                                        f"        config['micro_batch_size'] = 2\n"
                                                        f"        # Increase gradient accumulation to maintain effective batch size\n"
                                                        f"        current_grad_accum = config.get('gradient_accumulation_steps', 4)\n"
                                                        f"        config['gradient_accumulation_steps'] = current_grad_accum * 2\n"
                                                        f"        fixed = True\n"
                                                        f"        print(f'Reduced micro_batch_size from {{current_batch}} to 2, increased gradient_accumulation_steps to {{config[\"gradient_accumulation_steps\"]}}')\n"
                                                        f"    \n"
                                                        f"    # Option 3: Reduce sequence length if it's very long\n"
                                                        f"    current_seq_len = config.get('sequence_len', 2048)\n"
                                                        f"    if current_seq_len > 2048:\n"
                                                        f"        config['sequence_len'] = 2048\n"
                                                        f"        fixed = True\n"
                                                        f"        print(f'Reduced sequence_len from {{current_seq_len}} to 2048')\n"
                                                    )
                                                else:
                                                    # Unknown if data survived - try both fixes
                                                    fix_multipack_remote_cmd += (
                                                        f"    \n"
                                                        f"    # Unknown if data survived - try both fixes\n"
                                                        f"    # Disable multipack sampler (safest fix for batch issues)\n"
                                                        f"    if config.get('sample_packing', True):\n"
                                                        f"        config['sample_packing'] = False\n"
                                                        f"        fixed = True\n"
                                                        f"        print('Disabled sample_packing (multipack sampler)')\n"
                                                        f"    \n"
                                                        f"    # Increase sequence length if it's small (to retain more samples)\n"
                                                        f"    current_seq_len = config.get('sequence_len', 2048)\n"
                                                        f"    if current_seq_len < 4096:\n"
                                                        f"        config['sequence_len'] = 4096\n"
                                                        f"        fixed = True\n"
                                                        f"        print(f'Increased sequence_len from {{current_seq_len}} to 4096 to retain more samples')\n"
                                                    )
                                                
                                                fix_multipack_remote_cmd += (
                                                    f"    \n"
                                                    f"    if fixed:\n"
                                                    f"        with open('axolotl_config.yaml', 'w') as f:\n"
                                                    f"            yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                                    f"        print('Config fixed successfully for multipack sampler error')\n"
                                                    f"    else:\n"
                                                    f"        print('Config already adjusted')\n"
                                                    f"except Exception as e:\n"
                                                    f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                                    f"    sys.exit(1)\n"
                                                    f"PYTHON_EOF"
                                                )
                                                fix_multipack_cmd = [
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                    f"root@{ssh_host}",
                                                    fix_multipack_remote_cmd
                                                ]
                                                fix_multipack_result = subprocess.run(fix_multipack_cmd, capture_output=True, text=True, timeout=30)
                                                if fix_multipack_result.returncode == 0 and "fixed successfully" in fix_multipack_result.stdout:
                                                    terminal_output.append(f"[SUCCESS] Config file fixed for multipack sampler error!")
                                                    terminal_output.append(f"[INFO] Changes made:")
                                                    terminal_output.append(f"[INFO]   • Set train_on_inputs=True (ensures all samples survive filtering)")
                                                    if data_survived is False:
                                                        terminal_output.append(f"[INFO]   • Increased sequence_len to 4096 (to retain more samples)")
                                                        terminal_output.append(f"[INFO]   • Disabled sample_packing (no data survived)")
                                                    elif data_survived is True:
                                                        terminal_output.append(f"[INFO]   • Disabled sample_packing (multipack sampler issue)")
                                                        terminal_output.append(f"[INFO]   • Adjusted batch size if needed")
                                                    else:
                                                        terminal_output.append(f"[INFO]   • Disabled sample_packing")
                                                        terminal_output.append(f"[INFO]   • Adjusted sequence_len and batch settings")
                                                    terminal_output.append(f"[INFO] You can now click 'Redo Phase' to restart training with the corrected config.")
                                                    st.session_state[terminal_output_key] = terminal_output
                                                elif fix_multipack_result.returncode == 0:
                                                    terminal_output.append(f"[INFO] Config check completed: {fix_multipack_result.stdout.strip()}")
                                                    st.session_state[terminal_output_key] = terminal_output
                                                else:
                                                    error_msg = fix_multipack_result.stderr or fix_multipack_result.stdout
                                                    terminal_output.append(f"[WARNING] Could not fix config: {error_msg[:200]}")
                                                    st.session_state[terminal_output_key] = terminal_output
                                            except Exception as e:
                                                terminal_output.append(f"[WARNING] Error attempting to fix config: {str(e)}")
                                                st.session_state[terminal_output_key] = terminal_output
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
                                    
                                    # Always save and rerun at the end to ensure terminal updates
                                    # CRITICAL: Save terminal output IMMEDIATELY before any final operations
                                    st.session_state[terminal_output_key] = list(terminal_output)  # Save current state first
                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                    
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Status check complete.")
                                    
                                    # If status is completed and there are more jobs, add a note about auto-transition
                                    if status_val == "completed":
                                        job_queue = active_job.get("job_queue")
                                        current_job_index = active_job.get("current_job_index")
                                        if job_queue and current_job_index is not None and current_job_index + 1 < len(job_queue):
                                            terminal_output.append(f"[INFO] Queue transition will happen automatically on next status check.")
                                            terminal_output.append(f"[INFO] The next job will start automatically when you click 'Check Training Status' again.")
                                    
                                    # CRITICAL: Save terminal output before rerun - ensure it's a fresh list
                                    # Create a fresh copy to avoid any reference issues - this ensures Streamlit detects the change
                                    final_output = [line for line in terminal_output]  # List comprehension creates new list
                                    st.session_state[terminal_output_key] = final_output  # Save to session state
                                    # Force Streamlit to recognize the change by updating a timestamp AND version counter
                                    st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                    # Increment version counter to force re-render
                                    current_version = st.session_state.get(f"{terminal_output_key}_version", 0)
                                    st.session_state[f"{terminal_output_key}_version"] = current_version + 1
                                    
                                    # Debug: Verify save (only if debug is enabled)
                                    if st.session_state.get("debug_terminal", False):
                                        saved_length = len(st.session_state.get(terminal_output_key, []))
                                        terminal_output.append(f"[DEBUG] Terminal saved: {saved_length} lines, key: {terminal_output_key}")
                                        st.session_state[terminal_output_key] = list(terminal_output)
                                        st.session_state[f"{terminal_output_key}_updated"] = datetime.now().isoformat()
                                    
                                    # CRITICAL: Use st.rerun() to force Streamlit to re-execute the entire script
                                    # This will cause the terminal display (line ~2695) to re-read from session_state
                                    # and show all the updates we just saved
                                    st.rerun()
                            except Exception as e:
                                import traceback
                                error_msg = str(e)
                                error_traceback = traceback.format_exc()
                                # Ensure terminal_output is initialized
                                if terminal_output_key not in st.session_state:
                                    st.session_state[terminal_output_key] = []
                                terminal_output = list(st.session_state[terminal_output_key])
                                terminal_output.append(f"[ERROR] {error_msg}")
                                terminal_output.append(f"[ERROR] Traceback: {error_traceback[:500]}")
                                st.session_state[terminal_output_key] = terminal_output
                                st.error(f"Error: {error_msg}")
                                # Ensure we rerun to show error
                                st.rerun()
                    
                    with col3:
                        if st.button("🗑️ Clear Terminal", key="clear_terminal_phase3", help="Clear terminal output"):
                            st.session_state[terminal_output_key] = []
                            st.rerun()
                    
                    with col4:
                        # Force advance button - only show if there's a queue or if we're on the last job
                        job_queue = active_job.get("job_queue")
                        current_job_index = active_job.get("current_job_index")
                        has_more_jobs = job_queue and current_job_index is not None and current_job_index + 1 < len(job_queue)
                        is_last_job = job_queue and current_job_index is not None and current_job_index + 1 >= len(job_queue)
                        
                        if has_more_jobs or is_last_job:
                            # Use active_job.get directly to avoid UnboundLocalError
                            force_advance_confirm_key = f"force_advance_confirm_{active_job.get('instance_id', 'unknown')}"
                            
                            if st.session_state.get(force_advance_confirm_key, False):
                                st.warning("⚠️ **Force Advance Confirmation**")
                                st.markdown("""
                                **Only use this if:**
                                - The training log shows the job has completed
                                - The program isn't advancing automatically
                                - You've verified the training process has stopped
                                
                                **This will:**
                                - Force transition to the next job (if available), OR
                                - Force advance to Finalize phase (if this is the last job)
                                """)
                                
                                col_confirm1, col_confirm2 = st.columns(2)
                                with col_confirm1:
                                    if st.button("✅ Confirm Force Advance", key="confirm_force_advance", type="primary"):
                                        try:
                                            terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ FORCE ADVANCE triggered by user")
                                            
                                            if has_more_jobs:
                                                terminal_output.append(f"[FORCE] Forcing queue transition from job {current_job_index + 1} to job {current_job_index + 2}...")
                                                terminal_output.append(f"[FORCE] WARNING: This bypasses normal completion checks. Ensure job {current_job_index + 1} is actually complete!")
                                                st.session_state[terminal_output_key] = terminal_output
                                                
                                                # Manually set status to completed to trigger queue transition
                                                # We'll reuse the queue transition logic from the status check
                                                active_job["training_status"] = {"status": "completed", "forced": True}
                                                training_manager._save_job(active_job)
                                                
                                                # Trigger a status check which will detect "completed" and run queue transition
                                                st.session_state["trigger_status_check"] = True
                                                st.session_state[force_advance_confirm_key] = False
                                                st.rerun()
                                            elif is_last_job:
                                                terminal_output.append(f"[FORCE] Forcing advance to Finalize phase...")
                                                terminal_output.append(f"[FORCE] WARNING: This bypasses normal completion checks. Ensure training is actually complete!")
                                                st.session_state[terminal_output_key] = terminal_output
                                                
                                                # Advance to Phase 4
                                                st.session_state[phase_key] = 4
                                                st.session_state[terminal_output_key] = []
                                                st.session_state[force_advance_confirm_key] = False
                                                st.rerun()
                                        except Exception as e:
                                            terminal_output.append(f"[ERROR] Force advance failed: {str(e)}")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.error(f"Error: {str(e)}")
                                            st.session_state[force_advance_confirm_key] = False
                                
                                with col_confirm2:
                                    if st.button("❌ Cancel", key="cancel_force_advance"):
                                        st.session_state[force_advance_confirm_key] = False
                                        st.rerun()
                            else:
                                if has_more_jobs:
                                    if st.button("⚡ Force Next Job", key="force_advance_button", type="secondary", help="Force advance to next job (use only if log shows completion but program isn't advancing)"):
                                        st.session_state[force_advance_confirm_key] = True
                                        st.rerun()
                                elif is_last_job:
                                    if st.button("⚡ Force Finalize", key="force_finalize_button", type="secondary", help="Force advance to Finalize phase (use only if log shows completion but program isn't advancing)"):
                                        st.session_state[force_advance_confirm_key] = True
                                        st.rerun()
                    
                    with col3:
                        confirm_key_phase3 = f"confirm_redo_phase_3_{active_job.get('instance_id')}"
                        if confirm_key_phase3 not in st.session_state:
                            st.session_state[confirm_key_phase3] = False
                        
                        if st.session_state[confirm_key_phase3]:
                            if st.button("✅ Confirm Redo Phase 3", key="confirm_redo_phase_3_btn", type="primary"):
                                try:
                                    instance_id = active_job.get("instance_id")
                                    
                                    # Clear terminal output
                                    st.session_state[terminal_output_key] = []
                                    terminal_output = []
                                    terminal_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] Redoing Phase 3 - killing processes and cleaning up...")
                                    
                                    # Get SSH info - check for port override first
                                    ssh_host = active_job.get("ssh_host")
                                    ssh_port_override = active_job.get("ssh_port_override")
                                    
                                    # Use port override if provided, otherwise use saved port
                                    if ssh_port_override:
                                        ssh_port = ssh_port_override
                                        terminal_output.append(f"[INFO] Using SSH port override: {ssh_port}")
                                    else:
                                        ssh_port = active_job.get("ssh_port", 22)
                                    
                                    # If not in job, get from API
                                    if not ssh_host:
                                        job_status = training_manager.get_job_status(instance_id)
                                        ssh_host = job_status.get("ssh_host")
                                        api_ssh_port = job_status.get("ssh_port", 22)
                                        
                                        # Use port override if provided, otherwise use API port
                                        if ssh_port_override:
                                            ssh_port = ssh_port_override
                                            terminal_output.append(f"[INFO] Using SSH port override: {ssh_port} (instead of API port: {api_ssh_port})")
                                        else:
                                            ssh_port = api_ssh_port
                                        
                                        # Save to job for future use
                                        if ssh_host:
                                            active_job["ssh_host"] = ssh_host
                                            active_job["ssh_port"] = ssh_port
                                            training_manager._save_job(active_job)
                                    
                                    if ssh_host:
                                        terminal_output.append(f"[SSH] Connecting to {ssh_host}:{ssh_port}")
                                        import subprocess
                                        
                                        # Step 1: Kill all training-related processes
                                        terminal_output.append(f"[SSH] Killing all training processes...")
                                        processes_killed = kill_all_training_processes(ssh_host, ssh_port, terminal_output)
                                        
                                        # CRITICAL: Do not proceed if processes are still running
                                        if not processes_killed:
                                            terminal_output.append(f"[ERROR] ⚠️ CRITICAL: Training processes are still running!")
                                            terminal_output.append(f"[ERROR] Cannot start new training while old processes are active.")
                                            terminal_output.append(f"[ERROR] Please wait a moment and try 'Redo Phase' again, or manually kill processes via SSH.")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.error("❌ Cannot proceed: Training processes are still running. Please try 'Redo Phase' again.")
                                            st.stop()
                                        else:
                                            terminal_output.append(f"[SSH] ✓ All processes confirmed killed - safe to proceed")
                                    
                                    # Step 2: Forcefully delete output directories and recreate them
                                    terminal_output.append(f"[SSH] Forcefully deleting output directories...")
                                    cleanup_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        "echo 'Deleting output directory...' && rm -rf /workspace/output 2>&1 && echo 'Output directory deleted' && mkdir -p /workspace/output/training && echo 'Directories recreated' && ls -la /workspace/output/ 2>&1 && echo 'Output directories forcefully deleted and recreated'"
                                    ]
                                    cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30, input="yes\n")
                                    if cleanup_result.returncode == 0:
                                        terminal_output.append(f"[SSH] ✓ Output directories forcefully deleted and recreated")
                                        stdout_filtered = filter_malloc_warnings(cleanup_result.stdout)
                                        if stdout_filtered.strip():
                                            for line in stdout_filtered.strip().split("\n"):
                                                if line.strip() and "forcefully" not in line.lower() and "Deleting" not in line:
                                                    terminal_output.append(f"[SSH]   {line}")
                                    else:
                                        stderr_filtered = filter_malloc_warnings(cleanup_result.stderr)
                                        stdout_filtered = filter_malloc_warnings(cleanup_result.stdout)
                                        terminal_output.append(f"[ERROR] Cleanup command failed (return code: {cleanup_result.returncode})")
                                        if stderr_filtered.strip():
                                            terminal_output.append(f"[ERROR] stderr: {stderr_filtered[:300]}")
                                        if stdout_filtered.strip():
                                            terminal_output.append(f"[ERROR] stdout: {stdout_filtered[:300]}")
                                        # Try to verify if deletion actually happened despite error
                                        verify_cmd = [
                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                            f"root@{ssh_host}",
                                            "if [ -d /workspace/output ]; then echo 'DIRECTORY_EXISTS' && ls -la /workspace/output/; else echo 'DIRECTORY_DELETED'; fi"
                                        ]
                                        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=10)
                                        if verify_result.returncode == 0:
                                            verify_output = verify_result.stdout.strip()
                                            if "DIRECTORY_DELETED" in verify_output:
                                                terminal_output.append(f"[SSH] ✓ Verification: Directory was deleted (recreation may have failed)")
                                            else:
                                                terminal_output.append(f"[SSH] ⚠️ Verification: Directory still exists - deletion may have failed")
                                                terminal_output.append(f"[SSH]   {verify_output}")
                                    
                                    # Step 2.4: Analyze config and suggest batch size optimization
                                    terminal_output.append(f"[SSH] Analyzing training config for optimization...")
                                    try:
                                        # Read current config
                                        read_config_cmd = [
                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                            f"root@{ssh_host}",
                                            "cd /workspace/data && python3 << 'PYTHON_EOF'\n"
                                            "import yaml\n"
                                            "import sys\n"
                                            "try:\n"
                                            "    with open('axolotl_config.yaml', 'r') as f:\n"
                                            "        config = yaml.safe_load(f) or {}\n"
                                            "    \n"
                                            "    # Extract key parameters\n"
                                            "    micro_batch_size = config.get('micro_batch_size', 1)\n"
                                            "    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)\n"
                                            "    num_epochs = config.get('num_epochs', 10)\n"
                                            "    effective_batch_size = micro_batch_size * gradient_accumulation_steps\n"
                                            "    \n"
                                            "    print(f'CURRENT_CONFIG:')\n"
                                            "    print(f'micro_batch_size={micro_batch_size}')\n"
                                            "    print(f'gradient_accumulation_steps={gradient_accumulation_steps}')\n"
                                            "    print(f'effective_batch_size={effective_batch_size}')\n"
                                            "    print(f'num_epochs={num_epochs}')\n"
                                            "except Exception as e:\n"
                                            "    print(f'ERROR: {e}', file=sys.stderr)\n"
                                            "    sys.exit(1)\n"
                                            "PYTHON_EOF"
                                        ]
                                        config_read_result = subprocess.run(read_config_cmd, capture_output=True, text=True, timeout=15)
                                        
                                        if config_read_result.returncode == 0:
                                            # Parse config output
                                            current_micro_batch = 1
                                            current_grad_accum = 1
                                            current_epochs = 10
                                            
                                            for line in config_read_result.stdout.split('\n'):
                                                if 'micro_batch_size=' in line:
                                                    current_micro_batch = int(line.split('=')[1])
                                                elif 'gradient_accumulation_steps=' in line:
                                                    current_grad_accum = int(line.split('=')[1])
                                                elif 'num_epochs=' in line:
                                                    current_epochs = int(line.split('=')[1])
                                            
                                            terminal_output.append(f"[INFO] Current config:")
                                            terminal_output.append(f"  • micro_batch_size: {current_micro_batch}")
                                            terminal_output.append(f"  • gradient_accumulation_steps: {current_grad_accum}")
                                            terminal_output.append(f"  • effective_batch_size: {current_micro_batch * current_grad_accum}")
                                            terminal_output.append(f"  • num_epochs: {current_epochs}")
                                            
                                            # Check GPU memory to suggest optimal batch size
                                            check_memory_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1"
                                            ]
                                            memory_result = subprocess.run(check_memory_cmd, capture_output=True, text=True, timeout=10)
                                            
                                            suggested_batch_size = current_micro_batch
                                            memory_info = "Unable to check"
                                            
                                            if memory_result.returncode == 0 and memory_result.stdout.strip():
                                                try:
                                                    parts = memory_result.stdout.strip().split(',')
                                                    if len(parts) >= 3:
                                                        total_mb = int(parts[0].strip())
                                                        used_mb = int(parts[1].strip())
                                                        free_mb = int(parts[2].strip())
                                                        total_gb = total_mb / 1024
                                                        used_gb = used_mb / 1024
                                                        free_gb = free_mb / 1024
                                                        usage_percent = (used_mb / total_mb) * 100
                                                        
                                                        memory_info = f"Total: {total_gb:.1f} GiB, Used: {used_gb:.1f} GiB ({usage_percent:.1f}%), Free: {free_gb:.1f} GiB"
                                                        
                                                        # Suggest batch size based on free memory
                                                        # Rough estimate: each batch_size increase uses ~2-4 GiB for 4B model
                                                        if free_gb > 20 and current_micro_batch < 4:
                                                            # Can safely increase to 4
                                                            suggested_batch_size = min(4, current_micro_batch * 2)
                                                        elif free_gb > 10 and current_micro_batch < 3:
                                                            # Can increase to 3
                                                            suggested_batch_size = min(3, current_micro_batch + 1)
                                                        elif free_gb > 5 and current_micro_batch < 2:
                                                            # Can increase to 2
                                                            suggested_batch_size = 2
                                                        
                                                        terminal_output.append(f"[INFO] GPU Memory: {memory_info}")
                                                        
                                                        if suggested_batch_size > current_micro_batch:
                                                            speedup_estimate = suggested_batch_size / current_micro_batch
                                                            terminal_output.append(f"[OPTIMIZATION] 💡 Suggested micro_batch_size: {suggested_batch_size} (current: {current_micro_batch})")
                                                            terminal_output.append(f"[OPTIMIZATION] Estimated speedup: ~{speedup_estimate:.1f}x faster training")
                                                            terminal_output.append(f"[OPTIMIZATION] This will be applied when restarting training")
                                                            
                                                            # Store suggestion for later use
                                                            st.session_state[f"optimize_batch_size_{instance_id}"] = suggested_batch_size
                                                        else:
                                                            terminal_output.append(f"[INFO] Current batch size is optimal for available memory")
                                                except:
                                                    pass
                                            
                                            terminal_output.append(f"[INFO] Memory check: {memory_info}")
                                        else:
                                            terminal_output.append(f"[WARNING] Could not read config: {config_read_result.stderr[:200]}")
                                    except Exception as e:
                                        terminal_output.append(f"[WARNING] Config analysis failed: {str(e)}")
                                    
                                    # Step 2.5: Fix config issues (tokenizer type, sample_packing, batch size optimization, etc.)
                                    terminal_output.append(f"[SSH] Checking and fixing config issues...")
                                    
                                    # Check if we have a suggested batch size optimization
                                    suggested_batch_size = st.session_state.get(f"optimize_batch_size_{instance_id}")
                                    batch_size_update = ""
                                    if suggested_batch_size:
                                        batch_size_update = (
                                            f"    # Optimize batch size for better performance\n"
                                            f"    old_batch_size = config.get('micro_batch_size', 1)\n"
                                            f"    if old_batch_size < {suggested_batch_size}:\n"
                                            f"        config['micro_batch_size'] = {suggested_batch_size}\n"
                                            f"        fixed = True\n"
                                            f"        print(f'Optimized micro_batch_size: {{old_batch_size}} -> {suggested_batch_size}')\n"
                                        )
                                    
                                    fix_tokenizer_remote_cmd_redo = (
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
                                        f"    # ALWAYS disable sample_packing to prevent multipack sampler errors\n"
                                        f"    if config.get('sample_packing', False):\n"
                                        f"        config['sample_packing'] = False\n"
                                        f"        fixed = True\n"
                                        f"        print('Disabled sample_packing to prevent multipack sampler errors')\n"
                                        f"    \n"
                                        f"    # ALWAYS ensure train_on_inputs is True to maximize sample retention\n"
                                        f"    if not config.get('train_on_inputs', True):\n"
                                        f"        config['train_on_inputs'] = True\n"
                                        f"        fixed = True\n"
                                        f"        print('Set train_on_inputs=True to maximize sample retention')\n"
                                        f"    \n"
                                        f"    # Fix adapter/lora_model_dir issue: Ensure adapter: 'lora' is set for LoRA mode\n"
                                        f"    # Remove lora_model_dir if it points to output_dir (causes path issues)\n"
                                        f"    adapter_val = config.get('adapter')\n"
                                        f"    output_dir = config.get('output_dir', '')\n"
                                        f"    lora_model_dir = config.get('lora_model_dir', '')\n"
                                        f"    \n"
                                        f"    # CRITICAL: Ensure adapter: 'lora' is set if lora_* parameters exist (required for Axolotl to enable LoRA)\n"
                                        f"    # BUT preserve adapter paths for incremental training (V2+)\n"
                                        f"    # Axolotl needs explicit adapter: 'lora' to enable LoRA mode, not just lora_* parameters\n"
                                        f"    has_lora_params = config.get('lora_r') is not None and config.get('lora_alpha') is not None\n"
                                        f"    if has_lora_params:\n"
                                        f"        # Set adapter to 'lora' if it's missing or null (but keep it if already set to 'lora' or a path)\n"
                                        f"        if not adapter_val or adapter_val is None:\n"
                                        f"            config['adapter'] = 'lora'\n"
                                        f"            fixed = True\n"
                                        f"            print('Set adapter: lora (required for LoRA mode)')\n"
                                        f"        elif adapter_val == 'lora':\n"
                                        f"            # Keep it - don't remove adapter: 'lora'\n"
                                        f"            print('Adapter: lora already set (LoRA mode enabled)')\n"
                                        f"        elif isinstance(adapter_val, str) and adapter_val.startswith('/'):\n"
                                        f"            # Keep adapter path for incremental training (V2+) - don't override it\n"
                                        f"            print(f'Keeping adapter path for incremental training: {{adapter_val}}')\n"
                                        f"    \n"
                                        f"    # CRITICAL: Remove lora_model_dir if it's set to output_dir (causes Axolotl to try loading from there)\n"
                                        f"    # lora_model_dir should only be set to a valid adapter path (for incremental training)\n"
                                        f"    # If lora_model_dir is set to output_dir, always remove it (unless there's a valid adapter path)\n"
                                        f"    if lora_model_dir == output_dir:\n"
                                        f"        # Only keep it if adapter is set to a valid path (incremental training)\n"
                                        f"        if not adapter_val or adapter_val == 'lora' or adapter_val is None or not isinstance(adapter_val, str) or not adapter_val.startswith('/'):\n"
                                        f"            if 'lora_model_dir' in config:\n"
                                        f"                del config['lora_model_dir']\n"
                                        f"                fixed = True\n"
                                        f"                print('Removed lora_model_dir (was pointing to output_dir - causes adapter loading errors)')\n"
                                        f"    \n"
                                        f"    # Verify adapter path exists for incremental training\n"
                                        f"    if adapter_val and isinstance(adapter_val, str) and adapter_val.startswith('/'):\n"
                                        f"        import os\n"
                                        f"        adapter_config_path = os.path.join(adapter_val, 'adapter_config.json')\n"
                                        f"        if not os.path.exists(adapter_config_path):\n"
                                        f"            print(f'WARNING: Adapter config not found at {{adapter_config_path}}')\n"
                                        f"            print(f'Adapter path may be invalid - training may fail')\n"
                                        f"        else:\n"
                                        f"            # Verify base model matches\n"
                                        f"            try:\n"
                                        f"                import json\n"
                                        f"                with open(adapter_config_path, 'r') as f:\n"
                                        f"                    adapter_config = json.load(f)\n"
                                        f"                    adapter_base = adapter_config.get('base_model_name', '')\n"
                                        f"                    config_base = config.get('base_model', '')\n"
                                        f"                    if adapter_base and config_base and adapter_base != config_base:\n"
                                        f"                        print(f'WARNING: Base model mismatch - adapter: {{adapter_base}}, config: {{config_base}}')\n"
                                        f"                    elif adapter_base == config_base:\n"
                                        f"                        print(f'✓ Base model matches: {{config_base}}')\n"
                                        f"            except Exception as e:\n"
                                        f"                print(f'Could not verify base model match: {{e}}')\n"
                                        f"    \n"
                                        f"    # Ensure lora_* parameters are set for LoRA mode\n"
                                        f"    # These are always required when using LoRA (with adapter: 'lora' or without)\n"
                                        f"    if not adapter_val or adapter_val == 'lora' or adapter_val is None:\n"
                                        f"        if 'lora_r' not in config:\n"
                                        f"            config['lora_r'] = 8\n"
                                        f"            fixed = True\n"
                                        f"            print('Set lora_r=8 (default)')\n"
                                        f"        if 'lora_alpha' not in config:\n"
                                        f"            config['lora_alpha'] = 16\n"
                                        f"            fixed = True\n"
                                        f"            print('Set lora_alpha=16 (default)')\n"
                                        f"        if 'lora_dropout' not in config:\n"
                                        f"            config['lora_dropout'] = 0.05\n"
                                        f"            fixed = True\n"
                                        f"            print('Set lora_dropout=0.05 (default)')\n"
                                        f"        if 'lora_target_modules' not in config:\n"
                                        f"            config['lora_target_modules'] = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']\n"
                                        f"            fixed = True\n"
                                        f"            print('Set lora_target_modules (default)')\n"
                                        f"        if 'lora_out_dir' not in config and output_dir:\n"
                                        f"            config['lora_out_dir'] = f'{{output_dir}}/adapter'\n"
                                        f"            fixed = True\n"
                                        f"            print(f'Set lora_out_dir={{output_dir}}/adapter')\n"
                                        f"    \n"
                                        f"{batch_size_update}"
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
                                        f"        fixed = True\n"
                                        f"        print('Fixed: Qwen -> Qwen2Tokenizer')\n"
                                        f"    \n"
                                        f"    # Also fix model_type if needed\n"
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
                                        f"            print('Fixed model_type: -> GemmaForCausalLM')\n"
                                        f"    \n"
                                        f"    if fixed:\n"
                                        f"        with open('axolotl_config.yaml', 'w') as f:\n"
                                        f"            yaml.dump(config, f, default_flow_style=False, sort_keys=False)\n"
                                        f"        print('Config file updated successfully')\n"
                                        f"    else:\n"
                                        f"        print('Config already correct')\n"
                                        f"except Exception as e:\n"
                                        f"    print(f'Error: {{e}}', file=sys.stderr)\n"
                                        f"    sys.exit(1)\n"
                                        f"PYTHON_EOF"
                                    )
                                    fix_tokenizer_cmd = [
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        f"root@{ssh_host}",
                                        fix_tokenizer_remote_cmd_redo
                                    ]
                                    fix_result = subprocess.run(fix_tokenizer_cmd, capture_output=True, text=True, timeout=30)
                                    if fix_result.returncode == 0:
                                        stdout_filtered = filter_malloc_warnings(fix_result.stdout)
                                        if "Fixed" in fix_result.stdout or "updated successfully" in fix_result.stdout or "Optimized" in fix_result.stdout:
                                            terminal_output.append(f"[SUCCESS] Config updated: {stdout_filtered}")
                                            if suggested_batch_size:
                                                terminal_output.append(f"[SUCCESS] ✓ Batch size optimized to {suggested_batch_size} for faster training")
                                        else:
                                            terminal_output.append(f"[INFO] Config already correct")
                                            if suggested_batch_size:
                                                terminal_output.append(f"[INFO] Batch size optimization will be applied")
                                    else:
                                        error_msg = fix_result.stderr or fix_result.stdout
                                        terminal_output.append(f"[WARNING] Could not update config: {error_msg[:200]}")
                                    
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
                                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                        "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                                        f"root@{ssh_host}",
                                        ssh_command
                                    ]
                                    terminal_output.append(f"[DEBUG] Executing training command via SSH on port: {ssh_port}")
                                    terminal_output.append(f"[DEBUG] SSH command: ssh -p {ssh_port} root@{ssh_host}")
                                    try:
                                        restart_result = subprocess.run(restart_cmd, capture_output=True, text=True, timeout=5)
                                        terminal_output.append(f"[DEBUG] Training restart command return code: {restart_result.returncode}")
                                        
                                        if restart_result.returncode == 0:
                                            terminal_output.append(f"[SSH] Training command executed")
                                            if restart_result.stdout:
                                                stdout_filtered = filter_malloc_warnings(restart_result.stdout)
                                                # Show all output, not just filtered
                                                for line in stdout_filtered.strip().split("\n"):
                                                    if line.strip():
                                                        terminal_output.append(f"[SSH] {line[:200]}")
                                        else:
                                            # Even if return code is non-zero, the process might have started
                                            terminal_output.append(f"[SSH] Training command sent (return code: {restart_result.returncode}, checking process status...)")
                                            if restart_result.stdout:
                                                stdout_filtered = filter_malloc_warnings(restart_result.stdout)
                                                for line in stdout_filtered.strip().split("\n"):
                                                    if line.strip():
                                                        terminal_output.append(f"[STDOUT] {line[:200]}")
                                            if restart_result.stderr:
                                                stderr_filtered = filter_malloc_warnings(restart_result.stderr)
                                                # Show all stderr output
                                                for line in stderr_filtered.strip().split("\n"):
                                                    if line.strip():
                                                        terminal_output.append(f"[STDERR] {line[:200]}")
                                    except subprocess.TimeoutExpired:
                                        # Timeout is expected - the process is running in background
                                        terminal_output.append(f"[SSH] Training command sent (timeout expected for background process)")
                                    except Exception as e:
                                        terminal_output.append(f"[ERROR] Exception while executing training restart command: {str(e)}")
                                        import traceback
                                        terminal_output.append(f"[ERROR] Traceback: {traceback.format_exc()[:300]}")
                                    
                                    # Wait a moment and then check if the process actually started and if logs are being written
                                    import time
                                    time.sleep(3)
                                    
                                    # Quick check to see if log file is being created
                                    try:
                                        quick_log_check_cmd = [
                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                            f"root@{ssh_host}",
                                            "test -f /workspace/output/training/training.log && (wc -l /workspace/output/training/training.log && tail -20 /workspace/output/training/training.log) || echo 'log_not_created'"
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
                                                        terminal_output.append(f"[SSH] Recent log content (showing more lines):")
                                                        for line in log_content.strip().split('\n'):
                                                            if line.strip():
                                                                terminal_output.append(f"[SSH]   {line[:200]}")
                                        else:
                                            terminal_output.append(f"[WARNING] Log file not created yet - training may not have started")
                                            
                                            # Check if there are any Python processes that might have failed
                                            check_python_errors_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "ps aux | grep python | grep -E '(axolotl|accelerate|train)' | grep -v grep | head -5 || echo 'no_python_process'"
                                            ]
                                            python_process_check = subprocess.run(check_python_errors_cmd, capture_output=True, text=True, timeout=10)
                                            if "no_python_process" not in python_process_check.stdout and python_process_check.stdout.strip():
                                                terminal_output.append(f"[DIAGNOSTICS] Found Python processes:")
                                                for line in python_process_check.stdout.strip().split('\n'):
                                                    if line.strip():
                                                        terminal_output.append(f"[DIAGNOSTICS]   {line[:200]}")
                                            else:
                                                terminal_output.append(f"[DIAGNOSTICS] No Python training processes found")
                                    except subprocess.TimeoutExpired:
                                        terminal_output.append(f"[WARNING] Log check timed out")
                                    except Exception as e:
                                        terminal_output.append(f"[WARNING] Could not check log file: {str(e)}")
                                    
                                    # Always check if process started, regardless of command result
                                    terminal_output.append(f"[SSH] Verifying training process started...")
                                    
                                    # Wait a moment and verify training process is running
                                    try:
                                        verify_cmd = [
                                            "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                                            # Process not found - check logs for more details
                                            terminal_output.append(f"[WARNING] Training command executed but process not found yet")
                                            
                                            # Check training log for errors
                                            check_log_cmd = [
                                                "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                f"root@{ssh_host}",
                                                "tail -50 /workspace/output/training/training.log 2>/dev/null || echo 'no_log'"
                                            ]
                                            log_check = subprocess.run(check_log_cmd, capture_output=True, text=True, timeout=15)
                                            if "no_log" not in log_check.stdout:
                                                log_output = filter_malloc_warnings(log_check.stdout)
                                                terminal_output.append(f"[WARNING] Last log output (showing more lines for debugging):")
                                                for line in log_output.strip().split("\n"):
                                                    if line.strip():
                                                        terminal_output.append(f"[LOG] {line[:200]}")
                                            else:
                                                terminal_output.append(f"[WARNING] Training log file not found at /workspace/output/training/training.log")
                                            
                                            terminal_output.append(f"[INFO] Training may still be starting. Use 'Check Training Status' to verify.")
                                            terminal_output.append(f"[INFO] Check /workspace/output/training/training.log for details")
                                            # Reset training status so "Start Training" button appears if training didn't start
                                            active_job["training_status"] = {"status": "not_started", "message": "Training command sent but process not verified"}
                                            training_manager._save_job(active_job)
                                    except subprocess.TimeoutExpired:
                                        terminal_output.append(f"[WARNING] Process verification timed out - training may still be starting")
                                        terminal_output.append(f"[INFO] Check training status again in a few moments")
                                    except Exception as e:
                                        terminal_output.append(f"[WARNING] Could not verify training process: {str(e)}")
                                        terminal_output.append(f"[INFO] Training command was sent - check status to verify training started")
                                
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.session_state[confirm_key_phase3] = False
                                    st.rerun()
                                except Exception as e:
                                    error_msg = str(e)
                                    terminal_output.append(f"[ERROR] Failed to redo Phase 3: {error_msg}")
                                    # Reset training status so "Start Training" button appears after error
                                    active_job["training_status"] = {"status": "not_started", "error": error_msg}
                                    training_manager._save_job(active_job)
                                    st.session_state[terminal_output_key] = terminal_output
                                    st.session_state[confirm_key_phase3] = False
                                    st.error(f"Error: {error_msg}")
                            if st.button("❌ Cancel", key="cancel_redo_phase_3"):
                                st.session_state[confirm_key_phase3] = False
                                st.rerun()
                        else:
                            if st.button("🔄 Redo Phase", key="redo_phase_3", type="secondary", help="Kill training processes, clean up, and restart training. This will clear the terminal and restart training from scratch."):
                                st.session_state[confirm_key_phase3] = True
                                st.rerun()
                    
                    with col4:
                        training_status = active_job.get("training_status", {})
                        status_from_job = training_status.get("status")
                        
                        # Also check if completion indicators are in logs (even if status hasn't updated yet)
                        completion_detected_in_logs = False
                        terminal_output_for_check = st.session_state.get(terminal_output_key, [])
                        if terminal_output_for_check:
                            terminal_text = "\n".join(terminal_output_for_check).lower()
                            has_completed_log = "training completed" in terminal_text
                            has_saved_log = "model successfully saved" in terminal_text
                            completion_detected_in_logs = has_completed_log and has_saved_log
                        
                        # Show button if status is completed OR if completion is detected in logs
                        if status_from_job == "completed" or completion_detected_in_logs:
                            if completion_detected_in_logs and status_from_job != "completed":
                                st.warning("⚠️ Training completion detected in logs. Click to advance to Phase 4.")
                            else:
                                st.success("✅ Training completed! Click 'Next Phase' to finalize.")
                            
                            if st.button("➡️ Next Phase", key="next_phase_3", type="primary"):
                                # Ensure status is set to completed
                                if status_from_job != "completed":
                                    training_status["status"] = "completed"
                                    active_job["training_status"] = training_status
                                    # Also update job status to completed
                                    active_job["status"] = "completed"
                                    training_manager._save_job(active_job)
                                
                                # Get the phase_key to ensure we're using the right one
                                instance_id = active_job.get("instance_id")
                                phase_key = f"training_phase_{instance_id}"
                                
                                # Set phase to 4 - use session state directly
                                st.session_state[phase_key] = 4
                                
                                # Clear terminal for next phase
                                st.session_state[terminal_output_key] = []
                                
                                # Force rerun to show Phase 4
                                st.rerun()
                    
                    # CRITICAL: Display terminal AFTER all button handlers have run
                    # This ensures the terminal reads the UPDATED session state values
                    # Read from session state to get latest updates
                    terminal_updated_key = f"{terminal_output_key}_updated"
                    terminal_version_key = f"{terminal_output_key}_version"
                    
                    # Read version counter - this creates a dependency
                    terminal_version = st.session_state.get(terminal_version_key, 0)
                    
                    # Read the actual terminal output
                    raw_output = st.session_state.get(terminal_output_key, [])
                    
                    # Read timestamp
                    terminal_timestamp = st.session_state.get(terminal_updated_key, None)
                    
                    # Create a completely fresh list copy
                    current_terminal_output = list(raw_output) if raw_output else []
                    
                    # Use version in display to force dependency tracking
                    version_suffix = f"v{terminal_version}"
                    
                    # Update the placeholder with terminal content
                    with terminal_placeholder.container():
                        if current_terminal_output:
                            # Keep only last 200 lines
                            display_output = current_terminal_output[-200:] if len(current_terminal_output) > 200 else current_terminal_output
                            output_text = "\n".join(display_output)
                            st.code(output_text, language="text")
                            if len(current_terminal_output) > 200:
                                st.caption(f"Showing last 200 of {len(current_terminal_output)} lines")
                            # Show version in caption - this creates the dependency
                            if len(current_terminal_output) > 0:
                                timestamp_display = terminal_timestamp[:19] if terminal_timestamp else 'never'
                                st.caption(f"📊 Total: {len(current_terminal_output)} lines | {version_suffix} | Updated: {timestamp_display}")
                        else:
                            st.info(f"Terminal is empty. Click 'Check Training Status.' ({version_suffix})")
                
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
                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
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
                        
                        # Also check for stored stats in job results (single job mode)
                        if job_queue and len(job_queue) > 0:
                            # Single job - check for stats in standard location
                            job_results_path = Path(f"models/{model_name}/training/queue/job_results")
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
                                ssh_port_override = active_job.get("ssh_port_override")
                                
                                # Use port override if provided
                                if ssh_port_override:
                                    ssh_port = ssh_port_override
                                
                                # If not in job, get from API
                                if not ssh_host:
                                    terminal_output.append(f"[API] Getting instance status for SSH info...")
                                    job_status = training_manager.get_job_status(instance_id)
                                    ssh_host = job_status.get("ssh_host")
                                    api_ssh_port = job_status.get("ssh_port", 22)
                                    
                                    # Use port override if provided, otherwise use API port
                                    if ssh_port_override:
                                        ssh_port = ssh_port_override
                                        terminal_output.append(f"[INFO] Using SSH port override: {ssh_port} (instead of API port: {api_ssh_port})")
                                    else:
                                        ssh_port = api_ssh_port
                                    
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
                                    # Search more specific paths first to avoid scanning entire /workspace/output
                                    # Limit search to training directory and common checkpoint locations
                                    check_files_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                                        f"root@{ssh_host}",
                                        """bash -c '
                                        # Search in most likely locations first (faster)
                                        find /workspace/output/training -maxdepth 4 -type f \\( -name "*.bin" -o -name "*.safetensors" -o -name "adapter_config.json" \\) 2>/dev/null | head -20
                                        if [ $? -ne 0 ] || [ -z "$(find /workspace/output/training -maxdepth 4 -type f \\( -name "*.bin" -o -name "*.safetensors" -o -name "adapter_config.json" \\) 2>/dev/null | head -1)" ]; then
                                            # Fallback to broader search if nothing found
                                            find /workspace/output -maxdepth 6 -type f \\( -name "*.bin" -o -name "*.safetensors" -o -name "adapter_config.json" \\) 2>/dev/null | head -20
                                        fi
                                        '"""
                                    ]
                                    check_result = subprocess.run(check_files_cmd, capture_output=True, text=True, timeout=120)
                                    if check_result.returncode == 0 and check_result.stdout.strip():
                                        stdout_filtered = filter_malloc_warnings(check_result.stdout)
                                        available_files = [f.strip() for f in stdout_filtered.strip().split('\n') if f.strip()]
                                        terminal_output.append(f"[SSH] Found {len(available_files)} weight file(s) on remote instance")
                                        for f in available_files[:5]:
                                            terminal_output.append(f"[SSH]   - {f}")
                                    else:
                                        terminal_output.append(f"[SSH] No weight files found in standard locations")
                                    
                                    # Step 4: Download LoRA adapter weights ONLY (not full model)
                                    weights_dir = version_dir / "weights"
                                    weights_dir.mkdir(parents=True, exist_ok=True)
                                    terminal_output.append(f"[SCP] Downloading LoRA adapter weights to: {weights_dir}")
                                    terminal_output.append(f"[INFO] Only downloading adapter files (adapter_config.json, adapter_model.safetensors, etc.)")
                                    terminal_output.append(f"[INFO] Full model weights are NOT needed - adapter will be loaded onto base model locally")
                                    
                                    # LoRA adapter files we need (in order of preference)
                                    adapter_files = [
                                        "adapter_config.json",  # Required - adapter configuration
                                        "adapter_model.safetensors",  # Preferred - adapter weights in safetensors format
                                        "adapter_model.bin",  # Fallback - adapter weights in bin format
                                        "training_args.bin",  # Optional - training arguments
                                    ]
                                    
                                    # Step 4a: Find the adapter location (may be in checkpoint directory)
                                    terminal_output.append(f"[SSH] Locating adapter files on remote instance...")
                                    find_adapter_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                                        f"root@{ssh_host}",
                                        """bash -c '
                                        # First, try to find the latest checkpoint directory with adapter (limit depth for speed)
                                        latest_checkpoint=$(find /workspace/output/training -maxdepth 3 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
                                        if [ -n "$latest_checkpoint" ] && [ -f "$latest_checkpoint/adapter/adapter_config.json" ]; then
                                            echo "adapter_path:$latest_checkpoint/adapter"
                                        # Check if adapter is directly in output/training/adapter
                                        elif [ -f /workspace/output/training/adapter/adapter_config.json ]; then
                                            echo "adapter_path:/workspace/output/training/adapter"
                                        # Check other common locations
                                        elif [ -f /workspace/output/adapter/adapter_config.json ]; then
                                            echo "adapter_path:/workspace/output/adapter"
                                        # Use find to locate adapter_config.json with limited depth (faster)
                                        elif adapter_found=$(find /workspace/output/training -maxdepth 5 -name "adapter_config.json" -type f 2>/dev/null | head -1); then
                                            adapter_dir=$(dirname "$adapter_found")
                                            echo "adapter_path:$adapter_dir"
                                        # Last resort: broader search but still limited
                                        elif adapter_found=$(find /workspace/output -maxdepth 6 -name "adapter_config.json" -type f 2>/dev/null | head -1); then
                                            adapter_dir=$(dirname "$adapter_found")
                                            echo "adapter_path:$adapter_dir"
                                        else
                                            echo "adapter_path:not_found"
                                        fi
                                        '"""
                                    ]
                                    find_result = subprocess.run(find_adapter_cmd, capture_output=True, text=True, timeout=120)
                                    adapter_dir_remote = "/workspace/output/training/adapter"  # Default fallback
                                    
                                    if find_result.returncode == 0 and "adapter_path:" in find_result.stdout:
                                        for line in find_result.stdout.split('\n'):
                                            if 'adapter_path:' in line:
                                                found_path = line.split('adapter_path:')[1].strip()
                                                if found_path != "not_found":
                                                    adapter_dir_remote = found_path
                                                    terminal_output.append(f"[SSH] ✓ Found adapter at: {adapter_dir_remote}")
                                                    break
                                                else:
                                                    terminal_output.append(f"[SSH] ⚠ Adapter not found, will try default locations")
                                    else:
                                        terminal_output.append(f"[SSH] ⚠ Could not locate adapter, will try default locations")
                                    
                                    downloaded_count = 0
                                    
                                    # First, try to download the entire adapter directory (most efficient)
                                    terminal_output.append(f"[SCP] Attempting to download adapter directory: {adapter_dir_remote}")
                                    # SCP will create weights_dir/adapter when downloading remote:/path/to/adapter
                                    # This is the desired behavior - files go to weights_dir/adapter/
                                    scp_adapter_cmd = [
                                        "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", "-r",
                                        f"root@{ssh_host}:{adapter_dir_remote}",
                                        str(weights_dir)
                                    ]
                                    scp_result = subprocess.run(scp_adapter_cmd, capture_output=True, text=True, timeout=300)
                                    
                                    if scp_result.returncode == 0:
                                        # Check if adapter directory was downloaded
                                        adapter_dir_local = weights_dir / "adapter"
                                        if adapter_dir_local.exists() and any(adapter_dir_local.iterdir()):
                                            downloaded_files_list = [f.name for f in adapter_dir_local.iterdir() if f.is_file()]
                                            downloaded_count = len(downloaded_files_list)
                                            terminal_output.append(f"[SCP] ✓ Downloaded adapter directory ({downloaded_count} files) to: {adapter_dir_local}")
                                            for f in downloaded_files_list[:10]:
                                                terminal_output.append(f"[FILE]   - {f}")
                                            
                                            # Verify we have the essential files
                                            has_config = (adapter_dir_local / "adapter_config.json").exists()
                                            has_weights = (adapter_dir_local / "adapter_model.safetensors").exists() or (adapter_dir_local / "adapter_model.bin").exists()
                                            
                                            if has_config and has_weights:
                                                terminal_output.append(f"[SUCCESS] ✓ Essential adapter files downloaded (config + weights)")
                                                # Skip individual file download - we already have everything
                                                downloaded_count = len(downloaded_files_list)
                                            elif has_config:
                                                terminal_output.append(f"[WARNING] ⚠ Adapter config found but weights file missing")
                                            else:
                                                terminal_output.append(f"[ERROR] ✗ Essential adapter files missing")
                                        else:
                                            # Directory doesn't exist or is empty
                                            terminal_output.append(f"[SCP] ⚠ Adapter directory not found or empty at: {adapter_dir_remote}")
                                    else:
                                        # SCP command failed - try downloading individual files
                                        terminal_output.append(f"[SCP] Directory download failed, trying individual files...")
                                        error_output = scp_result.stderr or scp_result.stdout
                                        error_output = filter_malloc_warnings(error_output)
                                        if "Welcome to vast.ai" in error_output:
                                            lines = error_output.split('\n')
                                            actual_errors = [line for line in lines if line and 'Welcome to vast.ai' not in line and 'Have fun!' not in line]
                                            error_output = '\n'.join(actual_errors) if actual_errors else error_output
                                        terminal_output.append(f"[SCP] Directory download error: {error_output[:200]}")
                                    
                                    # If directory download didn't work OR essential files are missing, try individual files
                                    adapter_dir_local = weights_dir / "adapter"
                                    # Check if we already have all essential files (skip individual download if we do)
                                    has_all_essential = False
                                    if adapter_dir_local.exists():
                                        has_config = (adapter_dir_local / "adapter_config.json").exists()
                                        has_weights = (adapter_dir_local / "adapter_model.safetensors").exists() or (adapter_dir_local / "adapter_model.bin").exists()
                                        has_all_essential = has_config and has_weights
                                    
                                    if not has_all_essential and (not adapter_dir_local.exists() or not any(adapter_dir_local.iterdir())):
                                        adapter_dir_local.mkdir(parents=True, exist_ok=True)
                                        
                                        # Try downloading each essential file individually
                                        for file_name in adapter_files:
                                            # Try multiple possible locations
                                            remote_paths = [
                                                f"{adapter_dir_remote}/{file_name}",  # Found location
                                                f"/workspace/output/training/adapter/{file_name}",  # Standard location
                                                f"/workspace/output/training/{file_name}",  # Root training dir
                                                # Also try latest checkpoint if we haven't already
                                            ]
                                            
                                            # If adapter_dir_remote doesn't contain "checkpoint", also try latest checkpoint
                                            if "checkpoint" not in adapter_dir_remote:
                                                find_checkpoint_cmd = [
                                                    "ssh"
        ] + get_ssh_base_options(ssh_port) + [
                                                    f"root@{ssh_host}",
                                                    "find /workspace/output/training -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1"
                                                ]
                                                checkpoint_result = subprocess.run(find_checkpoint_cmd, capture_output=True, text=True, timeout=15)
                                                if checkpoint_result.returncode == 0 and checkpoint_result.stdout.strip():
                                                    latest_checkpoint = checkpoint_result.stdout.strip()
                                                    remote_paths.insert(1, f"{latest_checkpoint}/adapter/{file_name}")
                                                    remote_paths.insert(2, f"{latest_checkpoint}/{file_name}")
                                            
                                            downloaded_file = False
                                            for remote_path in remote_paths:
                                                terminal_output.append(f"[SCP] Trying to download: {file_name} from {remote_path}")
                                                scp_file_cmd = [
                                                    "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                                                    f"root@{ssh_host}:{remote_path}",
                                                    str(adapter_dir_local / file_name)
                                                ]
                                                file_result = subprocess.run(scp_file_cmd, capture_output=True, text=True, timeout=300)
                                                
                                                if file_result.returncode == 0 and (adapter_dir_local / file_name).exists():
                                                    file_size = (adapter_dir_local / file_name).stat().st_size
                                                    terminal_output.append(f"[SCP] ✓ Downloaded {file_name} ({file_size / 1024 / 1024:.2f} MB)")
                                                    downloaded_count += 1
                                                    downloaded_file = True
                                                    break
                                            
                                            if not downloaded_file:
                                                if file_name == "adapter_config.json":
                                                    terminal_output.append(f"[ERROR] ✗ Failed to download {file_name} (REQUIRED)")
                                                elif file_name in ["adapter_model.safetensors", "adapter_model.bin"]:
                                                    terminal_output.append(f"[ERROR] ✗ Failed to download {file_name} (REQUIRED)")
                                                else:
                                                    terminal_output.append(f"[WARNING] ⚠ Could not download {file_name} (optional)")
                                    
                                    # Final verification - ensure we have at least the essential files
                                    adapter_dir_local = weights_dir / "adapter"
                                    if adapter_dir_local.exists():
                                        has_config = (adapter_dir_local / "adapter_config.json").exists()
                                        has_weights = (adapter_dir_local / "adapter_model.safetensors").exists() or (adapter_dir_local / "adapter_model.bin").exists()
                                        
                                        if not has_config:
                                            terminal_output.append(f"[ERROR] ✗ adapter_config.json is missing - adapter cannot be loaded")
                                        if not has_weights:
                                            terminal_output.append(f"[ERROR] ✗ Adapter weights file is missing - adapter cannot be loaded")
                                        
                                        if has_config and has_weights:
                                            terminal_output.append(f"[SUCCESS] ✓ All essential LoRA adapter files downloaded")
                                        else:
                                            terminal_output.append(f"[ERROR] ✗ Missing essential adapter files - download incomplete")
                                    
                                    # Verify downloaded files - only check adapter directory
                                    adapter_dir_local = weights_dir / "adapter"
                                    if adapter_dir_local.exists():
                                        downloaded_files = [f for f in adapter_dir_local.iterdir() if f.is_file()]
                                        
                                        # Filter to only show adapter files (exclude any accidentally downloaded full model files)
                                        adapter_file_names = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", 
                                                             "training_args.bin", "README.md", "tokenizer_config.json"]
                                        adapter_files = [f for f in downloaded_files if f.name in adapter_file_names or f.name.startswith("adapter_")]
                                        
                                        if adapter_files:
                                            terminal_output.append(f"[LOCAL] Downloaded {len(adapter_files)} LoRA adapter file(s)")
                                            for f in adapter_files[:10]:
                                                file_size = f.stat().st_size / 1024 / 1024  # Size in MB
                                                terminal_output.append(f"[FILE]   - {f.name} ({file_size:.2f} MB)")
                                        
                                            # Verify essential files
                                            has_config = (adapter_dir_local / "adapter_config.json").exists()
                                            has_weights = (adapter_dir_local / "adapter_model.safetensors").exists() or (adapter_dir_local / "adapter_model.bin").exists()
                                            
                                            if has_config and has_weights:
                                                # Mark weights as downloaded
                                                active_job["weights_downloaded"] = True
                                                active_job["version_dir"] = str(version_dir)
                                                active_job["weights_path"] = str(weights_dir)
                                                training_manager._save_job(active_job)
                                                
                                                terminal_output.append(f"[SUCCESS] LoRA adapter weights downloaded successfully!")
                                                terminal_output.append(f"[INFO] Adapter saved to: {adapter_dir_local}")
                                                terminal_output.append(f"[INFO] These adapter files will be loaded onto the base model locally")
                                                
                                                st.session_state[terminal_output_key] = terminal_output
                                                st.success("✅ LoRA adapter weights downloaded successfully! You can now proceed to end training.")
                                                st.rerun()
                                            else:
                                                missing = []
                                                if not has_config:
                                                    missing.append("adapter_config.json")
                                                if not has_weights:
                                                    missing.append("adapter weights (adapter_model.safetensors or .bin)")
                                                terminal_output.append(f"[ERROR] Missing essential adapter files: {', '.join(missing)}")
                                                st.session_state[terminal_output_key] = terminal_output
                                                st.error(f"❌ Missing essential adapter files: {', '.join(missing)}")
                                        else:
                                            terminal_output.append(f"[WARNING] No adapter files found after download")
                                            st.session_state[terminal_output_key] = terminal_output
                                            st.warning("⚠️ No adapter files found after download. Please check the instance manually.")
                                    else:
                                        terminal_output.append(f"[ERROR] Adapter directory was not created/downloaded")
                                        st.session_state[terminal_output_key] = terminal_output
                                        st.error("❌ Adapter directory not found. Please check the instance manually.")
                                
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
                                
                                # Initialize package_info early to avoid undefined variable errors
                                package_info = active_job.get("package_info")
                                if not package_info:
                                    # Try all_package_infos as fallback
                                    all_package_infos = active_job.get("all_package_infos", [])
                                    if all_package_infos and len(all_package_infos) > 0:
                                        package_info = all_package_infos[0]
                                
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
                                    if package_info:
                                        dataset_path = package_info.get("dataset_path")
                                        if dataset_path:
                                            terminal_output.append(f"[FILE DEBUG] Dataset path: {dataset_path}")
                                    else:
                                        terminal_output.append(f"[FILE DEBUG] No package_info available - using moved files only")
                                
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
                                if package_info:
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
                                else:
                                    terminal_output.append(f"[FILE DEBUG] No package_info available - cannot check dataset path")
                                
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
                    
                    # Check if instance is still running and offer inference setup
                    instance_id = active_job.get("instance_id")
                    ssh_host = active_job.get("ssh_host")
                    # Check for SSH port override first (user-specified port takes precedence)
                    ssh_port_override = active_job.get("ssh_port_override")
                    if ssh_port_override:
                        ssh_port = ssh_port_override
                    else:
                        ssh_port = active_job.get("ssh_port", 22)
                    instance_running = False
                    inference_ready = active_job.get("inference_ready", False)
                    inference_url = active_job.get("inference_url", "")
                    
                    if instance_id and vast_api_key:
                        try:
                            from utils.vast_ai_client import VastAIClient
                            vast_client = VastAIClient(api_key=vast_api_key)
                            instance_status = vast_client.get_instance_status(instance_id)
                            
                            # Check if instance is actually running
                            if "instances" in instance_status:
                                instance_data = instance_status["instances"]
                            else:
                                instance_data = instance_status
                            
                            actual_status = instance_data.get("actual_status", "unknown")
                            instance_running = actual_status in ["running", "starting"]
                            
                        except Exception as e:
                            error_msg = str(e)
                            if "INSTANCE_NOT_FOUND" in error_msg or "404" in error_msg:
                                instance_running = False
                            else:
                                # Instance might still exist, try SSH check
                                if ssh_host:
                                    try:
                                        import subprocess
                                        test_cmd = [
                                            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", 
                                            "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                                            f"root@{ssh_host}",
                                            "echo 'connected'"
                                        ]
                                        test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=5)
                                        instance_running = test_result.returncode == 0
                                    except:
                                        instance_running = False
                    
                    
                    # Additional actions
                    if active_job.get("finalized"):
                        st.success("✅ Training finalized! All phases complete.")
                        if instance_id and not instance_running:
                            st.info("ℹ️ Instance has been stopped or destroyed.")
                    else:
                        # Allow going back to Phase 3 to redo training if needed
                        confirm_key_phase3_from4 = f"confirm_redo_phase_3_from_4_{active_job.get('instance_id')}"
                        if confirm_key_phase3_from4 not in st.session_state:
                            st.session_state[confirm_key_phase3_from4] = False
                        
                        if st.session_state[confirm_key_phase3_from4]:
                            if st.button("✅ Confirm Go Back to Phase 3", key="confirm_redo_phase_3_from_4_btn", type="primary"):
                                st.session_state[phase_key] = 3
                                st.session_state[confirm_key_phase3_from4] = False
                                st.rerun()
                            if st.button("❌ Cancel", key="cancel_redo_phase_3_from_4"):
                                st.session_state[confirm_key_phase3_from4] = False
                                st.rerun()
                        else:
                            if st.button("🔄 Redo Phase 3", key="redo_phase_3_from_4", type="secondary", help="Go back to Phase 3 to restart training. This will return you to the training phase."):
                                st.session_state[confirm_key_phase3_from4] = True
                                st.rerun()
                
            except Exception as e:
                st.warning(f"Could not load training jobs: {str(e)}")
                import traceback
                with st.expander("Error Details", expanded=False):
                    st.code(traceback.format_exc())
        
        # Debug Info section - at bottom, collapsed by default
        st.markdown("---")
        with st.expander("🔍 Debug Info", expanded=False):
            # Only show debug info if there's an active job
            if active_job:
                terminal_output_key = f"terminal_output_{active_job.get('instance_id')}"
                current_terminal_output = list(st.session_state.get(terminal_output_key, []))
                last_10_lines = current_terminal_output[-10:] if len(current_terminal_output) >= 10 else current_terminal_output
                st.json({
                    "terminal_output_key": terminal_output_key,
                    "has_key": terminal_output_key in st.session_state,
                    "instance_id": active_job.get("instance_id"),
                    "terminal_length": len(current_terminal_output),
                    "last_10_lines": last_10_lines,
                    "session_state_keys": [k for k in st.session_state.keys() if "terminal" in k.lower()][:10]
                })
                # Show last lines to see what's actually in the terminal
                if last_10_lines:
                    st.text("Last 10 lines in terminal:")
                    for i, line in enumerate(last_10_lines, 1):
                        line_num = len(current_terminal_output) - len(last_10_lines) + i
                        st.text(f"{line_num}: {line[:100]}")
            else:
                st.info("No active job - debug info will appear when a training job is active.")
    
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
