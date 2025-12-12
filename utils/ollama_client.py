"""Ollama client for chat interactions"""

try:
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        import ollama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False

from typing import List, Dict, Optional

class OllamaClient:
    """Client for interacting with Ollama models"""
    
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        if OLLAMA_AVAILABLE:
            try:
                # Try using Client class (newer API)
                self.client = Client()
            except:
                # Fall back to direct ollama module
                self.client = None
        else:
            self.client = None
    
    def resolve_model_name(self, model_name: str = None) -> str:
        """Resolve a base model name to the actual full model name (with tag) in Ollama.
        If the model name already has a tag, return it as-is.
        Otherwise, find the first matching model with any tag."""
        model_to_resolve = model_name or self.model_name
        if not model_to_resolve:
            return model_to_resolve
        
        # If model name already has a tag, return as-is
        if ':' in model_to_resolve:
            return model_to_resolve
        
        # Get available models
        available_models = self.list_models()
        if not available_models:
            return model_to_resolve
        
        # Normalize the base name for comparison
        base_name = model_to_resolve.strip().lower()
        
        # Find matching models - prefer models with tags over base names
        # This is because Ollama requires the full model name with tag for API calls
        matches_with_tags = []
        exact_match = None
        
        for available in available_models:
            available_base = available.split(':')[0] if ':' in available else available
            if available_base.strip().lower() == base_name:
                # If it's an exact match (same as input), store it
                if available.strip().lower() == base_name:
                    exact_match = available
                # If it has a tag, add to tagged matches
                elif ':' in available:
                    matches_with_tags.append(available)
        
        # Prefer :latest tag, then any other tag, then exact match
        for match in matches_with_tags:
            if ':latest' in match.lower():
                return match
        
        # Return first tagged match if available
        if matches_with_tags:
            return matches_with_tags[0]
        
        # Return exact match if found
        if exact_match is not None:
            return exact_match
        
        # If no match found, return original (will fail later with better error message)
        return model_to_resolve
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, timeout: int = 120) -> str:
        """Send chat messages to Ollama"""
        if not OLLAMA_AVAILABLE:
            return "Error: Ollama package not installed. Please install with: pip install ollama"
        
        # Check if model exists before trying to use it
        if not self.model_exists():
            available = self.list_models()
            available_str = ", ".join(available) if available else "none"
            return f"Error: Model '{self.model_name}' not found in Ollama. Available models: {available_str}. Please pull the model first with: ollama pull {self.model_name}"
        
        # Resolve the actual model name (with tag if needed)
        resolved_model_name = self.resolve_model_name()
        
        # Try HTTP API first (more reliable with timeout)
        # Try non-streaming first as it's more reliable
        try:
            import requests
            import json
            
            print(f"DEBUG: Starting Ollama chat request")
            print(f"DEBUG: Model: {self.model_name} -> {resolved_model_name}")
            print(f"DEBUG: Messages: {len(messages)} messages")
            
            url = "http://localhost:11434/api/chat"
            
            # Use streaming if requested
            payload = {
                "model": resolved_model_name,
                "messages": messages,
                "stream": stream
            }
            
            if stream:
                # Streaming mode
                try:
                    response = requests.post(url, json=payload, timeout=(10, timeout), stream=True)
                    if response.status_code == 200:
                        full_response = ""
                        for line in response.iter_lines():
                            if line:
                                try:
                                    chunk = json.loads(line)
                                    if 'message' in chunk and 'content' in chunk['message']:
                                        content = chunk['message']['content']
                                        if content:
                                            full_response += content
                                    elif 'response' in chunk:
                                        full_response += chunk['response']
                                    
                                    # Check if done
                                    if chunk.get('done', False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                        return full_response.strip() if full_response else "Error: Received empty streaming response"
                    else:
                        return f"Error: Ollama API returned status {response.status_code}. {response.text}"
                except requests.exceptions.Timeout:
                    return f"Error: Request timed out after {timeout} seconds."
                except Exception as e:
                    return f"Error: Streaming request failed: {str(e)}"
            
            # Non-streaming mode
            try:
                response = requests.post(url, json=payload, timeout=(10, timeout))
            except requests.exceptions.Timeout as e:
                return f"Error: Request timed out after {timeout} seconds. The model may be taking too long to respond."
            except Exception as e:
                return f"Error: Request failed: {str(e)}"
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle non-streaming response
                content = None
                if 'message' in data:
                    if isinstance(data['message'], dict):
                        if 'content' in data['message']:
                            content = data['message']['content']
                        elif 'text' in data['message']:
                            content = data['message']['text']
                    elif isinstance(data['message'], str):
                        content = data['message']
                
                if content:
                    return content.strip()
                else:
                    return f"Error: Could not extract content from response. Full response: {json.dumps(data)}"
                
                # Old streaming code (not used now, but kept for reference)
                if False and stream:
                    # Handle streaming response
                    print("DEBUG: Processing streaming response...")
                    full_content = ""
                    line_count = 0
                    try:
                        for line in response.iter_lines():
                            if line:
                                line_count += 1
                                print(f"DEBUG: Got line {line_count}: {line[:100] if len(line) > 100 else line}")
                                try:
                                    chunk = json.loads(line)
                                    print(f"DEBUG: Parsed chunk keys: {list(chunk.keys())}")
                                    
                                    if 'message' in chunk:
                                        print(f"DEBUG: Message type: {type(chunk['message'])}")
                                        if isinstance(chunk['message'], dict):
                                            print(f"DEBUG: Message dict keys: {list(chunk['message'].keys())}")
                                            if 'content' in chunk['message']:
                                                content = chunk['message']['content']
                                                print(f"DEBUG: Extracted content: {repr(content[:100])}")
                                                if content:
                                                    full_content += content
                                            elif 'text' in chunk['message']:
                                                content = chunk['message']['text']
                                                print(f"DEBUG: Extracted text: {repr(content[:100])}")
                                                if content:
                                                    full_content += content
                                        elif isinstance(chunk['message'], str):
                                            print(f"DEBUG: Message is string: {repr(chunk['message'][:100])}")
                                            full_content += chunk['message']
                                    elif 'response' in chunk:
                                        print(f"DEBUG: Found response key: {repr(chunk['response'][:100])}")
                                        full_content += chunk['response']
                                    
                                    # Check if done
                                    if chunk.get('done', False):
                                        print(f"DEBUG: Stream complete. Total lines: {line_count}, Total content: {len(full_content)} chars")
                                        print(f"DEBUG: Final content preview: {repr(full_content[:200])}")
                                        break
                                except json.JSONDecodeError as e:
                                    print(f"DEBUG: JSON decode error on line {line_count}: {e}")
                                    print(f"DEBUG: Line content: {repr(line[:200])}")
                                    continue
                    except Exception as stream_error:
                        print(f"DEBUG: Error iterating stream: {stream_error}")
                        import traceback
                        print(traceback.format_exc())
                    
                    print(f"DEBUG: Final content length: {len(full_content)}")
                    print(f"DEBUG: Final content: {repr(full_content[:500])}")
                    if full_content:
                        result = full_content.strip()
                        print(f"DEBUG: Returning: {repr(result[:200])}")
                        return result
                    else:
                        print("DEBUG: No content extracted from stream")
                        return "Error: Received empty streaming response"
                
            else:
                error_text = response.text
                print(f"DEBUG: Ollama API error - Status {response.status_code}: {error_text}")
                return f"Error: Ollama API returned status {response.status_code}. {error_text}"
                
        except requests.exceptions.Timeout as timeout_error:
            return f"Error: Request timed out after {timeout} seconds. The model may be taking too long to respond. Try a smaller model or check if Ollama is running properly."
        except requests.exceptions.ConnectionError as conn_error:
            return f"Error: Cannot connect to Ollama server at http://localhost:11434. Make sure Ollama is running. Error: {str(conn_error)}"
        except Exception as http_error:
            return f"Error: HTTP request failed: {str(http_error)}"
        
        # Fallback to Python library if HTTP API fails
        try:
            # Resolve the actual model name (with tag if needed)
            resolved_model_name = self.resolve_model_name()
            
            if self.client:
                # Use Client class
                response = self.client.chat(
                    model=resolved_model_name,
                    messages=messages,
                    stream=stream
                )
            else:
                # Use direct ollama module
                response = ollama.chat(
                    model=resolved_model_name,
                    messages=messages,
                    stream=stream
                )
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        full_response += chunk['message']['content']
                return full_response if full_response else "Error: Empty response from model"
            else:
                # Non-streaming response
                if isinstance(response, dict):
                    content = response.get('message', {}).get('content', '')
                    if not content:
                        content = response.get('response', '')
                    if content:
                        return content
                    # If no content found, try to extract from response structure
                    return str(response.get('message', response))
                elif isinstance(response, str):
                    return response
                else:
                    return str(response) if response else "Error: Empty response from model"
                    
        except Exception as python_error:
            error_msg = str(python_error)
            # Provide helpful error message for 404 errors
            if "404" in error_msg or "not found" in error_msg.lower():
                available = self.list_models()
                available_str = ", ".join(available) if available else "none"
                return f"Error: Model '{self.model_name}' not found. Available models: {available_str}. Please pull the model first with: ollama pull {self.model_name}"
            return f"Error: {error_msg}. Ollama server may not be running or the model may be unavailable."
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text from a prompt"""
        if not OLLAMA_AVAILABLE:
            return "Error: Ollama package not installed"
        
        # Resolve the actual model name (with tag if needed)
        resolved_model_name = self.resolve_model_name()
        
        try:
            if self.client:
                response = self.client.generate(
                    model=resolved_model_name,
                    prompt=prompt,
                    stream=stream
                )
            else:
                response = ollama.generate(
                    model=resolved_model_name,
                    prompt=prompt,
                    stream=stream
                )
            
            if stream:
                full_response = ""
                for chunk in response:
                    if 'response' in chunk:
                        full_response += chunk['response']
                return full_response
            else:
                return response['response']
        except Exception as e:
            return f"Error: {str(e)}"
    
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        # Primary method: use subprocess since it's more reliable
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Has header and at least one model
                    models = []
                    for line in lines[1:]:  # Skip header line
                        if line.strip():
                            # Parse the NAME column (first column)
                            parts = line.split()
                            if parts:
                                model_name = parts[0]
                                # Add both full name and base name for matching
                                if ':' in model_name:
                                    base_name = model_name.split(':')[0]
                                    models.append(base_name)
                                    models.append(model_name)  # Also include full name
                                else:
                                    models.append(model_name)
                    return list(set(models))  # Remove duplicates
        except Exception as subprocess_error:
            print(f"Error using subprocess to list models: {subprocess_error}")
        
        # Fallback: try Python API if available
        if OLLAMA_AVAILABLE:
            try:
                if self.client:
                    # Using Client class
                    response = self.client.list()
                    models = response.get('models', [])
                else:
                    # Using direct ollama module
                    response = ollama.list()
                    models = response.get('models', [])
                
                # Extract model names - handle different response formats
                model_names = []
                for model in models:
                    # Try different possible keys for model name
                    if isinstance(model, dict):
                        # Could be 'name', 'model', or the model dict might have 'name' nested
                        name = model.get('name') or model.get('model')
                        if name:
                            model_names.append(name)
                    elif isinstance(model, str):
                        # Sometimes it's just a string
                        model_names.append(model)
                
                return model_names
            except Exception as e:
                print(f"Error listing models via API: {e}")
        
        return []
    
    def model_exists(self, model_name: str = None) -> bool:
        """Check if a model exists in Ollama"""
        model_to_check = model_name or self.model_name
        if not model_to_check:
            return False
            
        available_models = self.list_models()
        
        if not available_models:
            return False
        
        # Normalize model names for comparison (lowercase, remove extra spaces)
        model_to_check = model_to_check.strip().lower()
        
        # Check exact match (case-insensitive)
        for available in available_models:
            if available.strip().lower() == model_to_check:
                return True
        
        # Check if model name matches any available model (handling tags like :latest)
        check_base = model_to_check.split(':')[0] if ':' in model_to_check else model_to_check
        check_base = check_base.strip()
        
        for available in available_models:
            # Remove tag for comparison
            available_base = available.split(':')[0] if ':' in available else available
            available_base = available_base.strip().lower()
            
            if available_base == check_base:
                return True
        
        return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.list_models()

