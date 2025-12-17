"""Vast.ai integration for launching training jobs"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path
import requests


class VastAIClient:
    """Client for interacting with Vast.ai API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Vast.ai client
        
        Args:
            api_key: Vast.ai API key. If None, will try to get from environment variable VAST_API_KEY
        """
        self.api_key = api_key or os.getenv("VAST_API_KEY")
        if not self.api_key:
            raise ValueError("Vast.ai API key is required. Set VAST_API_KEY environment variable or pass api_key parameter.")
        
        # Vast.ai API base URL - try both possible formats
        self.base_url = "https://console.vast.ai/api/v0"
        # Alternative base URL (some endpoints might use this)
        self.alt_base_url = "https://console.vast.ai/api/v0"
        # Vast.ai API authentication - use API key as query parameter
        self.headers = {
            "Accept": "application/json"
        }
        self.api_key_param = self.api_key
    
    def search_offers(self, 
                     gpu_name: Optional[str] = None,
                     min_gpu_ram: Optional[int] = None,
                     min_disk_space: int = 50,
                     max_price: Optional[float] = None,
                     num_gpus: Optional[int] = None) -> List[Dict]:
        """
        Search for available GPU offers on Vast.ai
        
        Args:
            gpu_name: Filter by GPU name (e.g., "RTX 3090", "A100")
            min_gpu_ram: Minimum GPU RAM in GB
            min_disk_space: Minimum disk space in GB
            max_price: Maximum price per hour in USD
        
        Returns:
            List of available offers
        """
        try:
            # Vast.ai API uses POST to /bundles/ endpoint with JSON body
            # According to API docs: https://docs.vast.ai/api-reference/search/search-offers
            api_url = f"{self.base_url}/bundles/"
            
            # Build request body with filters
            # Vast.ai API structure - filters go directly in request body
            request_body = {
                "limit": 100,
                "type": "on-demand",
                "verified": {"eq": True},
                "rentable": {"eq": True},
                "rented": {"eq": False}
            }
            
            # Add GPU filters directly to request body
            # Note: gpu_ram is in MB, not GB (convert GB to MB)
            if gpu_name:
                request_body["gpu_name"] = {"eq": gpu_name}
            if min_gpu_ram:
                # Convert GB to MB (API expects MB)
                gpu_ram_mb = min_gpu_ram * 1024
                request_body["gpu_ram"] = {"gte": gpu_ram_mb}
            if min_disk_space:
                request_body["disk_space"] = {"gte": min_disk_space}
            if max_price:
                request_body["dph_total"] = {"lte": max_price}
            if num_gpus:
                request_body["num_gpus"] = {"eq": num_gpus}
            
            # Headers with Bearer token authentication
            headers_with_auth = self.headers.copy()
            headers_with_auth["Authorization"] = f"Bearer {self.api_key}"
            headers_with_auth["Content-Type"] = "application/json"
            
            # Make POST request
            response = requests.post(
                api_url,
                headers=headers_with_auth,
                json=request_body,
                timeout=30
            )
            
            # Log API response
            print(f"[API] search_offers response: status_code={response.status_code}")
            try:
                response_json = response.json()
                print(f"[API] search_offers response body (first 1000 chars): {json.dumps(response_json, indent=2)[:1000]}")
                if isinstance(response_json, dict):
                    offers = response_json.get("offers") or response_json.get("results") or response_json.get("data") or []
                    if isinstance(offers, list):
                        print(f"[API] search_offers found {len(offers)} offers")
            except:
                print(f"[API] search_offers response body (non-JSON): {response.text[:500]}")
            
            # Check response
            if response.status_code != 200:
                # Try to get error details
                try:
                    error_data = response.json()
                    error_msg = error_data.get("msg", response.text[:500])
                except:
                    error_msg = response.text[:500]
                raise Exception(f"Vast.ai API returned {response.status_code}: {error_msg}")
            
            # Check if we still got HTML
            if "text/html" in response.headers.get("content-type", "") or response.text.strip().startswith("<!DOCTYPE"):
                raise Exception(
                    f"Vast.ai API returned HTML instead of JSON. This usually means:\n"
                    f"1. The API endpoint URL is incorrect\n"
                    f"2. The API key is invalid or expired\n"
                    f"3. Authentication format is wrong\n\n"
                    f"URL attempted: {api_url}\n"
                    f"Please verify your API key is correct and check Vast.ai API documentation."
                )
            
            # Parse response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON response from Vast.ai API: {response.text[:500]}")
            
            # Response format from /bundles/ endpoint
            if isinstance(data, dict):
                # The /bundles/ endpoint returns offers in different possible keys
                offers = data.get("offers") or data.get("results") or data.get("data") or []
                if isinstance(offers, list):
                    return offers
                # Sometimes it's a dict with nested offers
                if isinstance(offers, dict):
                    return offers.get("offers", [])
            elif isinstance(data, list):
                return data
            
            return []
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error connecting to Vast.ai API: {str(e)}")
        except Exception as e:
            raise Exception(f"Error searching Vast.ai offers: {str(e)}")
    
    def create_instance(self,
                       offer_id: str,
                       image: str = "pytorch/pytorch:latest",
                       disk_space: int = 50,
                       env_vars: Optional[Dict[str, str]] = None,
                       onstart_cmd: Optional[str] = None) -> Dict:
        """
        Create a new instance on Vast.ai
        
        Args:
            offer_id: ID of the offer to use
            image: Docker image to use
            disk_space: Disk space in GB
            env_vars: Environment variables to set
            onstart_cmd: Command to run on instance start
        
        Returns:
            Instance creation response
        """
        # Vast.ai API payload format
        # Only include essential options to minimize cost
        # We explicitly avoid optional features that may increase pricing:
        # - No Jupyter notebook access
        # - No public IP (SSH only via Vast.ai's SSH gateway)
        # - Minimal disk space (only what's requested)
        # - No extra storage or network options
        # Build payload for instance creation
        payload = {
            "image": image,
            "disk": disk_space,
            "runtype": "ssh"  # SSH only, no Jupyter or other services
        }
        
        if onstart_cmd:
            payload["onstart"] = onstart_cmd
        
        if env_vars:
            payload["env"] = env_vars
        
        # Headers with Bearer token authentication (required for instance creation)
        headers_with_auth = self.headers.copy()
        headers_with_auth["Authorization"] = f"Bearer {self.api_key}"
        headers_with_auth["Content-Type"] = "application/json"
        
        try:
            response = requests.put(
                f"{self.base_url}/asks/{offer_id}/",
                headers=headers_with_auth,
                json=payload,
                timeout=60
            )
            
            # Log API response
            print(f"[API] create_instance response: status_code={response.status_code}")
            try:
                response_json = response.json()
                print(f"[API] create_instance response body: {json.dumps(response_json, indent=2)[:1000]}")  # Limit to 1000 chars
            except:
                print(f"[API] create_instance response body (non-JSON): {response.text[:500]}")
            
            # Get detailed error information
            if not response.ok:
                error_detail = ""
                try:
                    error_json = response.json()
                    error_detail = f" - {json.dumps(error_json, indent=2)}"
                except:
                    error_detail = f" - {response.text[:500]}"
                
                raise Exception(
                    f"Error creating Vast.ai instance: {response.status_code} {response.reason}{error_detail}\n"
                    f"Payload sent: {json.dumps(payload, indent=2)}"
                )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Extract more details from the error
            error_detail = ""
            if hasattr(e.response, 'text'):
                try:
                    error_json = e.response.json()
                    error_detail = f"\nAPI Error Details: {json.dumps(error_json, indent=2)}"
                except:
                    error_detail = f"\nAPI Error Response: {e.response.text[:500]}"
            raise Exception(f"Error creating Vast.ai instance: {str(e)}{error_detail}")
        except Exception as e:
            raise Exception(f"Error creating Vast.ai instance: {str(e)}")
    
    def get_instance_status(self, instance_id: str) -> Dict:
        """
        Get status of an instance
        
        Args:
            instance_id: ID of the instance
        
        Returns:
            Instance status information
        
        Raises:
            Exception with specific message if instance not found (404)
        """
        try:
            # Use Bearer token authentication
            headers_with_auth = self.headers.copy()
            headers_with_auth["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(
                f"{self.base_url}/instances/{instance_id}/",
                headers=headers_with_auth,
                timeout=30
            )
            
            # Log API response
            print(f"[API] get_instance_status response: status_code={response.status_code}, instance_id={instance_id}")
            try:
                response_json = response.json()
                print(f"[API] get_instance_status response body: {json.dumps(response_json, indent=2)[:1000]}")  # Limit to 1000 chars
            except:
                print(f"[API] get_instance_status response body (non-JSON): {response.text[:500]}")
            
            # Check for 404 explicitly - this means instance doesn't exist
            if response.status_code == 404:
                raise Exception(f"INSTANCE_NOT_FOUND: Instance {instance_id} not found (404)")
            
            # Check if we got HTML instead of JSON (might indicate wrong endpoint or auth issue)
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                # Try to parse as JSON anyway, but if it fails, we know it's HTML
                try:
                    return response.json()
                except:
                    raise Exception(f"INSTANCE_NOT_FOUND: Got HTML response (instance may not exist or auth failed)")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Check status code directly
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 404:
                    raise Exception(f"INSTANCE_NOT_FOUND: Instance {instance_id} not found (404)")
                elif e.response.status_code == 429:
                    # Rate limit error - provide helpful message
                    error_detail = ""
                    try:
                        error_json = e.response.json()
                        error_detail = f" - {json.dumps(error_json)}"
                    except:
                        error_detail = f" - {e.response.text[:200]}"
                    raise Exception(f"RATE_LIMIT: Too many requests (429). Please wait a moment before refreshing.{error_detail}")
                error_detail = ""
                try:
                    error_json = e.response.json()
                    error_detail = f" - {json.dumps(error_json)}"
                except:
                    error_detail = f" - {e.response.text[:200]}"
                raise Exception(f"Error getting instance status: {e.response.status_code} {error_detail}")
            else:
                raise Exception(f"Error getting instance status: {str(e)}")
        except Exception as e:
            # Re-raise with INSTANCE_NOT_FOUND marker if it's a 404
            error_msg = str(e)
            if "INSTANCE_NOT_FOUND" in error_msg:
                raise  # Already marked
            if "RATE_LIMIT" in error_msg:
                raise  # Already marked
            if "404" in error_msg or "not found" in error_msg.lower():
                raise Exception(f"INSTANCE_NOT_FOUND: {error_msg}")
            raise Exception(f"Error getting instance status: {str(e)}")
    
    def destroy_instance(self, instance_id: str) -> bool:
        """
        Destroy an instance
        
        Args:
            instance_id: ID of the instance to destroy
        
        Returns:
            True if successful
        """
        try:
            # Use Bearer token authentication (required for instance operations)
            headers_with_auth = self.headers.copy()
            headers_with_auth["Authorization"] = f"Bearer {self.api_key}"
            headers_with_auth["Content-Type"] = "application/json"
            
            response = requests.delete(
                f"{self.base_url}/instances/{instance_id}/",
                headers=headers_with_auth,
                timeout=30
            )
            
            # Check for 404 - instance already destroyed
            if response.status_code == 404:
                print(f"Instance {instance_id} already destroyed or not found (404)")
                return True
            
            response.raise_for_status()
            
            # Check response
            try:
                result = response.json()
                if isinstance(result, dict):
                    success = result.get("success", True)
                    if not success:
                        error_msg = result.get("error") or result.get("msg", "Unknown error")
                        raise Exception(f"Vast.ai API returned error: {error_msg}")
            except json.JSONDecodeError:
                # Response might not be JSON, that's okay if status is 200
                if response.status_code == 200:
                    return True
                raise Exception(f"Unexpected response format: {response.text[:200]}")
            
            return True
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = f" - {json.dumps(error_json)}"
                except:
                    error_detail = f" - {e.response.text[:200]}"
            raise Exception(f"Error destroying instance: {e.response.status_code if hasattr(e, 'response') else 'unknown'}{error_detail}")
        except Exception as e:
            raise Exception(f"Error destroying instance: {str(e)}")
    
    def list_instances(self) -> List[Dict]:
        """
        List all your instances
        
        Returns:
            List of instances
        """
        try:
            headers_with_auth = self.headers.copy()
            headers_with_auth["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(
                f"{self.base_url}/instances",
                headers=headers_with_auth,
                params={"owner": "me"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                # Filter out non-dict items (in case API returns mixed types)
                return [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                instances = data.get("instances", []) or data.get("results", []) or []
                # Filter out non-dict items
                if isinstance(instances, list):
                    return [item for item in instances if isinstance(item, dict)]
                return []
            return []
        except Exception as e:
            raise Exception(f"Error listing instances: {str(e)}")
    
    def start_instance(self, instance_id: str) -> bool:
        """
        Start a stopped instance
        
        Args:
            instance_id: ID of the instance to start
        
        Returns:
            True if successful
        """
        try:
            headers_with_auth = self.headers.copy()
            headers_with_auth["Authorization"] = f"Bearer {self.api_key}"
            headers_with_auth["Content-Type"] = "application/json"
            
            response = requests.put(
                f"{self.base_url}/instances/{instance_id}/start",
                headers=headers_with_auth,
                timeout=30
            )
            
            # Log API response
            print(f"[API] start_instance response: status_code={response.status_code}, instance_id={instance_id}")
            try:
                response_json = response.json()
                print(f"[API] start_instance response body: {json.dumps(response_json, indent=2)[:1000]}")  # Limit to 1000 chars
            except:
                print(f"[API] start_instance response body (non-JSON): {response.text[:500]}")
            
            response.raise_for_status()
            
            # Check response
            try:
                result = response.json()
                if isinstance(result, dict):
                    success = result.get("success", False) or result.get("status") == "ok"
                    if not success:
                        error_msg = result.get("error") or result.get("msg", "Unknown error")
                        raise Exception(f"Vast.ai API returned error: {error_msg}")
            except json.JSONDecodeError:
                # Response might not be JSON, that's okay if status is 200
                if response.status_code == 200:
                    return True
                raise Exception(f"Unexpected response format: {response.text[:200]}")
            
            return True
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = f" - {json.dumps(error_json)}"
                except:
                    error_detail = f" - {e.response.text[:200]}"
            raise Exception(f"Error starting instance: {e.response.status_code if hasattr(e, 'response') else 'unknown'}{error_detail}")
        except Exception as e:
            raise Exception(f"Error starting instance: {str(e)}")
    
    def stop_instance(self, instance_id: str) -> bool:
        """
        Stop a running instance (does not destroy it)
        
        Args:
            instance_id: ID of the instance to stop
        
        Returns:
            True if successful
        """
        try:
            headers_with_auth = self.headers.copy()
            headers_with_auth["Authorization"] = f"Bearer {self.api_key}"
            headers_with_auth["Content-Type"] = "application/json"
            
            response = requests.put(
                f"{self.base_url}/instances/{instance_id}/stop",
                headers=headers_with_auth,
                timeout=30
            )
            response.raise_for_status()
            
            # Check response
            try:
                result = response.json()
                if isinstance(result, dict):
                    success = result.get("success", False) or result.get("status") == "ok"
                    if not success:
                        error_msg = result.get("error") or result.get("msg", "Unknown error")
                        raise Exception(f"Vast.ai API returned error: {error_msg}")
            except json.JSONDecodeError:
                # Response might not be JSON, that's okay if status is 200
                if response.status_code == 200:
                    return True
                raise Exception(f"Unexpected response format: {response.text[:200]}")
            
            return True
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = f" - {json.dumps(error_json)}"
                except:
                    error_detail = f" - {e.response.text[:200]}"
            raise Exception(f"Error stopping instance: {e.response.status_code if hasattr(e, 'response') else 'unknown'}{error_detail}")
        except Exception as e:
            raise Exception(f"Error stopping instance: {str(e)}")
    

