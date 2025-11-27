"""DSPy Signatures and Modules for Retail Analytics Copilot"""
import dspy
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configuration constants (merged from config.py)
OLLAMA_MODEL = "phi3.5:3.8b-mini-instruct-q4_K_M"  # Local quantized model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Type definitions (merged from analytics_types.py)
class QueryIntent(BaseModel):
    """Detected intent of the user query"""
    requires_sql: bool = Field(description="Whether SQL query is needed")
    requires_rag: bool = Field(description="Whether document search is needed")
    sql_tables: List[str] = Field(default_factory=list, description="Relevant database tables")
    keywords: List[str] = Field(default_factory=list, description="Key terms for RAG search")


# Use litellm for better DSPy compatibility with Ollama
def get_ollama_lm():
    """Get Ollama LM using litellm (better DSPy compatibility)"""
    # Always use CustomOllamaLM for now to debug the issue
    # Once we fix the response extraction, we can try litellm again
    print(f"[INFO] Using CustomOllamaLM with model: {OLLAMA_MODEL}")
    return CustomOllamaLM()
    
    # Try litellm as alternative (commented out for debugging)
    # try:
    #     import litellm
    #     model_name = f"ollama/{OLLAMA_MODEL}"
    #     import os
    #     os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL
    #     lm = dspy.LM(model=model_name)
    #     print(f"[INFO] Using litellm with model: {model_name}")
    #     return lm
    # except (ImportError, Exception) as e:
    #     print(f"[WARNING] Using custom OllamaLM (litellm failed: {e})")
    #     return CustomOllamaLM()


class CustomOllamaLM(dspy.LM):
    """DSPy-compatible Ollama language model (fallback)"""
    
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        super().__init__(model)
        self.model = model
        self.base_url = base_url
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
        except ImportError:
            raise ImportError("ollama package required. Install with: pip install ollama")
    
    def basic_request(self, prompt: str, **kwargs) -> str:
        """Make request to Ollama (local, no external network calls)
        
        Constraint: Prompts should be ≤1k tokens (enforced by caller)
        """
        import sys
        from pathlib import Path
        
        # Debug: Write to file for better visibility
        debug_file = Path(__file__).parent.parent / "ollama_debug.log"
        def debug_log(msg):
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")
            print(msg, file=sys.stderr)
        
        # Debug: Log prompt being sent
        debug_log(f"[DEBUG] basic_request: Sending prompt (length: {len(prompt)})")
        if len(prompt) < 500:
            debug_log(f"[DEBUG] Prompt content: {prompt}")
        else:
            debug_log(f"[DEBUG] Prompt preview: {prompt[:200]}...")
        
        try:
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 2000)  # Increased for structured output
            
            # Local Ollama server - no external network calls
            # Use chat API for better instruction following
            # IMPORTANT: Set stream=False to get complete response
            debug_log(f"[DEBUG] Calling Ollama chat API with model={self.model}, max_tokens={max_tokens}")
            # Ensure we get complete response - explicitly disable streaming
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    stream=False  # Ensure we get complete response, not streamed
                )
            except Exception as chat_error:
                debug_log(f"[DEBUG] Chat API error: {chat_error}, trying without stream parameter")
                # Some Ollama versions might not support stream parameter
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )
            
            # Debug: Log raw response
            debug_log(f"[DEBUG] Raw response type: {type(response)}")
            debug_log(f"[DEBUG] Response type name: {type(response).__name__}")
            debug_log(f"[DEBUG] Has 'message' attribute: {hasattr(response, 'message')}")
            
            # CRITICAL: Handle ChatResponse objects from Ollama FIRST (they're not dicts!)
            # ChatResponse has a .message attribute which is a Message object with .content
            # We must check this BEFORE checking if it's a dict, because ChatResponse is not a dict
            # Check by both hasattr and type name to be safe
            is_chat_response = False
            if hasattr(response, 'message'):
                is_chat_response = True
                debug_log(f"[DEBUG] Detected ChatResponse via hasattr('message')")
            elif type(response).__name__ == 'ChatResponse' or 'ChatResponse' in str(type(response)):
                is_chat_response = True
                debug_log(f"[DEBUG] Detected ChatResponse via type name: {type(response).__name__}")
            
            if is_chat_response:
                # This is a ChatResponse object - extract the message content
                debug_log(f"[DEBUG] Response is ChatResponse object, accessing .message attribute")
                msg = response.message
                debug_log(f"[DEBUG] Message type: {type(msg)}")
                
                # Message object has .content attribute - this is the actual response text
                # CRITICAL: This is where we extract the actual text from ChatResponse
                if hasattr(msg, 'content'):
                    try:
                        content = msg.content
                        debug_log(f"[DEBUG] Extracted from ChatResponse.message.content: type={type(content)}")
                        if content is not None:
                            # Ensure content is a string
                            content_str = str(content) if not isinstance(content, str) else content
                            
                            # CRITICAL: Strip markdown code blocks if present (```json ... ```)
                            # DSPy's JSONAdapter expects raw JSON, not markdown-wrapped JSON
                            import re
                            original_length = len(content_str)
                            # Remove markdown code blocks (```json ... ``` or ``` ... ```)
                            content_str = re.sub(r'^```(?:json)?\s*\n', '', content_str, flags=re.MULTILINE)
                            content_str = re.sub(r'\n```\s*$', '', content_str, flags=re.MULTILINE)
                            content_str = content_str.strip()
                            
                            if original_length != len(content_str):
                                debug_log(f"[DEBUG] Stripped markdown code blocks: {original_length} -> {len(content_str)} chars")
                            
                            debug_log(f"[DEBUG] Content string length: {len(content_str)}")
                            debug_log(f"[DEBUG] Content preview: {content_str[:200]}")
                            
                            if len(content_str.strip()) > 1:
                                debug_log(f"[DEBUG] SUCCESS: Returning content length={len(content_str)}")
                                return content_str
                            else:
                                debug_log(f"[DEBUG] WARNING: Content is too short: '{content_str}'")
                        else:
                            debug_log(f"[DEBUG] WARNING: msg.content is None")
                    except Exception as e:
                        debug_log(f"[DEBUG] ERROR accessing msg.content: {e}")
                else:
                    debug_log(f"[DEBUG] WARNING: msg has no 'content' attribute")
                
                # Or message might be a dict
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    debug_log(f"[DEBUG] Extracted from ChatResponse.message dict: length={len(content)}")
                    if content and len(content.strip()) > 1:
                        return content
                
                # Try accessing as attribute
                try:
                    if hasattr(msg, '__dict__'):
                        debug_log(f"[DEBUG] Message __dict__ keys: {list(msg.__dict__.keys())}")
                        for key in ['content', 'text', 'response']:
                            if hasattr(msg, key):
                                val = getattr(msg, key)
                                if isinstance(val, str) and len(val.strip()) > 1:
                                    debug_log(f"[DEBUG] Found content in message.{key}: length={len(val)}")
                                    return val
                except Exception as e:
                    debug_log(f"[DEBUG] Error accessing message attributes: {e}")
            
            # Now check if it's a dict (for other response formats)
            if isinstance(response, dict):
                debug_log(f"[DEBUG] Response keys: {list(response.keys())}")
                debug_log(f"[DEBUG] Full response structure: {str(response)[:1000]}")
            
            # Extract response from chat format - comprehensive extraction
            if isinstance(response, dict):
                debug_log(f"[DEBUG] Response dict keys: {list(response.keys())}")
                
                # Method 1: Standard format - message.content
                content = ""
                if "message" in response:
                    msg = response["message"]
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        debug_log(f"[DEBUG] Method 1 (message.content): length={len(content)}")
                    elif isinstance(msg, str):
                        content = msg
                        debug_log(f"[DEBUG] Method 1 (message as string): length={len(content)}")
                
                # Method 2: Direct response field
                if not content and "response" in response:
                    content = response["response"]
                    debug_log(f"[DEBUG] Method 2 (response): length={len(content)}")
                
                # Method 3: text field
                if not content and "text" in response:
                    content = response["text"]
                    debug_log(f"[DEBUG] Method 3 (text): length={len(content)}")
                
                # Method 4: Check all string values
                if not content:
                    for key, value in response.items():
                        if isinstance(value, str) and len(value) > 10:
                            content = value
                            debug_log(f"[DEBUG] Method 4 (found in {key}): length={len(content)}")
                            break
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, str) and len(subvalue) > 10:
                                    content = subvalue
                                    debug_log(f"[DEBUG] Method 4 (found in {key}.{subkey}): length={len(content)}")
                                    break
                            if content:
                                break
                
                # Last resort
                if not content:
                    debug_log(f"[DEBUG] WARNING: No content found, using str(response)")
                    content = str(response)
                
                # Final validation and fix
                if len(content) <= 1:
                    debug_log(f"[DEBUG] CRITICAL: Only got '{content}' (length: {len(content)})")
                    debug_log(f"[DEBUG] Full response structure: {response}")
                    # Try one more time with a fresh call
                    try:
                        debug_log(f"[DEBUG] Attempting fresh Ollama call...")
                        fresh_response = self.client.chat(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            options={"temperature": temperature, "num_predict": max_tokens}
                        )
                        if isinstance(fresh_response, dict):
                            fresh_content = fresh_response.get("message", {}).get("content", "")
                            if len(fresh_content) > len(content):
                                debug_log(f"[DEBUG] Fresh call got longer content: {len(fresh_content)}")
                                content = fresh_content
                    except Exception as retry_error:
                        debug_log(f"[DEBUG] Retry failed: {retry_error}")
                elif len(content) < 10:
                    debug_log(f"[DEBUG] WARNING: Response short but > 1 char: '{content}' (length: {len(content)})")
                else:
                    debug_log(f"[DEBUG] SUCCESS: Extracted content length: {len(content)}")
                    debug_log(f"[DEBUG] Content preview (first 500): {content[:500]}")
                    debug_log(f"[DEBUG] Content preview (last 200): {content[-200:]}")
                
                # Ensure content is a proper string
                if not isinstance(content, str):
                    content = str(content)
                # Final safety check: ensure content is never just "m"
                content_stripped = content.strip()
                if not content or len(content) < 2 or content_stripped == 'm' or content_stripped == 'M':
                    debug_log(f"[DEBUG] CRITICAL: Content is '{content_stripped}' - using fallback JSON")
                    content = '{"reasoning": "Unable to extract valid response from Ollama", "requires_sql": false, "requires_rag": true}'
                
                return content
            
            # If we get here, response is not a dict and doesn't have .message
            # Try to extract from object attributes
            debug_log(f"[DEBUG] Response is not dict and has no .message (type: {type(response)}), trying attribute access")
            
            # Try common attribute names
            if hasattr(response, 'content'):
                content = response.content
                debug_log(f"[DEBUG] Found .content attribute: length={len(content)}")
                if content:
                    return content
            
            if hasattr(response, 'text'):
                content = response.text
                debug_log(f"[DEBUG] Found .text attribute: length={len(content)}")
                if content:
                    return content
            
            if hasattr(response, 'response'):
                content = response.response
                debug_log(f"[DEBUG] Found .response attribute: length={len(content)}")
                if content:
                    return content
            
            # Last resort: convert to string (but this might give us just "m")
            debug_log(f"[DEBUG] No attributes found, converting to string")
            content = str(response)
            debug_log(f"[DEBUG] Converted content length: {len(content)}")
            debug_log(f"[DEBUG] Converted content preview: {content[:200]}")
            
            # If string conversion gives us something suspiciously short, try to get more info
            if len(content.strip()) <= 1:
                debug_log(f"[DEBUG] WARNING: String conversion gave us only '{content}'")
                # Try to access all attributes
                if hasattr(response, '__dict__'):
                    debug_log(f"[DEBUG] Response __dict__: {list(response.__dict__.keys())}")
                    for attr_name in dir(response):
                        if not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(response, attr_name)
                                if isinstance(attr_value, str) and len(attr_value.strip()) > len(content.strip()):
                                    debug_log(f"[DEBUG] Found longer string in attribute '{attr_name}': length={len(attr_value)}")
                                    content = attr_value
                                    break
                                # Also check nested objects
                                if hasattr(attr_value, 'content'):
                                    nested_content = getattr(attr_value, 'content')
                                    if isinstance(nested_content, str) and len(nested_content.strip()) > len(content.strip()):
                                        debug_log(f"[DEBUG] Found content in nested '{attr_name}.content': length={len(nested_content)}")
                                        content = nested_content
                                        break
                            except Exception as attr_error:
                                debug_log(f"[DEBUG] Error accessing '{attr_name}': {attr_error}")
            
            # Final check: if content is still just "m", use fallback
            content_stripped = content.strip()
            if len(content_stripped) <= 1 or content_stripped == 'm' or content_stripped == 'M':
                debug_log(f"[DEBUG] CRITICAL: Content is still '{content_stripped}' after all attempts - using fallback")
                content = '{"reasoning": "Unable to extract valid response from Ollama", "requires_sql": false, "requires_rag": true}'
            
            return content
        except Exception as e:
            debug_log(f"[DEBUG] ERROR in chat API: {e}")
            import traceback
            debug_log(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
            # Fallback to generate if chat fails
            try:
                debug_log(f"[DEBUG] Falling back to generate API")
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                )
                debug_log(f"[DEBUG] Generate response type: {type(response)}")
                if isinstance(response, dict):
                    content = response.get("response", "")
                    # Handle streaming response
                    if not content and "text" in response:
                        content = response.get("text", "")
                    debug_log(f"[DEBUG] Generate extracted content length: {len(content)}")
                    return content
                return str(response)
            except Exception as e2:
                debug_log(f"[DEBUG] ERROR in generate API: {e2}")
                return f"Error: {str(e2)}"
    
    def __call__(self, prompt: str = None, **kwargs) -> str:
        """Handle DSPy calls - prompt may be in kwargs"""
        import sys
        from pathlib import Path
        
        debug_file = Path(__file__).parent.parent / "ollama_debug.log"
        def debug_log(msg):
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")
            print(msg, file=sys.stderr)
        
        debug_log(f"[DEBUG] __call__ invoked with prompt={prompt is not None}, kwargs keys: {list(kwargs.keys())}")
        
        if prompt is None:
            prompt = kwargs.pop("prompt", "")
        if not prompt:
            # Try to get from messages if available
            messages = kwargs.get("messages", [])
            if messages:
                prompt = messages[-1].get("content", "")
                debug_log(f"[DEBUG] Extracted prompt from messages: {prompt[:200]}")
        
        if not prompt:
            debug_log(f"[DEBUG] WARNING: No prompt found in __call__")
            return ""
        
        result = self.basic_request(prompt, **kwargs)
        
        # CRITICAL: Strip markdown code blocks if present (```json ... ```)
        # DSPy's JSONAdapter expects raw JSON, not markdown-wrapped JSON
        import re
        if isinstance(result, str):
            # Remove markdown code blocks (```json ... ``` or ``` ... ```)
            result = re.sub(r'^```(?:json)?\s*\n', '', result, flags=re.MULTILINE)
            result = re.sub(r'\n```\s*$', '', result, flags=re.MULTILINE)
            result = result.strip()
        
        # Safety check: ensure result is never just "m" or a single backtick
        result_stripped = result.strip()
        if len(result_stripped) <= 1 or result_stripped == 'm' or result_stripped == 'M' or result_stripped == '`':
            debug_log(f"[DEBUG] __call__: Result is '{result_stripped}' - using fallback")
            result = '{"reasoning": "Unable to get response - using fallback", "requires_sql": false, "requires_rag": true}'
        
        debug_log(f"[DEBUG] __call__ returning result length: {len(result)}")
        return result
    
    def request(self, prompt: str, **kwargs) -> list:
        """DSPy-compatible request method"""
        import sys
        from pathlib import Path
        import traceback
        
        # Simple debug that writes to both file and stdout
        debug_file = Path(__file__).parent.parent / "ollama_debug.log"
        try:
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[DEBUG] request() called\n")
                f.write(f"Prompt length: {len(prompt)}\n")
                f.write(f"Prompt: {prompt[:500]}\n")
                f.write(f"Kwargs: {kwargs}\n")
        except Exception as e:
            print(f"DEBUG FILE ERROR: {e}", file=sys.stderr)
        
        print(f"[DEBUG REQUEST] Called with prompt length: {len(prompt)}", file=sys.stderr, flush=True)
        
        try:
            response = self.basic_request(prompt, **kwargs)
        except Exception as e:
            error_msg = f"ERROR in basic_request: {e}\n{traceback.format_exc()}"
            try:
                with open(debug_file, "a", encoding="utf-8") as f:
                    f.write(f"{error_msg}\n")
            except:
                pass
            print(error_msg, file=sys.stderr, flush=True)
            response = f"Error: {str(e)}"
        
        # Debug: print response
        try:
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(f"Response length: {len(response)}\n")
                f.write(f"Response: {response[:1000]}\n")
        except:
            pass
        
        print(f"[DEBUG REQUEST] Response length: {len(response)}", file=sys.stderr, flush=True)
        if len(response) < 10:
            print(f"[DEBUG REQUEST] WARNING: Short response: '{response}'", file=sys.stderr, flush=True)
        else:
            print(f"[DEBUG REQUEST] Response preview: {response[:200]}", file=sys.stderr, flush=True)
        
        # Ensure response is a string and not empty
        if not isinstance(response, str):
            response = str(response)
        
        # Critical check: if response is suspiciously short, log everything
        if len(response) <= 1:
            print(f"[DEBUG REQUEST] CRITICAL: Response is only '{response}' (length: {len(response)})", file=sys.stderr, flush=True)
            print(f"[DEBUG REQUEST] This suggests a major issue with Ollama or response extraction", file=sys.stderr, flush=True)
            # Try to get response again as a last resort
            try:
                retry_response = self.basic_request(prompt, **kwargs)
                if len(retry_response) > len(response):
                    print(f"[DEBUG REQUEST] Retry got longer response: {len(retry_response)} chars", file=sys.stderr, flush=True)
                    response = retry_response
            except:
                pass
        
        # CRITICAL: Ensure response is a proper string before creating result
        if not isinstance(response, str):
            response = str(response)
        
        # Remove any leading/trailing whitespace that might cause issues
        response = response.strip()
        
        # CRITICAL: Strip markdown code blocks if present (```json ... ```)
        # DSPy's JSONAdapter expects raw JSON, not markdown-wrapped JSON
        import re
        original_length = len(response)
        # Remove markdown code blocks (```json ... ``` or ``` ... ```)
        # Handle both single-line and multi-line code blocks
        response = re.sub(r'^```(?:json)?\s*\n?', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n?```\s*$', '', response, flags=re.MULTILINE)
        response = response.strip()
        
        if original_length != len(response):
            print(f"[DEBUG REQUEST] Stripped markdown code blocks: {original_length} -> {len(response)} chars", file=sys.stderr, flush=True)
        
        # CRITICAL DEBUG: Log the exact response we got
        print(f"[DEBUG REQUEST] Raw response from basic_request (after stripping): {repr(response[:200])}", file=sys.stderr, flush=True)
        print(f"[DEBUG REQUEST] Response length: {len(response)}", file=sys.stderr, flush=True)
        print(f"[DEBUG REQUEST] Response type: {type(response)}", file=sys.stderr, flush=True)
        
        # Final check: if response is still suspiciously short, try multiple recovery strategies
        # Also check if response is just "m" (the error we're seeing)
        response_stripped = response.strip()
        if len(response) <= 1 or response_stripped == 'm' or response_stripped == 'M':
            print(f"[DEBUG REQUEST] CRITICAL ERROR: Response is only '{response}' (length: {len(response)})!", file=sys.stderr, flush=True)
            print(f"[DEBUG REQUEST] Attempting multiple recovery strategies...", file=sys.stderr, flush=True)
            
            # Strategy 1: Emergency retry with a simple prompt
            try:
                emergency_response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    options={"temperature": 0.7, "num_predict": 100}
                )
                if isinstance(emergency_response, dict):
                    emergency_content = emergency_response.get("message", {}).get("content", "")
                    if len(emergency_content) > len(response):
                        print(f"[DEBUG REQUEST] Strategy 1 (emergency retry) got longer response: {len(emergency_content)} chars", file=sys.stderr, flush=True)
                        response = emergency_content
            except Exception as emergency_error:
                print(f"[DEBUG REQUEST] Strategy 1 failed: {emergency_error}", file=sys.stderr, flush=True)
            
            # Strategy 2: Try generate API instead of chat
            response_stripped_strat2 = response.strip()
            if len(response) <= 1 or response_stripped_strat2 == 'm' or response_stripped_strat2 == 'M':
                try:
                    print(f"[DEBUG REQUEST] Strategy 2: Trying generate API...", file=sys.stderr, flush=True)
                    gen_response = self.client.generate(
                        model=self.model,
                        prompt=prompt,
                        options={"temperature": 0.7, "num_predict": max_tokens}
                    )
                    if isinstance(gen_response, dict):
                        gen_content = gen_response.get("response", "") or gen_response.get("text", "")
                        if len(gen_content) > len(response):
                            print(f"[DEBUG REQUEST] Strategy 2 (generate API) got longer response: {len(gen_content)} chars", file=sys.stderr, flush=True)
                            response = gen_content
                except Exception as gen_error:
                    print(f"[DEBUG REQUEST] Strategy 2 failed: {gen_error}", file=sys.stderr, flush=True)
            
                    # Strategy 3: If still short or just "m" or "`", return a valid JSON error that DSPy can parse
                    response_stripped_check = response.strip()
                    if len(response) <= 1 or response_stripped_check == 'm' or response_stripped_check == 'M' or response_stripped_check == '`':
                print(f"[DEBUG REQUEST] Strategy 3: Response is '{response_stripped_check}' - using fallback JSON", file=sys.stderr, flush=True)
                # Return a valid JSON response that matches the expected format
                # This is a fallback to prevent DSPy from failing completely
                # IMPORTANT: This must be valid JSON that DSPy can parse for QueryRouter signature
                response = '{"reasoning": "Unable to get response from language model - using fallback", "requires_sql": false, "requires_rag": true}'
                print(f"[DEBUG REQUEST] Strategy 3 fallback response: {repr(response)}", file=sys.stderr, flush=True)
        
        # Additional safety check: if response is still suspicious after all strategies, use fallback
        final_check = response.strip()
        if len(final_check) <= 1 or final_check == 'm' or final_check == 'M':
            print(f"[DEBUG REQUEST] Final safety check: Response still '{final_check}' - forcing fallback", file=sys.stderr, flush=True)
            response = '{"reasoning": "Unable to get response from language model - using fallback", "requires_sql": false, "requires_rag": true}'
        
        # Create result in the exact format DSPy expects: list of dicts with "content" key
        result = [{"content": response}]
        
        # Verify structure before returning
        if not isinstance(result, list):
            print(f"[DEBUG REQUEST] ERROR: result is not a list: {type(result)}", file=sys.stderr, flush=True)
            result = [{"content": str(response)}]
        elif len(result) == 0:
            print(f"[DEBUG REQUEST] ERROR: result list is empty", file=sys.stderr, flush=True)
            result = [{"content": str(response)}]
        elif not isinstance(result[0], dict) or "content" not in result[0]:
            print(f"[DEBUG REQUEST] ERROR: Invalid result[0] structure: {result[0] if result else 'empty'}", file=sys.stderr, flush=True)
            result = [{"content": str(response)}]
        
        # Final logging - verify what we're returning
        content_in_result = result[0].get("content", "")
        
        # CRITICAL: Final check on the actual content in result - ensure it's never just "m"
        content_stripped_final = content_in_result.strip()
        if len(content_stripped_final) <= 1 or content_stripped_final == 'm' or content_stripped_final == 'M':
            print(f"[DEBUG REQUEST] CRITICAL: Content in result is '{content_stripped_final}' - REPLACING with fallback", file=sys.stderr, flush=True)
            result[0]["content"] = '{"reasoning": "Unable to get response from language model - using fallback", "requires_sql": false, "requires_rag": true}'
            content_in_result = result[0]["content"]
        
        print(f"[DEBUG REQUEST] ===== FINAL RESULT BEFORE RETURN =====", file=sys.stderr, flush=True)
        print(f"[DEBUG REQUEST] Result type: {type(result)}", file=sys.stderr, flush=True)
        print(f"[DEBUG REQUEST] Result length: {len(result)}", file=sys.stderr, flush=True)
        print(f"[DEBUG REQUEST] Result[0] type: {type(result[0])}", file=sys.stderr, flush=True)
        print(f"[DEBUG REQUEST] Result[0] keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'N/A'}", file=sys.stderr, flush=True)
        print(f"[DEBUG REQUEST] Content in result[0]['content']: length={len(content_in_result)}", file=sys.stderr, flush=True)
        print(f"[DEBUG REQUEST] First 500 chars: {repr(content_in_result[:500])}", file=sys.stderr, flush=True)
        print(f"[DEBUG REQUEST] ======================================", file=sys.stderr, flush=True)
        
        try:
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"FINAL RESULT BEFORE RETURN\n")
                f.write(f"Result type: {type(result)}\n")
                f.write(f"Result length: {len(result)}\n")
                f.write(f"Content length: {len(content_in_result)} chars\n")
                f.write(f"Content: {content_in_result[:1000]}\n")
                f.write(f"{'='*80}\n")
        except Exception as log_error:
            print(f"[DEBUG REQUEST] Could not write to log: {log_error}", file=sys.stderr, flush=True)
        
        return result


# DSPy Signatures
class QueryRouter(dspy.Signature):
    """Route query to appropriate handler (RAG, SQL, or both)
    
    IMPORTANT: Respond with valid JSON format:
    {
      "requires_sql": true/false,
      "requires_rag": true/false,
      "reasoning": "brief explanation"
    }
    """
    question: str = dspy.InputField(desc="User's retail analytics question")
    requires_sql: bool = dspy.OutputField(desc="Whether SQL query is needed (true/false)")
    requires_rag: bool = dspy.OutputField(desc="Whether document search is needed (true/false)")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for routing decision")


class SQLQueryGenerator(dspy.Signature):
    """Generate SQL query from natural language question"""
    question: str = dspy.InputField(desc="User's question")
    schema: str = dspy.InputField(desc="Database schema information")
    sql_query: str = dspy.OutputField(desc="Valid SQL query to answer the question. IMPORTANT: The table 'Order Details' has a space and must be quoted as \"Order Details\" in SQL statements. Prefer joins: Orders + \"Order Details\" + Products. Revenue formula: SUM(UnitPrice * Quantity * (1 - Discount)) from \"Order Details\".")


class ConstraintPlanner(dspy.Signature):
    """Extract constraints from question and documents (dates, KPIs, categories, entities)"""
    question: str = dspy.InputField(desc="User's question")
    rag_context: str = dspy.InputField(desc="Context from documents")
    date_ranges: str = dspy.OutputField(desc="Extracted date ranges (e.g., '1997-06-01 to 1997-06-30')")
    kpi_formulas: str = dspy.OutputField(desc="Extracted KPI formulas or definitions")
    categories: str = dspy.OutputField(desc="Extracted product categories or entities")
    other_constraints: str = dspy.OutputField(desc="Other constraints or requirements")


class AnswerSynthesizer(dspy.Signature):
    """Synthesize final answer from RAG context and SQL results"""
    question: str = dspy.InputField(desc="Original question")
    rag_context: str = dspy.InputField(desc="Context from documents")
    sql_results: str = dspy.InputField(desc="Results from SQL query")
    format_hint: str = dspy.InputField(desc="Expected output format (e.g., 'int', 'float', '{category:str, quantity:int}')")
    answer: str = dspy.OutputField(desc="Answer matching format_hint exactly, with citations")


class SQLRepair(dspy.Signature):
    """Repair a SQL query that failed to execute"""
    original_query: str = dspy.InputField(desc="The SQL query that failed")
    error_message: str = dspy.InputField(desc="Error message from database")
    schema: str = dspy.InputField(desc="Database schema information")
    repaired_query: str = dspy.OutputField(desc="Corrected SQL query")


# DSPy Modules
class QueryRoutingModule(dspy.Module):
    """DSPy module for query routing"""
    
    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(QueryRouter)
    
    def forward(self, question: str) -> QueryIntent:
        """Route query to determine intent"""
        result = self.router(question=question)
        
        # Handle boolean conversion (DSPy might return strings)
        def to_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'requires_sql', 'requires_rag')
            return bool(value)
        
        requires_sql = to_bool(getattr(result, 'requires_sql', False))
        requires_rag = to_bool(getattr(result, 'requires_rag', True))
        
        # Default: if question mentions data/numbers/query, likely needs SQL
        if not requires_sql and not requires_rag:
            sql_keywords = ['show', 'find', 'list', 'count', 'total', 'sum', 'average', 'top', 'best', 'worst']
            if any(kw in question.lower() for kw in sql_keywords):
                requires_sql = True
            else:
                requires_rag = True
        
        return QueryIntent(
            requires_sql=requires_sql,
            requires_rag=requires_rag,
            sql_tables=[],
            keywords=[]
        )
    
    def __call__(self, question: str) -> QueryIntent:
        """Allow calling module directly"""
        return self.forward(question)


class SQLGenerationModule(dspy.Module):
    """DSPy module for SQL generation"""
    
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(SQLQueryGenerator)
    
    def forward(self, question: str, schema: str) -> str:
        """Generate SQL query"""
        result = self.generator(question=question, schema=schema)
        return result.sql_query if hasattr(result, 'sql_query') else ""


class SQLRepairModule(dspy.Module):
    """DSPy module for SQL query repair"""
    
    def __init__(self):
        super().__init__()
        self.repair = dspy.ChainOfThought(SQLRepair)
    
    def forward(self, original_query: str, error_message: str, schema: str) -> str:
        """Repair a failed SQL query"""
        result = self.repair(
            original_query=original_query,
            error_message=error_message,
            schema=schema
        )
        return result.repaired_query if hasattr(result, 'repaired_query') else original_query


class ConstraintPlannerModule(dspy.Module):
    """DSPy module for constraint extraction"""
    
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(ConstraintPlanner)
    
    def forward(self, question: str, rag_context: str) -> dict:
        """Extract constraints from question and context"""
        result = self.planner(
            question=question,
            rag_context=rag_context or "No document context available"
        )
        return {
            "date_ranges": getattr(result, 'date_ranges', ''),
            "kpi_formulas": getattr(result, 'kpi_formulas', ''),
            "categories": getattr(result, 'categories', ''),
            "other_constraints": getattr(result, 'other_constraints', '')
        }


class AnswerSynthesisModule(dspy.Module):
    """DSPy module for answer synthesis"""
    
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(AnswerSynthesizer)
    
    def forward(self, question: str, rag_context: str, sql_results: str, format_hint: str = "") -> str:
        """Synthesize final answer matching format_hint"""
        result = self.synthesizer(
            question=question,
            rag_context=rag_context or "No document context available",
            sql_results=sql_results or "No SQL results available",
            format_hint=format_hint or "text"
        )
        return result.answer if hasattr(result, 'answer') else "Unable to generate answer"

