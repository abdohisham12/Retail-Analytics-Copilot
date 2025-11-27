"""Test if we can create and read debug files"""
from pathlib import Path
import sys

# Test 1: Create a simple test file
test_file = Path("test_output.txt")
with open(test_file, "w", encoding="utf-8") as f:
    f.write("Test output\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Working directory: {Path.cwd()}\n")
print(f"Created test file: {test_file.absolute()}")

# Test 2: Check debug log path
debug_file = Path("agent/dspy_signatures.py").parent.parent / "ollama_debug.log"
print(f"Debug log path: {debug_file.absolute()}")
print(f"Debug log exists: {debug_file.exists()}")
if debug_file.exists():
    print(f"Debug log size: {debug_file.stat().st_size} bytes")
    with open(debug_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"Debug log has {len(lines)} lines")
        if lines:
            print(f"Last 5 lines:")
            for line in lines[-5:]:
                print(f"  {line.rstrip()}")

# Test 3: Try to call Ollama directly
print("\nTesting Ollama directly...")
try:
    import ollama
    client = ollama.Client()
    resp = client.chat(
        model="phi3.5:3.8b-mini-instruct-q4_K_M",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=False
    )
    print(f"Ollama response type: {type(resp)}")
    if isinstance(resp, dict):
        content = resp.get("message", {}).get("content", "")
        print(f"Content: {repr(content)}")
        print(f"Content length: {len(content)}")
        with open(test_file, "a", encoding="utf-8") as f:
            f.write(f"\nOllama test:\n")
            f.write(f"Response type: {type(resp)}\n")
            f.write(f"Content: {repr(content)}\n")
            f.write(f"Content length: {len(content)}\n")
except Exception as e:
    print(f"Ollama test failed: {e}")
    import traceback
    traceback.print_exc()
    with open(test_file, "a", encoding="utf-8") as f:
        f.write(f"\nOllama test failed: {e}\n")
        f.write(traceback.format_exc())

print(f"\nTest complete. Check {test_file.absolute()}")

