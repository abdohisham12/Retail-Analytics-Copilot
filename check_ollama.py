"""Check if Ollama is running and model is available"""
import ollama

try:
    client = ollama.Client()
    models = client.list()
    print("✅ Ollama is running!")
    print(f"\nAvailable models:")
    for model in models.get('models', []):
        print(f"  - {model.get('name', 'unknown')}")
    
    # Check if required model is available
    required_model = "phi3.5:3.8b-mini-instruct-q4_K_M"
    model_names = [m.get('name', '') for m in models.get('models', [])]
    if any(required_model in name for name in model_names):
        print(f"\n✅ Required model '{required_model}' is available!")
    else:
        print(f"\n⚠️  Required model '{required_model}' not found.")
        print(f"   Run: ollama pull {required_model}")
        
except Exception as e:
    print(f"❌ Ollama is not accessible: {e}")
    print("\nTo start Ollama:")
    print("  1. Install Ollama from https://ollama.ai")
    print("  2. Run: ollama serve")
    print("  3. Pull model: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M")

