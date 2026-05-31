"""
AI Client Factory for Google Colab
Provides unified interface with OpenRouter model rotation and Gemini SDK support
"""
import os
from openai import OpenAI

# List of highly reliable free models on OpenRouter
OPENROUTER_FREE_MODELS = [
    "google/gemini-2.5-flash:free",
    "meta-llama/llama-3-8b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "openchat/openchat-7b:free"
]


class OpenRouterAdapter:
    """Wraps OpenAI SDK to query OpenRouter and rotate free models on failure"""
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.api_key = api_key
        self.base_url = base_url

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model: str, messages: list, temperature: float = 1.0, **kwargs):
        # Build model fallback list
        models_to_try = []
        
        # Determine model
        req_model = model if model else "openrouter/free"
        if req_model == "auto":
            req_model = "openrouter/free"
            
        models_to_try.append(req_model)
        
        # Ensure openrouter/free is in there
        if "openrouter/free" not in models_to_try:
            models_to_try.append("openrouter/free")
            
        # Add individual free models as fallbacks
        for free_m in OPENROUTER_FREE_MODELS:
            if free_m not in models_to_try:
                models_to_try.append(free_m)

        last_exception = None
        for current_model in models_to_try:
            try:
                print(f"  [OpenRouter] Querying model: {current_model}")
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs
                )
                return response
            except Exception as e:
                print(f"  [OpenRouter] Model {current_model} failed: {e}. Trying next fallback...")
                last_exception = e
                continue
                
        raise last_exception if last_exception else Exception("All OpenRouter models failed")


class GeminiAdapter:
    """Wraps Google GenAI SDK to conform with standard OpenAI chat interfaces"""
    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model: str, messages: list, temperature: float = 1.0, **kwargs):
        from google.genai import types
        
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)]
                    )
                )
            elif role in ("assistant", "model"):
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)]
                    )
                )

        model_name = model if model and model != "auto" else "gemini-2.5-flash"
        
        config_kwargs = {}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
            
        config = types.GenerateContentConfig(**config_kwargs)
        
        print(f"  [Gemini] Requesting model: {model_name}")
        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )
        
        from types import SimpleNamespace
        message = SimpleNamespace(content=response.text)
        choice = SimpleNamespace(message=message)
        
        # Parse usage stats
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        try:
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage.prompt_tokens = response.usage_metadata.prompt_token_count
                usage.completion_tokens = response.usage_metadata.candidates_token_count
        except Exception:
            pass
            
        return SimpleNamespace(choices=[choice], usage=usage)


def create_ai_client(provider: str, api_key: str, base_url: str = None):
    """Factory to return unified client interface"""
    provider = provider.lower()
    
    if provider == "openrouter":
        return OpenRouterAdapter(api_key, base_url or "https://openrouter.ai/api/v1")
    elif provider in ("google", "gemini"):
        return GeminiAdapter(api_key)
    elif provider == "groq":
        return OpenAI(api_key=api_key, base_url=base_url or "https://api.groq.com/openai/v1")
    elif provider == "openai":
        return OpenAI(api_key=api_key, base_url=base_url or "https://api.openai.com/v1")
    else:
        # Fallback standard OpenAI compatible endpoint
        return OpenAI(api_key=api_key, base_url=base_url or "https://api.openai.com/v1")
