"""
Model provider abstraction layer for supporting multiple AI providers.
Supports both local models (Ollama) and cloud APIs (OpenAI).
"""

import os
import json
import time
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, Optional, List
from openai import OpenAI

# Configuration
MODEL_PROVIDER = os.getenv('MODEL_PROVIDER', 'ollama').lower()
ENABLE_FALLBACK = os.getenv('ENABLE_FALLBACK', 'true').lower() == 'true'

# Ollama configuration
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', "mistral:7b")
OLLAMA_MAX_RETRIES = int(os.getenv('OLLAMA_MAX_RETRIES', '3'))
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '30'))
OLLAMA_RETRY_DELAY = int(os.getenv('OLLAMA_RETRY_DELAY', '2'))

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '1500'))
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))

class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, stream: bool = False) -> Any:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def check_health(self) -> bool:
        """Check if the provider is available."""
        pass
    
    @abstractmethod
    def get_fallback_response(self, question_type: str) -> str:
        """Get a fallback response when the provider is unavailable."""
        pass

class OllamaProvider(ModelProvider):
    """Provider for local Ollama models."""
    
    def __init__(self):
        self.api_url = OLLAMA_API_URL
        self.model = OLLAMA_MODEL
        self.max_retries = OLLAMA_MAX_RETRIES
        self.timeout = OLLAMA_TIMEOUT
        self.retry_delay = OLLAMA_RETRY_DELAY
    
    def generate_response(self, prompt: str, stream: bool = False) -> Any:
        """Generate response using Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        
        return self._make_request(payload, stream)
    
    def _make_request(self, payload: Dict[str, Any], stream: bool = False) -> Any:
        """Make request to Ollama with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logging.info(f"Attempting Ollama request (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    stream=stream,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    logging.info(f"Ollama request successful on attempt {attempt + 1}")
                    return response
                else:
                    logging.warning(f"Ollama request failed with status {response.status_code}: {response.text}")
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise requests.exceptions.RequestException(f"HTTP {response.status_code}: {response.text}")
                        
            except requests.exceptions.RequestException as e:
                last_exception = e
                logging.warning(f"Ollama request error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
        
        # All retries exhausted
        error_msg = f"Ollama service unavailable after {self.max_retries + 1} attempts"
        if last_exception:
            error_msg += f": {last_exception}"
        
        logging.error(error_msg)
        raise requests.exceptions.RequestException(error_msg)
    
    def check_health(self) -> bool:
        """Check Ollama health."""
        try:
            version_url = self.api_url.replace('/api/generate', '/api/version')
            response = requests.get(version_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_fallback_response(self, question_type: str) -> str:
        """Get fallback response for Ollama."""
        fallback_responses = {
            'greeting': "Hello! I'm your wildlife and conservation assistant for India. I'm currently experiencing some technical difficulties, but I'm here to help you learn about India's amazing biodiversity, national parks, wildlife sanctuaries, and conservation efforts. How can I assist you today?",
            'gratitude': "You're very welcome! I'm glad I could help you learn about wildlife and conservation in India. If you have any more questions about India's biodiversity, national parks, or conservation efforts, feel free to ask!",
            'capability': "I'm an AI assistant specialized in wildlife, biodiversity, conservation, and sanctuary information specifically for India. I can help you with information about Indian national parks, wildlife sanctuaries, endangered species, conservation organizations, and environmental protection efforts. I'm currently experiencing some connectivity issues, but I'm still here to assist you with your wildlife and conservation questions about India!",
            'wildlife_technical': "I apologize, but I'm currently experiencing technical difficulties accessing my knowledge base. However, I'm designed to help with wildlife, biodiversity, and conservation topics in India. Please try your question again in a moment, or rephrase it, and I'll do my best to provide you with accurate information about India's wildlife and conservation efforts.",
            'conservation_technical': "I'm currently experiencing some technical issues, but I specialize in conservation and environmental topics in India. Please try your question again shortly, and I'll help you with information about conservation projects, protected areas, and environmental initiatives in India.",
            'general_environmental': "I'm temporarily experiencing technical difficulties, but I'm here to help with environmental and wildlife topics related to India. Please try asking your question again in a moment."
        }
        
        return fallback_responses.get(question_type, fallback_responses['capability'])

class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.max_tokens = OPENAI_MAX_TOKENS
        self.temperature = OPENAI_TEMPERATURE
    
    def generate_response(self, prompt: str, stream: bool = False) -> Any:
        """Generate response using OpenAI."""
        try:
            # Convert single prompt to messages format
            messages = self._convert_prompt_to_messages(prompt)
            
            if stream:
                return self._generate_streaming_response(messages)
            else:
                return self._generate_non_streaming_response(messages)
                
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise
    
    def _convert_prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Convert Ollama-style prompt to OpenAI messages format."""
        # Split the prompt into system and user parts
        lines = prompt.split('\n')
        messages = []
        current_role = None
        current_content = []
        
        for line in lines:
            if line.startswith('System:') or line.startswith('[General Role]'):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                current_role = "system"
                current_content = [line.replace('System:', '').strip()]
            elif line.startswith('User:'):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                current_role = "user"
                current_content = [line.replace('User:', '').strip()]
            elif line.startswith('Assistant:'):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                current_role = "assistant"
                current_content = [line.replace('Assistant:', '').strip()]
            else:
                if current_content or line.strip():
                    current_content.append(line)
        
        # Add the last message
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": '\n'.join(current_content).strip()
            })
        
        # Ensure we have at least a system message and user message
        if not messages or messages[0]["role"] != "system":
            # Add default system message if none exists
            default_system = """You are an AI assistant focused on wildlife, biodiversity, conservation, sanctuaries, and related topics in India. 

Use bullet points or numbered lists when possible. Use bold formatting for project names, initiative titles, and key headings in bullet points. 

Only answer questions related to wildlife, biodiversity, conservation, sanctuaries, and similar topics in India. If a question is about wildlife in other countries, politely respond that you can only answer questions about India unless explicitly asked about other countries.

When mentioning NGOs or organizations, provide information concisely without repetition. Never mention the same NGO or organization twice in your response."""
            
            messages.insert(0, {"role": "system", "content": default_system})
        
        # Ensure the last message is from user
        if not messages or messages[-1]["role"] != "user":
            # Extract the last user query from the prompt
            if "Assistant:" in prompt:
                # This is likely a conversation, get the last user input
                user_parts = prompt.split("User:")
                if len(user_parts) > 1:
                    last_user_msg = user_parts[-1].split("Assistant:")[0].strip()
                    messages.append({"role": "user", "content": last_user_msg})
        
        return messages
    
    def _generate_streaming_response(self, messages: List[Dict[str, str]]):
        """Generate streaming response from OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True
        )
        
        return response
    
    def _generate_non_streaming_response(self, messages: List[Dict[str, str]]):
        """Generate non-streaming response from OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False
        )
        
        # Create a mock response object similar to requests.Response
        class MockResponse:
            def __init__(self, content):
                self.status_code = 200
                self._content = content
            
            def json(self):
                return {"response": self._content}
        
        return MockResponse(response.choices[0].message.content)
    
    def check_health(self) -> bool:
        """Check OpenAI API health."""
        try:
            # Make a simple API call to check connectivity
            self.client.models.list()
            return True
        except Exception as e:
            logging.error(f"OpenAI health check failed: {e}")
            return False
    
    def get_fallback_response(self, question_type: str) -> str:
        """Get fallback response for OpenAI."""
        fallback_responses = {
            'greeting': "Hello! I'm your wildlife and conservation assistant for India. I'm currently experiencing some connectivity issues with my AI service, but I'm here to help you learn about India's amazing biodiversity, national parks, wildlife sanctuaries, and conservation efforts. How can I assist you today?",
            'gratitude': "You're very welcome! I'm glad I could help you learn about wildlife and conservation in India. If you have any more questions about India's biodiversity, national parks, or conservation efforts, feel free to ask!",
            'capability': "I'm an AI assistant specialized in wildlife, biodiversity, conservation, and sanctuary information specifically for India. I can help you with information about Indian national parks, wildlife sanctuaries, endangered species, conservation organizations, and environmental protection efforts. I'm currently experiencing some connectivity issues with my AI service, but I'm still here to assist you with your wildlife and conservation questions about India!",
            'wildlife_technical': "I apologize, but I'm currently experiencing technical difficulties with my AI service. However, I'm designed to help with wildlife, biodiversity, and conservation topics in India. Please try your question again in a moment, or rephrase it, and I'll do my best to provide you with accurate information about India's wildlife and conservation efforts.",
            'conservation_technical': "I'm currently experiencing some issues with my AI service, but I specialize in conservation and environmental topics in India. Please try your question again shortly, and I'll help you with information about conservation projects, protected areas, and environmental initiatives in India.",
            'general_environmental': "I'm temporarily experiencing technical difficulties with my AI service, but I'm here to help with environmental and wildlife topics related to India. Please try asking your question again in a moment."
        }
        
        return fallback_responses.get(question_type, fallback_responses['capability'])

class ModelManager:
    """Manages multiple model providers with fallback support."""
    
    def __init__(self):
        self.providers = {}
        self.primary_provider = None
        self.fallback_provider = None
        
        # Initialize providers based on configuration
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers."""
        # Try to initialize Ollama provider
        try:
            ollama = OllamaProvider()
            self.providers['ollama'] = ollama
            logging.info("Ollama provider initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize Ollama provider: {e}")
        
        # Try to initialize OpenAI provider
        try:
            if OPENAI_API_KEY:
                openai = OpenAIProvider()
                self.providers['openai'] = openai
                logging.info("OpenAI provider initialized successfully")
            else:
                logging.info("OpenAI API key not provided, skipping OpenAI provider initialization")
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Set primary and fallback providers
        self._set_providers()
    
    def _set_providers(self):
        """Set primary and fallback providers based on configuration."""
        if MODEL_PROVIDER in self.providers:
            self.primary_provider = self.providers[MODEL_PROVIDER]
            logging.info(f"Primary provider set to: {MODEL_PROVIDER}")
        else:
            # If preferred provider is not available, use the first available
            if self.providers:
                provider_name = list(self.providers.keys())[0]
                self.primary_provider = self.providers[provider_name]
                logging.warning(f"Preferred provider '{MODEL_PROVIDER}' not available, using: {provider_name}")
            else:
                logging.error("No model providers available!")
                return
        
        # Set fallback provider if enabled
        if ENABLE_FALLBACK and len(self.providers) > 1:
            for name, provider in self.providers.items():
                if provider != self.primary_provider:
                    self.fallback_provider = provider
                    logging.info(f"Fallback provider set to: {name}")
                    break
    
    def generate_response(self, prompt: str, stream: bool = False) -> Any:
        """Generate response using primary provider with fallback."""
        if not self.primary_provider:
            raise RuntimeError("No model providers available")
        
        try:
            return self.primary_provider.generate_response(prompt, stream)
        except Exception as e:
            logging.warning(f"Primary provider failed: {e}")
            
            if self.fallback_provider and ENABLE_FALLBACK:
                logging.info("Attempting fallback provider")
                try:
                    return self.fallback_provider.generate_response(prompt, stream)
                except Exception as fallback_error:
                    logging.error(f"Fallback provider also failed: {fallback_error}")
                    raise e  # Raise original error
            else:
                raise e
    
    def check_health(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health_status = {}
        for name, provider in self.providers.items():
            health_status[name] = provider.check_health()
        return health_status
    
    def get_fallback_response(self, question_type: str) -> str:
        """Get fallback response from available provider."""
        if self.primary_provider:
            return self.primary_provider.get_fallback_response(question_type)
        elif self.fallback_provider:
            return self.fallback_provider.get_fallback_response(question_type)
        else:
            return "I apologize, but I'm currently experiencing technical difficulties. Please try again later."

# Global model manager instance
model_manager = ModelManager()
