# Model Provider Configuration Guide

BioSphere backend now supports multiple AI model providers, giving you flexibility to use either local models or cloud APIs.

## 🚀 Quick Start

### Option 1: Local Models (Ollama)
```bash
# 1. Install and start Ollama
# Follow instructions at: https://ollama.ai/

# 2. Pull a model (e.g., Mistral)
ollama pull mistral:7b

# 3. Configure environment
echo "MODEL_PROVIDER=ollama" >> .env
echo "OLLAMA_MODEL=mistral:7b" >> .env

# 4. Start the application
python app_ollama.py
```

### Option 2: OpenAI API
```bash
# 1. Get your API key from https://platform.openai.com/api-keys

# 2. Configure environment
echo "MODEL_PROVIDER=openai" >> .env
echo "OPENAI_API_KEY=your_api_key_here" >> .env
echo "OPENAI_MODEL=gpt-3.5-turbo" >> .env

# 3. Start the application
python app_ollama.py
```

## 📋 Environment Variables

### Core Configuration
```bash
# Primary model provider: 'ollama' or 'openai'
MODEL_PROVIDER=ollama

# Enable fallback to secondary provider if primary fails
ENABLE_FALLBACK=true
```

### Ollama Configuration (Local Models)
```bash
OLLAMA_API_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=mistral:7b                    # or llama2, codellama, etc.
OLLAMA_MAX_RETRIES=3
OLLAMA_TIMEOUT=30
OLLAMA_RETRY_DELAY=2
OLLAMA_HEALTH_CHECK_TIMEOUT=15
```

### OpenAI Configuration (Cloud API)
```bash
OPENAI_API_KEY=sk-your-api-key-here       # Get from https://platform.openai.com/api-keys
OPENAI_MODEL=gpt-3.5-turbo                # or gpt-4, gpt-4-turbo
OPENAI_MAX_TOKENS=1500
OPENAI_TEMPERATURE=0.7
```

### Cache and Performance
```bash
CACHE_DURATION=5                          # Cache duration in seconds
MIN_REQUEST_INTERVAL=1                    # Min seconds between requests
FRONTEND_URL=http://localhost:5173        # Frontend URL for CORS
```

## 🔧 Migration from Ollama-Only

If you're upgrading from an Ollama-only setup:

```bash
# 1. Install new dependencies
pip install openai>=1.0.0

# 2. Run migration script
python migrate_to_multi_provider.py

# 3. Configure your .env file
cp .env.example .env
# Edit .env with your preferences
```

## 🏥 Health Monitoring

### Check Service Health
```bash
curl http://localhost:5000/health
```
Response:
```json
{
  "status": "healthy",
  "providers": {
    "ollama": true,
    "openai": false
  },
  "timestamp": "2024-07-23T10:30:00Z",
  "message": "Model services available"
}
```

### Check Model Provider Status
```bash
curl http://localhost:5000/model/status
```
Response:
```json
{
  "providers_health": {
    "ollama": true,
    "openai": false
  },
  "primary_provider": "ollama",
  "fallback_provider": null,
  "fallback_enabled": true,
  "timestamp": "2024-07-23T10:30:00Z"
}
```

## 🎯 Provider Selection Strategy

### Automatic Fallback
When `ENABLE_FALLBACK=true`, the system will:
1. Try the primary provider first
2. If primary fails, automatically switch to secondary provider
3. Log the switch for monitoring

### Manual Provider Testing
Test specific providers:
```python
from model_providers import model_manager

# Check health of all providers
health = model_manager.check_health()
print(health)  # {'ollama': True, 'openai': False}

# Test a specific request
response = model_manager.generate_response("Hello, how are you?")
```

## 💰 Cost Considerations

### Local Models (Ollama)
- ✅ **Pros:** Free to use, privacy-focused, offline capability
- ❌ **Cons:** Requires local compute resources, slower on CPU

### OpenAI API
- ✅ **Pros:** Fast, high-quality responses, no local resources needed
- ❌ **Cons:** Costs money per request, requires internet

### Cost Optimization Tips
1. **Use Ollama as primary, OpenAI as fallback** for development
2. **Use OpenAI as primary, Ollama as fallback** for production with budget
3. **Monitor token usage** with OpenAI API
4. **Cache responses** to reduce API calls

## 🔍 Troubleshooting

### Common Issues

#### "No model providers available"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Check OpenAI API key
echo $OPENAI_API_KEY

# Verify configuration
python -c "from model_providers import model_manager; print(model_manager.check_health())"
```

#### "OpenAI rate limit exceeded"
- Add retry logic (already built-in)
- Consider using `gpt-3.5-turbo` instead of `gpt-4`
- Implement request queuing for high-volume usage

#### "Ollama model not found"
```bash
# List available models
ollama list

# Pull a model
ollama pull mistral:7b

# Update OLLAMA_MODEL in .env
```

## 🛠️ Development Tips

### Adding New Providers
1. Create a new provider class inheriting from `ModelProvider`
2. Implement required methods: `generate_response()`, `check_health()`, `get_fallback_response()`
3. Register the provider in `ModelManager._initialize_providers()`

### Testing Different Models
```python
# Test different OpenAI models
os.environ['OPENAI_MODEL'] = 'gpt-4'
# Restart application

# Test different Ollama models
os.environ['OLLAMA_MODEL'] = 'llama2:13b'
# Restart application
```

## 📊 Performance Benchmarks

| Provider | Model | Avg Response Time | Quality | Cost |
|----------|-------|------------------|---------|------|
| Ollama | mistral:7b | 3-10s | Good | Free |
| Ollama | llama2:13b | 5-15s | Better | Free |
| OpenAI | gpt-3.5-turbo | 1-3s | Excellent | $0.002/1K tokens |
| OpenAI | gpt-4 | 2-8s | Best | $0.03/1K tokens |

*Response times depend on hardware and network conditions*

## 🔗 References

- [Ollama Documentation](https://ollama.ai/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Python OpenAI Library](https://github.com/openai/openai-python)
- [Model Provider Architecture](./model_providers.py)
