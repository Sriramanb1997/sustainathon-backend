#!/usr/bin/env python3
"""
Test script for the multi-provider model setup.
Run this to verify your configuration is working correctly.
"""

import os
import sys
import requests
import json
from pathlib import Path

def test_import():
    """Test if model_providers can be imported."""
    try:
        from model_providers import model_manager
        print("✅ Model providers imported successfully")
        return model_manager
    except ImportError as e:
        print(f"❌ Failed to import model providers: {e}")
        print("Make sure you've installed all dependencies: pip install -r requirements.txt")
        return None

def test_health_check(model_manager):
    """Test model provider health."""
    try:
        health_status = model_manager.check_health()
        print(f"🏥 Health check results: {health_status}")
        
        if any(health_status.values()):
            print("✅ At least one provider is healthy")
            return True
        else:
            print("❌ No providers are healthy")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_simple_request(model_manager):
    """Test a simple model request."""
    try:
        print("🧪 Testing simple model request...")
        response = model_manager.generate_response("Hello, reply with just 'Hi there!'", stream=False)
        
        if hasattr(response, 'json'):
            # Ollama response
            content = response.json().get('response', '').strip()
        else:
            # OpenAI response or string
            content = str(response).strip()
        
        print(f"📝 Model response: {content}")
        
        if content and len(content) > 0:
            print("✅ Simple request test passed")
            return True
        else:
            print("❌ Simple request test failed - empty response")
            return False
            
    except Exception as e:
        print(f"❌ Simple request test failed: {e}")
        return False

def test_api_endpoint():
    """Test the actual API endpoint if server is running."""
    try:
        print("🌐 Testing API endpoint...")
        response = requests.get("http://localhost:5000/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 API health check: {data}")
            print("✅ API endpoint test passed")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running (this is OK if you haven't started it yet)")
        print("   To start: python app_ollama.py")
        return None
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        return False

def check_configuration():
    """Check environment configuration."""
    print("⚙️  Checking configuration...")
    
    provider = os.getenv('MODEL_PROVIDER', 'ollama')
    print(f"   Primary provider: {provider}")
    
    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            print(f"   OpenAI API key: {masked_key}")
        else:
            print("   ❌ OpenAI API key not set")
            return False
    
    fallback = os.getenv('ENABLE_FALLBACK', 'true')
    print(f"   Fallback enabled: {fallback}")
    
    print("✅ Configuration check passed")
    return True

def main():
    """Run all tests."""
    print("🚀 BioSphere Model Provider Test Suite")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path('app_ollama.py').exists():
        print("❌ Error: app_ollama.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    tests_passed = 0
    total_tests = 0
    
    # Configuration check
    total_tests += 1
    if check_configuration():
        tests_passed += 1
    
    # Import test
    total_tests += 1
    model_manager = test_import()
    if model_manager:
        tests_passed += 1
    else:
        print("\n❌ Cannot continue without model_manager")
        sys.exit(1)
    
    # Health check test
    total_tests += 1
    if test_health_check(model_manager):
        tests_passed += 1
    
    # Simple request test
    total_tests += 1
    if test_simple_request(model_manager):
        tests_passed += 1
    
    # API endpoint test (optional)
    api_result = test_api_endpoint()
    if api_result is not None:
        total_tests += 1
        if api_result:
            tests_passed += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your setup is working correctly.")
        print("\n📋 Next steps:")
        print("1. Start the server: python app_ollama.py")
        print("2. Test with frontend or Postman")
        print("3. Check the API documentation in README.md")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure dependencies are installed: pip install -r requirements.txt")
        print("2. Check your .env configuration")
        print("3. For Ollama: ensure Ollama is running (ollama serve)")
        print("4. For OpenAI: verify your API key is correct")
        print("5. Check MODEL_CONFIGURATION.md for detailed setup instructions")

if __name__ == "__main__":
    main()
