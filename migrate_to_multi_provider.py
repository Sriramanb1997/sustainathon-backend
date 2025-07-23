#!/usr/bin/env python3
"""
Migration script to update Sustainathon Backend from Ollama-only to multi-provider support.
This script helps migrate your existing setup to support both local models (Ollama) and cloud APIs (OpenAI).
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, check=True):
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error: {result.stderr}")
        return False
    return True

def install_dependencies():
    """Install new dependencies."""
    print("📦 Installing new dependencies...")
    
    # Check if virtual environment is active
    if 'VIRTUAL_ENV' not in os.environ:
        print("⚠️  Warning: No virtual environment detected. Consider activating your virtual environment first.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Install openai package
    if not run_command("pip install openai>=1.0.0"):
        return False
    
    print("✅ Dependencies installed successfully!")
    return True

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        shutil.copy(env_example, env_file)
        print("✅ .env file created! Please edit it with your configuration.")
        return True
    elif not env_file.exists():
        print("⚠️  No .env file found. Please create one using .env.example as a template.")
        return False
    else:
        print("✅ .env file already exists.")
        return True

def backup_original():
    """Create backup of original app.py if it exists."""
    app_py = Path('app.py')
    if app_py.exists():
        backup_name = 'app_original_backup.py'
        print(f"📋 Creating backup: {backup_name}")
        shutil.copy(app_py, backup_name)
        print("✅ Backup created!")

def check_model_providers():
    """Check if model providers are properly configured."""
    print("🔍 Checking model provider configuration...")
    
    try:
        from model_providers import model_manager
        health_status = model_manager.check_health()
        
        print("Model Provider Health Status:")
        for provider, is_healthy in health_status.items():
            status = "✅ Healthy" if is_healthy else "❌ Unavailable"
            print(f"  {provider}: {status}")
        
        if any(health_status.values()):
            print("✅ At least one model provider is available!")
            return True
        else:
            print("⚠️  No model providers are currently available.")
            print("   Make sure either Ollama is running locally or OpenAI API key is configured.")
            return False
            
    except Exception as e:
        print(f"❌ Error checking model providers: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 MIGRATION COMPLETED!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. Configure your preferred model provider in .env:")
    print("   - For Ollama (local): MODEL_PROVIDER=ollama")
    print("   - For OpenAI (cloud): MODEL_PROVIDER=openai")
    print("\n2. Set up your API credentials:")
    print("   - Ollama: Make sure Ollama is running on localhost:11434")
    print("   - OpenAI: Add your API key to OPENAI_API_KEY in .env")
    print("\n3. Test the setup:")
    print("   python app_ollama.py")
    print("   # In another terminal:")
    print("   curl http://localhost:5000/health")
    print("\n4. Available model endpoints:")
    print("   - GET /health - Check overall service health")
    print("   - GET /model/status - Check specific model provider status")
    print("\n5. Environment Variables Reference:")
    print("   MODEL_PROVIDER=ollama|openai    # Choose your primary provider")
    print("   ENABLE_FALLBACK=true            # Enable fallback to secondary provider")
    print("   OPENAI_API_KEY=sk-...           # Your OpenAI API key")
    print("   OPENAI_MODEL=gpt-3.5-turbo     # OpenAI model to use")
    print("\n📖 For more details, check the updated README.md")

def main():
    """Main migration function."""
    print("🚀 Sustainathon Backend Model Provider Migration")
    print("=" * 50)
    print("This script will help you migrate from Ollama-only to multi-provider support.")
    print("You'll be able to use both local models (Ollama) and cloud APIs (OpenAI).")
    print()
    
    # Check if we're in the right directory
    if not Path('app_ollama.py').exists():
        print("❌ Error: app_ollama.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Steps
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Creating environment file", create_env_file),
        ("Checking model providers", check_model_providers),
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"❌ Failed: {step_name}")
            print("Please fix the issues above and run the script again.")
            sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()
