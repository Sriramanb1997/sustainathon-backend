from flask import Flask, session, request, jsonify, url_for, redirect, Response
from flask_cors import CORS
import requests
import os
import uuid
from datetime import datetime
from authlib.integrations.flask_client import OAuth
import json
import time
import logging
import re
from functools import wraps

from utils import retrieve_context
from db import store_chat, get_chat, list_chats, delete_chat, store_user, get_user, delete_user, store_context, retrieve_context, get_user_chats, rename_chat, \
    list_users, store_contexts_from_file, delete_context_collection, get_context_collection_count, list_links, add_link, delete_link, get_structured_context, check_sanctuary_exists
from model_providers import model_manager
    
    
# Simple cache and rate limiting to prevent excessive frontend polling
ENDPOINT_CACHE = {}
CACHE_DURATION = int(os.getenv('CACHE_DURATION', 5))  # seconds
LAST_REQUEST_TIME = {}
MIN_REQUEST_INTERVAL = int(os.getenv('MIN_REQUEST_INTERVAL', 1))  # minimum seconds between requests for same endpoint

FRONTEND_URL = os.getenv('FRONTEND_URL', "http://localhost:5173")  # Update this to your frontend URL if different
LMSTUDIO_CHAT_API_URL = "http://localhost:1234/v1/chat/completions"
MAX_HISTORY = 5  # Configurable conversation history size

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": FRONTEND_URL}}, supports_credentials=True)

# Configure Flask logging
app.logger.setLevel(logging.INFO)
# Disable Flask's default request logging to reduce noise
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

app.secret_key = "strong_key"

# Google OAuth Configuration
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='913121818157-fmkruu2cs5qte6narrg43jc3otcgu74t.apps.googleusercontent.com',
    client_secret='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',
    access_token_url='https://oauth2.googleapis.com/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v2/',
    client_kwargs={'scope': 'openid email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',  # OpenID Connect metadata
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs'  # Manually set the JWKS URI
)

def cache_and_rate_limit(cache_key_func=None):
    """
    Decorator to add caching and rate limiting to endpoints.
    Helps prevent excessive frontend polling.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{f.__name__}_{request.url}"
            
            current_time = time.time()
            
            # Check rate limiting
            if cache_key in LAST_REQUEST_TIME:
                time_since_last = current_time - LAST_REQUEST_TIME[cache_key]
                if time_since_last < MIN_REQUEST_INTERVAL:
                    # Too frequent, check cache
                    if cache_key in ENDPOINT_CACHE:
                        cache_data, cache_time = ENDPOINT_CACHE[cache_key]
                        if current_time - cache_time < CACHE_DURATION:
                            return cache_data
            
            # Update last request time
            LAST_REQUEST_TIME[cache_key] = current_time
            
            # Check cache first
            if cache_key in ENDPOINT_CACHE:
                cache_data, cache_time = ENDPOINT_CACHE[cache_key]
                if current_time - cache_time < CACHE_DURATION:
                    return cache_data
            
            # Execute function and cache result
            result = f(*args, **kwargs)
            ENDPOINT_CACHE[cache_key] = (result, current_time)
            
            return result
        return wrapper
    return decorator

# Default system contexts
system_contexts = [
    {"role": "system", "content": """
[General Role]
You are an AI assistant focused on wildlife, biodiversity, conservation, sanctuaries, and related topics in India.

[Formatting Instructions]
- Use bullet points or numbered lists when possible.
- **Use bold formatting for project names, initiative titles, and key headings** in bullet points (e.g., **Project Name**: Description).
- Double-check your formatting before responding.

[Domain Restrictions]
Only answer questions related to wildlife, biodiversity, conservation, sanctuaries, and similar topics in India. If a question is about wildlife, biodiversity, or conservation in other countries, politely respond that you can only answer questions about India unless the user explicitly asks for information about other countries.

[Question Type Handling]
- GREETING questions: Respond warmly and briefly explain your capabilities without using any provided context.
- GRATITUDE questions: Acknowledge the thanks politely and ask if there's anything else you can help with regarding wildlife and conservation in India.
- CAPABILITY questions: Clearly explain your role and abilities related to wildlife, biodiversity, and conservation in India without using any provided context.
- TECHNICAL questions (wildlife/conservation): Use provided context when available to give detailed, accurate answers. Format longer responses as bullet points for easy reading, using **bold formatting for project names, species names, and key headings**, unless the information is better suited for tables.
- MIXED INTENT questions (gratitude/greeting + technical): Briefly acknowledge the thanks/greeting, then provide a comprehensive answer to the technical question using available context, formatted as bullet points with **bold headings for project names and key topics** or tables as appropriate.
- OFF_TOPIC questions: Politely redirect the conversation back to wildlife, conservation, or environmental topics in India.

[Information Availability Guidelines]
- If asked about a specific wildlife sanctuary, national park, or conservation area that is NOT in your context database, clearly state that you don't have specific information about that particular location.
- Do NOT provide information about different sanctuaries when asked about a specific one that you don't have data for.
- If the specific location is not available, acknowledge this limitation and suggest that the user might want to contact relevant forest departments or conservation organizations for accurate information.
- Only provide information about locations that are actually mentioned in your context when asked about specific places.
- Avoid giving general information about similar places when asked about a specific location that you don't have data for.

[NGO and Organization Information]
- When mentioning NGOs or organizations, provide information concisely without repetition.
- If multiple organizations are provided in the context, mention each one only once.
- Avoid repeating the same organization's information multiple times in your response.
- Combine and summarize information about organizations rather than listing them separately multiple times.
- NEVER mention the same NGO or organization twice in your response, even if it appears multiple times in the context.
- If the same organization appears multiple times with different information, combine all relevant details into a single entry.

[Source Attribution]
Do not include any source links or references in your response. Source attribution will be handled automatically.

[Table Formatting]
- **ALWAYS provide contextual explanation before presenting tables** - Start with a brief description or introduction that explains what the table contains.
- **ALWAYS use tables for NGO/organization information** instead of bullet points or paragraph lists, but introduce the table with relevant context.
- When presenting information about multiple NGOs, sanctuaries, species, or any structured data, first provide a brief explanation, then format it as a clear table.
- Use tables for any comparative or list-based information to improve readability, but always precede tables with descriptive text.
- Tables should have clear headers and be well-organized.
- Structure your response as: [Brief explanation/context] + [Table with data] + [Additional details if relevant].

[Bullet Point Formatting]
- **Use bullet points for most technical answers** that cannot be formatted as tables, to improve readability and organization.
- Structure information clearly with bullet points for projects, activities, conservation efforts, species information, and habitat details.
- Only use direct paragraph text for very short, simple answers.
- Always start with a brief introduction, then use bullet points to organize the main information.

[Directness and Response Style]
- Do not use phrases like 'based on the context' or 'as mentioned in the context.' Just provide the answer directly.
- NEVER echo or repeat the reference material structure in your response.
- Do not include phrases like "Context 1:", "Context 2:", etc. in your answers.
- Do not repeat the detailed formatting from reference materials.
- Extract and synthesize information naturally without referencing how it was provided to you.
- Focus on answering the user's question directly using the information available.
- For greetings and capability questions, do not use any technical context even if provided.
- Avoid redundant or repetitive information about the same organizations.
"""}
]

# Configure logging
import logging
import sys

# Configure logging with explicit settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure logs go to stdout
    ],
    force=True  # Override any existing logging configuration
)

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_model_request(prompt, stream=False):
    """
    Make a request to the configured model provider with proper error handling.
    
    Args:
        prompt: The prompt to send to the model
        stream: Whether to use streaming response
    
    Returns:
        Response object or None if all retries failed
    
    Raises:
        requests.exceptions.RequestException: If all retries are exhausted
    """
    try:
        return model_manager.generate_response(prompt, stream=stream)
    except Exception as e:
        logging.error(f"Model request failed: {e}")
        raise

def check_model_health():
    """
    Check if the configured model provider is available and responsive.
    
    Returns:
        bool: True if primary model is available, False otherwise
    """
    try:
        health_status = model_manager.check_health()
        # Check if primary provider is healthy
        for provider_name, is_healthy in health_status.items():
            if is_healthy:
                return True
        return False
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return False

def get_model_fallback_response(question_type):
    """
    Get a fallback response when the model is unavailable.
    
    Args:
        question_type: The classified type of the question
        
    Returns:
        str: Fallback response message
    """
    return model_manager.get_fallback_response(question_type)
    """
    Get a fallback response when Ollama is unavailable.
    
    Args:
        question_type: The classified type of the question
        
    Returns:
        str: A contextually appropriate fallback message
    """
    fallback_responses = {
        'greeting': "Hello! I'm an AI assistant focused on wildlife, biodiversity, and conservation topics in India. I'm currently experiencing some technical difficulties, but I'll be back to help you soon!",
        'gratitude': "You're welcome! I'm glad I could help. If you have any more questions about wildlife, conservation, or environmental topics in India, feel free to ask. I'm currently experiencing some technical issues, but I'll be back shortly.",
        'capability': "I'm an AI assistant that specializes in wildlife, biodiversity, conservation, and environmental topics specifically related to India. I can provide information about animals, habitats, conservation efforts, protected areas, and environmental protection. However, I'm currently experiencing technical issues. Please try again in a moment.",
        'wildlife_technical': "I'd love to help you with your wildlife-related question, but I'm currently experiencing technical difficulties. Please try again in a few moments, and I'll be happy to provide detailed information about wildlife and biodiversity topics in India.",
        'conservation_technical': "I'm here to help with conservation-related questions, but I'm currently experiencing some technical issues. Please try again shortly, and I'll provide you with information about conservation efforts and environmental protection in India.",
        'general_environmental': "I can help with environmental questions related to India, but I'm currently facing some technical challenges. Please try again in a moment.",
        'off_topic': "I specialize in wildlife, biodiversity, and conservation topics in India, but I'm currently experiencing technical difficulties. Once I'm back online, please feel free to ask me about these topics!"
    }
    
    return fallback_responses.get(question_type, fallback_responses['capability'])

@app.route('/login')
def login():
    redirect_uri = url_for('authorize', _external=True)
    logging.info(f"Redirecting to Google OAuth: {redirect_uri}")
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    token = google.authorize_access_token()  # Get the OAuth token
    user_info = google.get('https://www.googleapis.com/oauth2/v2/userinfo').json()  # Get user info from Google

    user_data = {
        "first_name": user_info.get('given_name', ''),
        "last_name": user_info.get('family_name', ''),
        "email": user_info.get('email', ''),
        "profile_picture": user_info.get('picture', '')
    }

    # Store token in session (optional, for future API calls)
    session['token'] = token
    session.permanent = True

    # Store user data (replace this with actual logic to store in database)
    user_id = user_info.get('email', '').split('@')[0] # Using email to generate unique user ID
    session['user_id'] = user_id
    store_user(user_id, user_data)

    logging.info(f"User authorized: {user_id}, email: {user_info.get('email', '')}")

    # Redirect the user to frontend with user info or token
    return redirect(f'{FRONTEND_URL}/chat?user_id={user_id}')


@app.route('/user/<user_id>', methods=['GET'])
@cache_and_rate_limit(lambda user_id: f"user_{user_id}")
def get_user_details_route(user_id):
    logging.debug(f"Fetching user details for user_id: {user_id}")  # Changed to debug to reduce log spam
    user_details = get_user(user_id)
    return jsonify(user_details)

def build_ollama_prompt(system_contexts, rag_context, conversation_history, classification_result=None):
    # Concatenate system instructions, RAG context, and chat history into a single prompt string
    prompt_parts = []
    for ctx in system_contexts:
        prompt_parts.append(ctx["content"])
    
    # Extract question type and mixed intent info
    if isinstance(classification_result, dict):
        question_type = classification_result['type']
        is_mixed = classification_result.get('is_mixed', False)
        secondary_types = classification_result.get('secondary_types', [])
    else:
        # Backward compatibility
        question_type = classification_result
        is_mixed = False
        secondary_types = []
    
    # Include context based on question type
    if rag_context.strip() and question_type not in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['GRATITUDE'], QUESTION_TYPES['CAPABILITY']]:
        if question_type == QUESTION_TYPES['WILDLIFE_TECHNICAL']:
            prompt_parts.append("Use the following wildlife and sanctuary information to answer the question:")
        elif question_type == QUESTION_TYPES['CONSERVATION_TECHNICAL']:
            prompt_parts.append("Use the following conservation and environmental information to answer the question:")
        else:
            prompt_parts.append("Use the following reference information to answer questions when relevant:")
        
        prompt_parts.append(rag_context)
        
        # Add specific formatting instructions based on question content
        user_question = ""
        if conversation_history:
            user_question = conversation_history[-1].get("content", "").lower()
        
        if any(keyword in user_question for keyword in ["ngo", "organization", "organizations", "ngos", "foundation", "foundations"]):
            prompt_parts.append("IMPORTANT: Format your response about organizations/NGOs as a clear table with columns like 'Organization Name', 'Location/Sanctuary', 'Focus Area', 'Activities'. Do NOT use bullet points or paragraph lists for organizational information.")
        else:
            prompt_parts.append("IMPORTANT: When using bullet points, format project names, species names, and key topics in **bold** (e.g., **Tiger Conservation Project**: Description of the project). This makes the content easier to scan and read.")
        
        # Handle mixed intent instructions
        if is_mixed and QUESTION_TYPES['GRATITUDE'] in secondary_types:
            prompt_parts.append("This is a mixed intent question - the user is expressing gratitude AND asking a technical question. Briefly acknowledge the thanks, then provide a comprehensive answer to the technical question based on the provided information, formatted with bullet points for easy reading.")
        elif is_mixed and QUESTION_TYPES['GREETING'] in secondary_types:
            prompt_parts.append("This is a mixed intent question - the user is greeting you AND asking a technical question. Briefly acknowledge the greeting, then provide a comprehensive answer to the technical question based on the provided information, formatted with bullet points for easy reading.")
        else:
            prompt_parts.append("Now, please answer the user's question based on this information without mentioning the reference material or context explicitly. Use bullet points to organize the information for better readability.")
    elif question_type == QUESTION_TYPES['GREETING']:
        prompt_parts.append("This is a greeting. Respond warmly and briefly explain your capabilities related to wildlife and conservation in India. Do not reference any previous conversation topics.")
    elif question_type == QUESTION_TYPES['GRATITUDE']:
        prompt_parts.append("The user is expressing gratitude. Respond politely and ask if there's anything else you can help them with regarding wildlife, biodiversity, or conservation in India.")
    elif question_type == QUESTION_TYPES['CAPABILITY']:
        prompt_parts.append("This is a question about your capabilities. Explain what you can help with regarding wildlife, biodiversity, conservation, and sanctuaries in India. Do not reference any previous conversation topics.")
    elif question_type == QUESTION_TYPES['OFF_TOPIC']:
        prompt_parts.append("This question is off-topic. Politely redirect to wildlife, conservation, or environmental topics in India.")
    
    # Handle conversation history based on question type - include full history for technical questions (including mixed intent)
    if question_type in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['GRATITUDE'], QUESTION_TYPES['CAPABILITY']] and not is_mixed:
        # For pure greetings, gratitude, and capability questions, limit conversation history to avoid influence from technical discussions
        # Only include the last greeting/capability exchange if any, or start fresh
        filtered_history = filter_conversation_history_for_non_technical(conversation_history)
        for msg in filtered_history:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
    else:
        # For technical questions (including mixed intent), include full conversation history for context
        for msg in conversation_history:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
    
    prompt_parts.append("Assistant:")  # Prompt Ollama to generate the next assistant reply
    return "\n".join(prompt_parts)

def filter_conversation_history_for_non_technical(conversation_history):
    """
    For greeting/gratitude/capability questions, filter conversation history to avoid technical influence.
    Keep only the most recent non-technical exchanges or start fresh.
    """
    if not conversation_history:
        return []
    
    # Look for the most recent greeting/gratitude/capability exchange
    filtered_history = []
    
    # Go through history in reverse to find the last greeting/gratitude/capability interaction
    for i in range(len(conversation_history) - 1, -1, -1):
        msg = conversation_history[i]
        if msg["role"] == "user":
            # Classify this historical question
            historical_classification = intelligent_question_classifier(msg["content"])
            historical_question_type = historical_classification['type'] if isinstance(historical_classification, dict) else historical_classification
            if historical_question_type in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['GRATITUDE'], QUESTION_TYPES['CAPABILITY']]:
                # Found a greeting/gratitude/capability question, include this exchange
                if i + 1 < len(conversation_history):  # Check if there's a response
                    filtered_history = [conversation_history[i], conversation_history[i + 1]]
                else:
                    filtered_history = [conversation_history[i]]
                break
    
    # If no recent greeting/capability found, start with empty history for clean response
    return filtered_history

def correct_typo_from_mistral(prompt):
    typo_prompt = f"Correct the typo in the prompt if any available. If there are no typo, you can return the same prompt: {prompt}"
    
    try:
        response = make_model_request(typo_prompt)
        if hasattr(response, 'json'):
            return response.json().get("response", "").strip()
        else:
            # For OpenAI responses, the content is directly available
            return response.strip()
    except Exception as e:
        logging.error(f"Failed to correct typo using model: {e}")
        return prompt  # Return original prompt if correction fails

def generate_heading_from_mistral(prompt):
    heading_prompt = f"Generate a short title for this conversation: {prompt}"
    
    try:
        response = make_model_request(heading_prompt, stream=False)
        if hasattr(response, 'json'):
            heading = response.json().get("response", "").strip()
        else:
            # For OpenAI responses, the content is directly available
            heading = response.strip()
        
        # Remove surrounding single or double quotes if present
        if (heading.startswith('"') and heading.endswith('"')) or (heading.startswith("'") and heading.endswith("'")):
            heading = heading[1:-1].strip()
        
        # Remove common prefixes that might be added by the AI
        prefixes_to_remove = [
            "Title:", "Heading:", "Subject:", "Topic:", "Title -", "Heading -", 
            "Subject -", "Topic -", "Chat:", "Conversation:", "Discussion:"
        ]
        
        for prefix in prefixes_to_remove:
            if heading.lower().startswith(prefix.lower()):
                heading = heading[len(prefix):].strip()
                break
        
        # Remove any leading dashes, colons, or spaces
        heading = heading.lstrip("- :").strip()
        
        # Ensure it's not empty and has reasonable length
        if not heading or len(heading) < 3:
            return "General Conversation"
        
        # Limit length to avoid overly long titles
        if len(heading) > 60:
            heading = heading[:57] + "..."
        
        return heading
    except Exception as e:
        logging.error(f"Failed to generate heading using model: {e}")
        return "General Conversation"  # Return default heading if generation fails

# Question classification constants
QUESTION_TYPES = {
    'GREETING': 'greeting',
    'GRATITUDE': 'gratitude',
    'CAPABILITY': 'capability', 
    'WILDLIFE_TECHNICAL': 'wildlife_technical',
    'CONSERVATION_TECHNICAL': 'conservation_technical',
    'GENERAL_ENVIRONMENTAL': 'general_environmental',
    'OFF_TOPIC': 'off_topic'
}

def classify_question_with_ai(question):
    """Use AI to classify ambiguous questions with confidence scoring and mixed intent detection"""
    classification_prompt = f"""
    Analyze the following question and provide:
    1. Primary category classification
    2. Confidence score (0-100)
    3. Mixed intent detection (yes/no)
    4. If mixed intent, list secondary categories

    Categories:
    1. GREETING - Greetings, introductions, or general pleasantries like hello, hi, good morning
    2. GRATITUDE - Thank you messages, appreciation, or acknowledgments like thanks, thank you
    3. CAPABILITY - Questions about what the assistant can do, its features, or capabilities
    4. WILDLIFE_TECHNICAL - Questions about specific animals, species, habitats, behavior, or wildlife-related topics
    5. CONSERVATION_TECHNICAL - Questions about conservation efforts, protected areas, environmental protection, or sustainability
    6. GENERAL_ENVIRONMENTAL - General environmental questions that may need some context
    7. OFF_TOPIC - Questions completely unrelated to wildlife, conservation, or environment

    Question: "{question}"
    
    Respond in this exact format:
    PRIMARY: [category]
    CONFIDENCE: [0-100]
    MIXED_INTENT: [yes/no]
    SECONDARY: [category1,category2] (only if mixed intent is yes)"""
    
    try:
        response = make_model_request(classification_prompt, stream=False)
        if hasattr(response, 'json'):
            response_text = response.json().get("response", "").strip()
        else:
            # For OpenAI responses, the content is directly available
            response_text = response.strip()
        return parse_enhanced_classification(response_text)
        
    except Exception as e:
        logging.warning(f"AI classification failed due to model unavailability: {e}")
        return create_default_classification()

def parse_enhanced_classification(response_text):
    """Parse the enhanced AI classification response"""
    classification_map = {
        'GREETING': QUESTION_TYPES['GREETING'],
        'GRATITUDE': QUESTION_TYPES['GRATITUDE'],
        'CAPABILITY': QUESTION_TYPES['CAPABILITY'],
        'WILDLIFE_TECHNICAL': QUESTION_TYPES['WILDLIFE_TECHNICAL'],
        'CONSERVATION_TECHNICAL': QUESTION_TYPES['CONSERVATION_TECHNICAL'],
        'GENERAL_ENVIRONMENTAL': QUESTION_TYPES['GENERAL_ENVIRONMENTAL'],
        'OFF_TOPIC': QUESTION_TYPES['OFF_TOPIC']
    }
    
    try:
        lines = response_text.upper().split('\n')
        result = {
            'primary_type': QUESTION_TYPES['GENERAL_ENVIRONMENTAL'],
            'confidence': 70,
            'is_mixed': False,
            'secondary_types': []
        }
        
        for line in lines:
            if line.startswith('PRIMARY:'):
                primary = line.split(':', 1)[1].strip()
                result['primary_type'] = classification_map.get(primary, QUESTION_TYPES['GENERAL_ENVIRONMENTAL'])
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip())
                    result['confidence'] = max(0, min(100, confidence))
                except ValueError:
                    result['confidence'] = 70
            elif line.startswith('MIXED_INTENT:'):
                mixed = line.split(':', 1)[1].strip().lower()
                result['is_mixed'] = mixed == 'yes'
            elif line.startswith('SECONDARY:'):
                if result['is_mixed']:
                    secondary_str = line.split(':', 1)[1].strip()
                    secondary_categories = [cat.strip() for cat in secondary_str.split(',')]
                    result['secondary_types'] = [
                        classification_map.get(cat, None) for cat in secondary_categories
                        if cat in classification_map
                    ]
        
        return result
    except Exception as e:
        logging.warning(f"Failed to parse enhanced classification: {e}")
        return create_default_classification()

def create_default_classification():
    """Create default classification result"""
    return {
        'primary_type': QUESTION_TYPES['GENERAL_ENVIRONMENTAL'],
        'confidence': 70,
        'is_mixed': False,
        'secondary_types': []
    }

def resolve_basic_pronouns(question, conversation_history):
    """Basic pronoun resolution using conversation history"""
    if not conversation_history:
        return question
    
    # Get the last few messages for context
    recent_messages = conversation_history[-4:]  # Last 2 exchanges
    
    # Common pronouns to resolve
    pronouns = ['it', 'they', 'them', 'this', 'that', 'these', 'those']
    
    question_lower = question.lower()
    if not any(pronoun in question_lower for pronoun in pronouns):
        return question  # No pronouns to resolve
    
    # Extract potential subjects from recent messages
    subjects = []
    technical_keywords = [
        'tiger', 'elephant', 'leopard', 'rhino', 'lion', 'bear', 'deer', 'bird',
        'sanctuary', 'national park', 'reserve', 'conservation', 'wildlife',
        'habitat', 'ecosystem', 'forest', 'species', 'protection', 'biodiversity'
    ]
    
    for msg in recent_messages:
        if msg['role'] == 'user' or msg['role'] == 'assistant':
            content = msg['content'].lower()
            # Find technical keywords that could be pronoun referents
            for keyword in technical_keywords:
                if keyword in content:
                    subjects.append(keyword)
    
    # Simple replacement for obvious cases
    resolved_question = question
    if subjects:
        # Use the most recent subject mentioned
        latest_subject = subjects[-1]
        
        # Basic pronoun replacements
        replacements = {
            r'\bit\b': latest_subject,
            r'\bthey\b': f"{latest_subject}s",
            r'\bthem\b': f"{latest_subject}s", 
            r'\bthis\b': f"this {latest_subject}",
            r'\bthat\b': f"that {latest_subject}"
        }
        
        for pattern, replacement in replacements.items():
            resolved_question = re.sub(pattern, replacement, resolved_question, flags=re.IGNORECASE)
    
    return resolved_question

def handle_mixed_intent_question(classification_result, user_question, conversation_history):
    """Handle questions with mixed intent by providing appropriate context"""
    if not classification_result['is_mixed']:
        return classification_result['primary_type'], ""
    
    # For mixed intent, prioritize based on confidence and context needs
    primary_type = classification_result['primary_type']
    secondary_types = classification_result['secondary_types']
    
    # Log mixed intent for monitoring
    logging.info(f"Mixed intent detected - Primary: {primary_type}, Secondary: {secondary_types}, Confidence: {classification_result['confidence']}")
    
    # If confidence is low, be more conservative with context usage
    if classification_result['confidence'] < 60:
        # For low confidence, prefer safer non-technical handling
        if QUESTION_TYPES['GREETING'] in [primary_type] + secondary_types:
            return QUESTION_TYPES['GREETING'], "mixed_intent_low_confidence"
        elif QUESTION_TYPES['CAPABILITY'] in [primary_type] + secondary_types:
            return QUESTION_TYPES['CAPABILITY'], "mixed_intent_low_confidence"
    
    # For higher confidence mixed intent, use primary classification but note mixed nature
    return primary_type, "mixed_intent"
    """Fast pattern matching for obvious greetings"""
    obvious_greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'thanks', 'thank you', 'good day', 'good night'
    ]
    text_lower = text.lower().strip()
    
    # Check for exact matches at start of text
    for greeting in obvious_greetings:
        if text_lower.startswith(greeting) or text_lower == greeting:
            return True
    return False

def is_obvious_capability(text):
    """Fast pattern matching for obvious capability questions"""
    capability_patterns = [
        r'^\s*what\s+(can|do)\s+you\s+do\s*\??$',
        r'^\s*what\s+are\s+your\s+(capabilities|features|functions)\s*\??$',
        r'^\s*what\s+all\s+can\s+you\s+do\s*\??$',
        r'^\s*how\s+can\s+you\s+help\s*\??$',
        r'^\s*what\s+services\s+do\s+you\s+provide\s*\??$'
    ]
    
    text_lower = text.lower().strip()
    for pattern in capability_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def intelligent_question_classifier(question, conversation_history=None):
    """Enhanced hybrid approach with mixed intent detection and pronoun resolution"""
    
    # Step 1: Basic pronoun resolution if conversation history available
    resolved_question = question
    if conversation_history:
        resolved_question = resolve_basic_pronouns(question, conversation_history)
        if resolved_question != question:
            logging.info(f"Pronoun resolution: '{question}' -> '{resolved_question}'")
    
    # Step 2: Check for mixed intent patterns first (gratitude + technical content)
    has_gratitude = is_obvious_gratitude(question)
    has_greeting = is_obvious_greeting(question) and not has_gratitude  # Don't double-count "thanks"
    has_capability = is_obvious_capability(question)
    has_technical = is_obvious_technical(question)
    
    # Debug logging for classification
    logging.info(f"Classification debug for '{question}': gratitude={has_gratitude}, greeting={has_greeting}, capability={has_capability}, technical={has_technical}")
    
    # Handle mixed intent cases where gratitude/greeting is combined with technical questions
    if (has_gratitude or has_greeting) and has_technical:
        logging.info(f"Mixed intent detected: {'gratitude' if has_gratitude else 'greeting'} + technical")
        # This is a mixed intent question - prioritize the technical aspect
        ai_result = classify_question_with_ai(resolved_question)
        if ai_result['primary_type'] in [QUESTION_TYPES['WILDLIFE_TECHNICAL'], QUESTION_TYPES['CONSERVATION_TECHNICAL']]:
            return {
                'type': ai_result['primary_type'],
                'confidence': min(ai_result['confidence'], 85),  # Slightly lower confidence for mixed intent
                'is_mixed': True,
                'secondary_types': [QUESTION_TYPES['GRATITUDE'] if has_gratitude else QUESTION_TYPES['GREETING']],
                'context_hint': 'mixed_gratitude_technical'
            }
        else:
            # If AI doesn't classify it as technical but we detected technical keywords, force wildlife technical
            logging.info(f"AI classified as {ai_result['primary_type']}, but forcing WILDLIFE_TECHNICAL due to mixed intent detection")
            return {
                'type': QUESTION_TYPES['WILDLIFE_TECHNICAL'],
                'confidence': 80,
                'is_mixed': True,
                'secondary_types': [QUESTION_TYPES['GRATITUDE'] if has_gratitude else QUESTION_TYPES['GREETING']],
                'context_hint': 'mixed_technical_forced'
            }
    
    # Step 3: Single intent fast checks (only if no mixed intent detected)
    if has_greeting and not has_technical:
        return {
            'type': QUESTION_TYPES['GREETING'],
            'confidence': 95,
            'is_mixed': False,
            'context_hint': 'fast_greeting'
        }
    
    if has_gratitude and not has_technical:
        return {
            'type': QUESTION_TYPES['GRATITUDE'],
            'confidence': 95,
            'is_mixed': False,
            'context_hint': 'fast_gratitude'
        }
    
    if has_capability and not has_technical:
        return {
            'type': QUESTION_TYPES['CAPABILITY'],
            'confidence': 95,
            'is_mixed': False,
            'context_hint': 'fast_capability'
        }
    
    if has_technical and not (has_gratitude or has_greeting or has_capability):
        # Pure technical question
        ai_result = classify_question_with_ai(resolved_question)
        if ai_result['primary_type'] in [QUESTION_TYPES['WILDLIFE_TECHNICAL'], QUESTION_TYPES['CONSERVATION_TECHNICAL']]:
            return {
                'type': ai_result['primary_type'],
                'confidence': min(ai_result['confidence'], 90),
                'is_mixed': ai_result['is_mixed'],
                'context_hint': 'technical_with_ai'
            }
        else:
            return {
                'type': QUESTION_TYPES['WILDLIFE_TECHNICAL'],
                'confidence': 75,
                'is_mixed': False,
                'context_hint': 'technical_fallback'
            }
    
    # Step 4: For ambiguous cases, use full AI classification
    ai_result = classify_question_with_ai(resolved_question)
    return {
        'type': ai_result['primary_type'],
        'confidence': ai_result['confidence'],
        'is_mixed': ai_result['is_mixed'],
        'context_hint': 'ai_classification'
    }

def is_obvious_gratitude(text):
    """Fast pattern matching for gratitude expressions"""
    gratitude_expressions = [
        'thanks', 'thank you', 'thank you so much', 'thanks a lot', 
        'much appreciated', 'appreciate it', 'thanks for the help',
        'thank you for the information', 'that was helpful'
    ]
    text_lower = text.lower().strip()
    
    # Check for exact matches or if the text starts with gratitude
    for expression in gratitude_expressions:
        if text_lower == expression or text_lower.startswith(expression):
            return True
    
    # Check for common gratitude patterns
    if re.search(r'\b(thanks?|thank\s+you)\b', text_lower):
        return True
    
    return False

def is_obvious_greeting(text):
    """Fast pattern matching for obvious greetings"""
    obvious_greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'good day', 'good night'
    ]
    text_lower = text.lower().strip()
    
    # Check for exact matches at start of text
    for greeting in obvious_greetings:
        if text_lower.startswith(greeting) or text_lower == greeting:
            return True
    return False

def is_obvious_capability(text):
    """Fast pattern matching for obvious capability questions"""
    capability_patterns = [
        r'^\s*what\s+(can|do)\s+you\s+do\s*\??$',
        r'^\s*what\s+are\s+your\s+(capabilities|features|functions)\s*\??$',
        r'^\s*what\s+all\s+can\s+you\s+do\s*\??$',
        r'^\s*how\s+can\s+you\s+help\s*\??$',
        r'^\s*what\s+services\s+do\s+you\s+provide\s*\??$'
    ]
    
    text_lower = text.lower().strip()
    for pattern in capability_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def is_obvious_technical(text):
    """Fast pattern matching for obvious technical questions"""
    technical_keywords = [
        'tiger', 'tigers', 'elephant', 'elephants', 'leopard', 'leopards', 
        'lion', 'lions', 'rhino', 'rhinoceros', 'rhinos', 'cheetah', 'cheetahs',
        'bear', 'bears', 'deer', 'monkey', 'monkeys', 'bird', 'birds', 
        'snake', 'snakes', 'reptile', 'reptiles', 'crocodile', 'crocodiles',
        'turtle', 'turtles', 'wolf', 'wolves', 'fox', 'foxes', 'jackal', 'jackals',
        'sanctuary', 'sanctuaries', 'national park', 'national parks', 'wildlife', 
        'conservation', 'biodiversity', 'species', 'habitat', 'habitats', 
        'ecosystem', 'ecosystems', 'forest', 'forests', 'reserve', 'reserves', 
        'protection', 'endangered', 'ngo', 'ngos', 'organization', 'organizations', 
        'foundation', 'foundations', 'trust', 'society'
    ]
    
    text_lower = text.lower()
    
    # Check for exact keyword matches
    for keyword in technical_keywords:
        if keyword in text_lower:
            return True
    
    # Check for NGO variations (ngo's, ngo, ngos, etc.)
    if re.search(r'\bngo\'?s?\b', text_lower):
        return True
        
    # Check for organization variations
    if re.search(r'\borg(?:anization|anisation)s?\b', text_lower):
        return True
    
    return False

def should_use_context(classification_result):
    """Determine if context retrieval is needed based on classification result"""
    if isinstance(classification_result, dict):
        question_type = classification_result['type']
        confidence = classification_result.get('confidence', 70)
        is_mixed = classification_result.get('is_mixed', False)
    else:
        # Backward compatibility
        question_type = classification_result
        confidence = 70
        is_mixed = False
    
    context_config = {
        QUESTION_TYPES['GREETING']: False,
        QUESTION_TYPES['GRATITUDE']: False,
        QUESTION_TYPES['CAPABILITY']: False,
        QUESTION_TYPES['WILDLIFE_TECHNICAL']: True,
        QUESTION_TYPES['CONSERVATION_TECHNICAL']: True,
        QUESTION_TYPES['GENERAL_ENVIRONMENTAL']: True,
        QUESTION_TYPES['OFF_TOPIC']: False
    }
    
    base_decision = context_config.get(question_type, True)
    
    # For mixed intent or low confidence, be more conservative
    if is_mixed and confidence < 70:
        # If mixed and low confidence, prefer not using context for safety
        return False
    
    return base_decision

# Backward compatibility function
def is_greeting(text):
    """Backward compatibility wrapper"""
    classification_result = intelligent_question_classifier(text)
    if isinstance(classification_result, dict):
        question_type = classification_result['type']
    else:
        question_type = classification_result
    return question_type in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['GRATITUDE'], QUESTION_TYPES['CAPABILITY']]


# Helper to build retrieval query from conversation history
# Uses last MAX_HISTORY turns (user and assistant)
def build_retrieval_query(conversation_history, user_question, max_turns=MAX_HISTORY):
    # Get last max_turns*2 messages (user+assistant), excluding the current user_question
    history = conversation_history[-max_turns*2:] if conversation_history else []
    # Add the current user question as the last turn
    turns = []
    for msg in history:
        if msg['role'] == 'user':
            turns.append(f"User: {msg['content']}")
        elif msg['role'] == 'assistant':
            turns.append(f"Assistant: {msg['content']}")
    turns.append(f"User: {user_question}")
    return "\n".join(turns)

@app.route('/ask_with_stream', methods=['GET'])
def ask_question_with_stream_route():
    try:
        # Immediate debug output
        print("🚀 DEBUG: Route /ask_with_stream was reached!")
        logging.info("🚀 Route /ask_with_stream was reached!")
        
        user_question = request.args.get("question")
        user_id = request.args.get("user_id")
        chat_id = request.args.get("chat_id")

        print(f"🔍 DEBUG: Received parameters - user_id: {user_id}, chat_id: {chat_id}, question: {user_question}")
        
        if not user_question or not user_id:
            return Response("data: " + json.dumps({"error": "'question' and 'user_id' are required"}) + "\n\n", content_type="text/event-stream")

        logging.info(f"Streaming question from user_id: {user_id}, chat_id: {chat_id}, question: {user_question}")
        print(f"📝 PRINT: Streaming question from user_id: {user_id}, chat_id: {chat_id}, question: {user_question}")
        
        # Add detailed logging for chat_id handling
        if chat_id:
            logging.info(f"Looking up existing chat with ID: {chat_id}")
        else:
            logging.info("No chat_id provided, will create new chat if needed")

        # Check model health before processing
        if not check_model_health():
            logging.warning("Model service is not available, returning fallback response")
            
            # Quick classification using simple pattern matching when model is down
            question_type = 'greeting' if is_obvious_greeting(user_question) else \
                           'gratitude' if is_obvious_gratitude(user_question) else \
                           'capability' if is_obvious_capability(user_question) else \
                           'wildlife_technical' if is_obvious_technical(user_question) else \
                           'general_environmental'
                           
            fallback_response = get_model_fallback_response(question_type)
            
            # Still maintain chat history even with fallback responses
            chat = get_chat(chat_id) if chat_id else None
            
            if chat_id:
                if chat:
                    logging.info(f"Found existing chat {chat_id} for fallback with {len(chat.get('messages', []))} messages")
                else:
                    logging.warning(f"Chat {chat_id} not found in database for fallback")
            
            if not chat:
                if chat_id:
                    # If chat_id was provided but chat doesn't exist, create new chat with provided ID
                    logging.warning(f"Chat {chat_id} not found for user {user_id} (fallback), creating new chat")
                    chat = {
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "heading": "Service Unavailable",
                        "messages": [],
                        "created_at": datetime.utcnow().isoformat()
                    }
                else:
                    # Only create new chat if no chat_id was provided
                    chat_id = str(uuid.uuid4())
                    logging.info(f"Creating new chat with ID: {chat_id} (fallback)")
                    chat = {
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "heading": "Service Unavailable",
                        "messages": [],
                        "created_at": datetime.utcnow().isoformat()
                    }
            
            conversation_history = chat.get("messages", [])
            conversation_history.append({"role": "user", "content": user_question})
            conversation_history.append({"role": "assistant", "content": fallback_response})
            store_chat(user_id, chat_id, chat["heading"], conversation_history)
            
            def generate_fallback():
                yield f"data: {json.dumps({'chat_id': chat_id})}\n\n"
                yield f"data: {json.dumps({'content': fallback_response})}\n\n"
            
            return Response(generate_fallback(), content_type="text/event-stream")

        chat = get_chat(chat_id) if chat_id else None
        
        if chat_id:
            if chat:
                logging.info(f"Found existing chat {chat_id} with {len(chat.get('messages', []))} messages")
            else:
                logging.warning(f"Chat {chat_id} not found in database")

        if not chat:
            if chat_id:
                # If chat_id was provided but chat doesn't exist, create a new chat with the provided ID
                # This handles cases where frontend has a chat_id but backend lost the data
                logging.warning(f"Chat {chat_id} not found for user {user_id}, creating new chat with provided ID")
                heading = generate_heading_from_mistral(user_question)
                chat = {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "heading": heading,
                    "messages": [],
                    "created_at": datetime.utcnow().isoformat()
                }
                store_chat(user_id, chat_id, heading, chat["messages"])
            else:
                # Only create new chat if no chat_id was provided (new conversation)
                chat_id = str(uuid.uuid4())
                logging.info(f"Creating new chat with ID: {chat_id}")
                heading = generate_heading_from_mistral(user_question)
                chat = {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "heading": heading,
                    "messages": [],
                    "created_at": datetime.utcnow().isoformat()
                }
                store_chat(user_id, chat_id, heading, chat["messages"])

        chat_heading = chat["heading"]
        conversation_history = chat["messages"]
        conversation_history.append({"role": "user", "content": user_question})

        print("🔥 DEBUG: About to start question classification...")
        print(f"🔥 DEBUG: Question to classify: '{user_question}'")
        
        # Enhanced intelligent question classification with mixed intent detection
        classification_result = intelligent_question_classifier(user_question, conversation_history[:-1])
        question_type = classification_result['type'] if isinstance(classification_result, dict) else classification_result
        
        print(f"🎯 DEBUG: Classification completed - Type: {question_type}")
        print(f"🎯 DEBUG: Full result: {classification_result}")
        
        # Log enhanced classification details
        try:
            logging.info("=" * 60)
            logging.info(f"📝 QUESTION CLASSIFICATION RESULTS")
            print("=" * 60)
            print(f"📝 QUESTION CLASSIFICATION RESULTS")
            logging.info(f"   Question: '{user_question}'")
            print(f"   Question: '{user_question}'")
            if isinstance(classification_result, dict):
                logging.info(f"   ✅ Primary Category: {question_type.upper()}")
                logging.info(f"   🎯 Confidence Score: {classification_result['confidence']}%")
                logging.info(f"   🔄 Mixed Intent: {'YES' if classification_result['is_mixed'] else 'NO'}")
                print(f"   ✅ Primary Category: {question_type.upper()}")
                print(f"   🎯 Confidence Score: {classification_result['confidence']}%")
                print(f"   🔄 Mixed Intent: {'YES' if classification_result['is_mixed'] else 'NO'}")
                if classification_result.get('is_mixed') and classification_result.get('secondary_types'):
                    logging.info(f"   📋 Secondary Categories: {', '.join(classification_result['secondary_types'])}")
                    print(f"   📋 Secondary Categories: {', '.join(classification_result['secondary_types'])}")
                if classification_result.get('context_hint'):
                    logging.info(f"   🔍 Classification Method: {classification_result['context_hint']}")
                    print(f"   🔍 Classification Method: {classification_result['context_hint']}")
            else:
                logging.info(f"   ✅ Category: {question_type.upper()}")
                print(f"   ✅ Category: {question_type.upper()}")
            logging.info("=" * 60)
            print("=" * 60)
        except Exception as e:
            print(f"❌ ERROR in classification logging: {e}")
            logging.error(f"❌ ERROR in classification logging: {e}")
        
        if should_use_context(classification_result):
            # For technical questions, build retrieval query normally
            retrieval_query = build_retrieval_query(conversation_history[:-1], user_question, max_turns=MAX_HISTORY)
            logging.info(f"🔍 Building retrieval query for context search...")
            logging.info(f"   Query built from conversation history: {len(conversation_history[:-1])} previous messages")
            
            # Check if the question is asking about a specific sanctuary vs. general information
            # First, check for general query patterns that should use all available context
            general_query_patterns = [
                r'\b(?:where are all|list all|all the|how many|what are all)\b.*?(?:tiger reserve|national park|wildlife sanctuary|sanctuary|reserve)',
                r'\b(?:tell me about all|show me all|give me a list)\b.*?(?:tiger reserve|national park|wildlife sanctuary|sanctuary|reserve)',
                r'\b(?:tiger reserves? in india|national parks in india|sanctuaries in india)\b'
            ]
            
            is_general_query = any(re.search(pattern, user_question.lower()) for pattern in general_query_patterns)
            
            if is_general_query:
                logging.info(f"🌍 General query detected - will use all available context")
                sanctuary_matches = []  # Don't treat as specific sanctuary query
            else:
                # Check if the question is asking about a specific sanctuary
                # Improved pattern to avoid matching articles and common words, and require proper sanctuary names
                sanctuary_pattern = r'\b(?!(?:the|all|any|where|what|which|how|many)\b)([A-Za-z][A-Za-z\s]{2,}?)\s+(?:wls|wildlife sanctuary|national park|tiger reserve|biosphere reserve|sanctuary|park|reserve)\b'
                sanctuary_matches = re.findall(sanctuary_pattern, user_question.lower())
                # Filter out obvious non-sanctuary matches
                sanctuary_matches = [match.strip() for match in sanctuary_matches if match.strip().lower() not in ['the', 'all', 'any', 'where', 'what', 'which', 'how', 'many', 'are', 'is']]
            
            # More specific state pattern - only match after prepositions that indicate location
            state_pattern = r'\b(?:in|from|of|at)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s*[,.]|\s*$)'
            state_matches = re.findall(state_pattern, user_question.lower())
            # Filter out common non-state words
            state_matches = [state.strip() for state in state_matches if state.strip().lower() not in ['about', 'the', 'a', 'an', 'this', 'that']]
            
            if sanctuary_matches:
                logging.info(f"🏞️  Sanctuary-specific query detected: {sanctuary_matches}")
                if state_matches:
                    logging.info(f"🗺️  State mentioned: {state_matches}")
            elif is_general_query:
                logging.info(f"🌍 General information query - will provide comprehensive context")
            
            context_metadatas = get_structured_context(retrieval_query)
            
            # Enhanced context retrieval logging
            logging.info("-" * 50)
            logging.info(f"📊 CONTEXT RETRIEVAL RESULTS")
            logging.info(f"   Retrieved Documents: {len(context_metadatas)}")
            
            if context_metadatas:
                # Extract and analyze organizations
                all_orgs = set()
                all_sanctuaries = set()
                all_urls = set()
                content_lengths = []
                
                for i, ctx in enumerate(context_metadatas, 1):
                    # Parse NGOs
                    ngos_data = ctx.get('ngos', '[]')
                    try:
                        ngos_list = json.loads(ngos_data) if isinstance(ngos_data, str) else ngos_data
                        if isinstance(ngos_list, list):
                            orgs = [ngo.strip() for ngo in ngos_list if ngo and ngo.strip()]
                            all_orgs.update(orgs)
                        elif ngos_data:
                            all_orgs.add(str(ngos_data).strip())
                    except (json.JSONDecodeError, TypeError):
                        if ngos_data:
                            all_orgs.add(str(ngos_data).strip())
                    
                    # Parse sanctuaries
                    sanctuary_data = ctx.get('sanctuary', '[]')
                    try:
                        sanctuary_list = json.loads(sanctuary_data) if isinstance(sanctuary_data, str) else sanctuary_data
                        if isinstance(sanctuary_list, list):
                            sanctuaries = [s.strip() for s in sanctuary_list if s and s.strip()]
                            all_sanctuaries.update(sanctuaries)
                        elif sanctuary_data:
                            all_sanctuaries.add(str(sanctuary_data).strip())
                    except (json.JSONDecodeError, TypeError):
                        if sanctuary_data:
                            all_sanctuaries.add(str(sanctuary_data).strip())
                    
                    # Collect URLs and content lengths
                    if ctx.get('url'):
                        all_urls.add(ctx['url'])
                    
                    content_length = len(ctx.get('content', ''))
                    content_lengths.append(content_length)
                    
                    # Log individual context details (first 3 for brevity)
                    if i <= 3:
                        logging.info(f"   📄 Context {i}:")
                        if ctx.get('url'):
                            logging.info(f"      🔗 URL: {ctx['url']}")
                        if orgs:
                            logging.info(f"      🏢 Organization(s): {', '.join(orgs)}")
                        if sanctuaries:
                            logging.info(f"      🌲 Sanctuary(ies): {', '.join(sanctuaries)}")
                        logging.info(f"      📝 Content length: {content_length} chars")
                
                # Summary statistics
                logging.info(f"   📈 SUMMARY:")
                logging.info(f"      🏢 Unique Organizations: {len(all_orgs)} ({', '.join(sorted(all_orgs))})")
                logging.info(f"      🌲 Unique Sanctuaries: {len(all_sanctuaries)} ({', '.join(sorted(all_sanctuaries))})")
                logging.info(f"      🔗 Unique Sources: {len(all_urls)}")
                if content_lengths:
                    avg_length = sum(content_lengths) / len(content_lengths)
                    logging.info(f"      📊 Content: avg {avg_length:.0f} chars, range {min(content_lengths)}-{max(content_lengths)}")
            else:
                logging.info(f"   ❌ No relevant context found")
            logging.info("-" * 50)
            
            # If asking about a specific sanctuary, validate if the retrieved context actually contains that sanctuary
            if sanctuary_matches:
                sanctuary_name = sanctuary_matches[0]
                state_name = state_matches[0] if state_matches else None
                
                # Check if the retrieved context actually contains the requested sanctuary
                found_requested_sanctuary = False
                if context_metadatas:
                    for ctx in context_metadatas:
                        sanctuary_data = ctx.get('sanctuary', '[]')
                        try:
                            sanctuary_list = json.loads(sanctuary_data) if isinstance(sanctuary_data, str) else sanctuary_data
                            if isinstance(sanctuary_list, list):
                                # Check if any sanctuary in the list matches the requested one (case-insensitive)
                                for sanctuary in sanctuary_list:
                                    if sanctuary and sanctuary_name.lower() in sanctuary.lower():
                                        found_requested_sanctuary = True
                                        break
                            elif sanctuary_data and sanctuary_name.lower() in str(sanctuary_data).lower():
                                found_requested_sanctuary = True
                        except (json.JSONDecodeError, TypeError):
                            if sanctuary_data and sanctuary_name.lower() in str(sanctuary_data).lower():
                                found_requested_sanctuary = True
                        
                        if found_requested_sanctuary:
                            break
                
                if not found_requested_sanctuary:
                    # The retrieved context doesn't contain the requested sanctuary
                    logging.info(f"🔍 Retrieved context doesn't contain requested sanctuary '{sanctuary_name}'")
                    logging.info(f"🔍 Checking if sanctuary '{sanctuary_name}' exists in database...")
                    
                    if not check_sanctuary_exists(sanctuary_name, state_name):
                        logging.info(f"❌ Sanctuary '{sanctuary_name}' not found in database")
                        # Set context to indicate the sanctuary is not in our database
                        rag_context = f"The sanctuary '{sanctuary_name}' mentioned is not currently in our conservation database. Our database primarily contains information about other wildlife sanctuaries and conservation projects."
                        context_sources = []
                    else:
                        logging.info(f"✅ Sanctuary '{sanctuary_name}' exists but specific context not retrieved")
                        # Sanctuary exists but specific context wasn't retrieved - provide general response
                        rag_context = f"While '{sanctuary_name}' exists in our database, detailed information about this specific sanctuary is not currently available in our context. Our database contains information about other wildlife sanctuaries and conservation projects."
                        context_sources = []
                else:
                    logging.info(f"✅ Found requested sanctuary '{sanctuary_name}' in retrieved context")
                    # Extract sources from context for proper attribution
                    context_sources = extract_sources_from_context(context_metadatas)
                    rag_context = build_structured_context(context_metadatas)
            else:
                # No specific sanctuary mentioned - use all retrieved context
                context_sources = extract_sources_from_context(context_metadatas)
                rag_context = build_structured_context(context_metadatas)
            
            final_context_length = len(rag_context) if isinstance(rag_context, str) else 0
            logging.info(f"📋 Final structured context: {final_context_length} characters")
            if context_sources:
                logging.info(f"📚 Sources for attribution: {len(context_sources)} URLs")
                for i, source in enumerate(context_sources, 1):
                    logging.info(f"   {i}. {source}")
        else:
            # For greetings/capability questions, don't retrieve any context
            rag_context = ""
            context_sources = []
            logging.info("-" * 50)
            logging.info(f"🚫 CONTEXT RETRIEVAL SKIPPED")
            logging.info(f"   Reason: Question type '{question_type}' doesn't require context")
            logging.info(f"   Categories that skip context: greetings, gratitude, capability, off-topic")
            logging.info("-" * 50)

        # Final processing summary
        logging.info("🎯 PROCESSING SUMMARY")
        logging.info(f"   User: {user_id} | Chat: {chat_id}")
        logging.info(f"   Question Type: {question_type.upper()}")
        logging.info(f"   Context Used: {'YES' if rag_context else 'NO'}")
        if rag_context:
            logging.info(f"   Context Length: {len(rag_context)} chars from {len(context_sources) if context_sources else 0} sources")
        logging.info(f"   Conversation Length: {len(conversation_history)} messages")
        logging.info("=" * 60)

        prompt = build_ollama_prompt(system_contexts, rag_context, conversation_history, classification_result)

        def generate():
            buffer = ""
            try:
                # Send chat_id at the start of the stream
                yield f"data: {json.dumps({'chat_id': chat_id})}\n\n"
                
                # Use the unified model request method for streaming
                response = make_model_request(prompt, stream=True)
                
                # Handle different response types (Ollama vs OpenAI streaming)
                if hasattr(response, 'iter_lines'):
                    # Ollama response - iterate over lines
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                data = json.loads(line)
                                token = data.get("response", "")
                                buffer += token
                                yield f"data: {json.dumps({'content': token})}\n\n"
                            except json.JSONDecodeError:
                                continue
                else:
                    # OpenAI streaming response
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            token = chunk.choices[0].delta.content
                            buffer += token
                            yield f"data: {json.dumps({'content': token})}\n\n"

                # Filter and clean the AI response (removes any hallucinated sources)
                filtered_buffer = filter_sources(buffer, rag_context, question_type, context_sources)
                
                # Stream sources separately for technical questions
                if question_type not in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['GRATITUDE'], 
                                       QUESTION_TYPES['CAPABILITY'], QUESTION_TYPES['OFF_TOPIC']] and context_sources:
                    # Add sources section to streaming output
                    sources_header = '\n\n### Sources\n'
                    yield f"data: {json.dumps({'content': sources_header})}\n\n"
                    for source in context_sources:
                        source_line = f'- {source}\n'
                        yield f"data: {json.dumps({'content': source_line})}\n\n"
                    
                    # Also add to the buffer for storage
                    filtered_buffer = manually_append_sources(filtered_buffer, context_sources, question_type)
                
                conversation_history.append({"role": "assistant", "content": filtered_buffer})
                store_chat(user_id, chat_id, chat["heading"], conversation_history)

                logging.info(f"Model streaming response for user_id: {user_id}, chat_id: {chat_id}, length: {len(filtered_buffer)}")

            except Exception as e:
                error_msg = f"Model service is currently unavailable. Please try again."
                logging.error(f"Model request failed: {e}")
                yield f"data: {json.dumps({'error': error_msg, 'details': str(e), 'chat_id': chat_id})}\n\n"
                
                # Store error message in conversation history for context
                error_response = "I apologize, but I'm currently unable to process your request due to a service issue. Please try again in a moment."
                conversation_history.append({"role": "assistant", "content": error_response})
                store_chat(user_id, chat_id, chat["heading"], conversation_history)

        return Response(generate(), content_type="text/event-stream")

    except Exception as e:
        return Response("data: " + json.dumps({"error": "Internal server error", "details": str(e)}) + "\n\n", content_type="text/event-stream")

@app.route('/delete_chat/<chat_id>', methods=['DELETE'])
def delete_chat_route(chat_id):
    """Delete a specific chat."""
    data = request.args.to_dict()
    if "user_id" not in data:
        return jsonify({"error": "'user_id' field is required"}), 400

    logging.info(f"Deleting chat_id: {chat_id} for user_id: {data.get('user_id')}")
    delete_chat(chat_id)
    return jsonify({"message": "Chat deleted successfully"})


@app.route('/list_users', methods=['GET'])
def list_users_route():
    """Retrieve all users."""
    try:
        logging.info("Listing all users")
        users = list_users()
        return jsonify({"users": users})

    except Exception as e:
        return jsonify({"error": "Failed to retrieve users", "details": str(e)}), 500

@app.route('/links', methods=['GET'])
def list_links_api():
    """Retrieve all users."""
    try:
        logging.info("Listing all links")
        links = list_links()
        return jsonify({"links": links})

    except Exception as e:
        return jsonify({"error": "Failed to retrieve links", "details": str(e)}), 500

@app.route('/link', methods=['POST'])
def add_link_api():
    """Retrieve all users."""
    try:
        data = request.get_json()
        logging.info(f"Adding new link: {data}")
        add_link(data)
        return jsonify({"status": "ok"})

    except Exception as e:
        return jsonify({"error": "Failed to save link", "details": str(e)}), 500

@app.route('/link/<link_id>', methods=['DELETE'])
def delete_link_api(link_id):
    """Retrieve all users."""
    try:
        logging.info(f"Deleting link_id: {link_id}")
        link = delete_link(link_id)
        return jsonify({"link": link, "status" : "ok"})

    except Exception as e:
        return jsonify({"error": "Failed to delete link", "details": str(e)}), 500


@app.route('/list_chats', methods=['GET'])
@cache_and_rate_limit(lambda: f"list_chats_{request.args.get('user_id', 'unknown')}")
def list_chats_route():
    """List all chats for a user with filtering, sorting, search, and pagination."""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "'user_id' is required"}), 400

        logging.debug(f"Listing chats for user_id: {user_id}")  # Changed to debug to reduce log spam

        # Get optional query parameters safely
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        sort_order = request.args.get("sort_order", "desc").lower()
        search_query = request.args.get("search", "").strip()

        # Ensure `page` is integer and handle invalid values
        try:
            page = int(request.args.get("page", 1) or 1)
        except ValueError:
            return jsonify({"error": "Invalid 'page' value"}), 400

        logging.info(f"Listing chats for user_id: {user_id}, page: {page}")

        # Fetch filtered, sorted, paginated chats (no limit)
        chat_data = list_chats(user_id, start_date, end_date, sort_order, search_query, page, None)

        return jsonify({
            "user_id": user_id,
            "page": page,
            **chat_data
        })

    except Exception as e:
        return jsonify({"error": "Invalid request", "details": str(e)}), 400


@app.route('/get_chat/<chat_id>', methods=['GET'])
def get_chat_route(chat_id):
    """Retrieve a chat by ID."""
    logging.info(f"Fetching chat_id: {chat_id}")
    chat = get_chat(chat_id)
    if chat:
        return jsonify(chat)
    else:
        return jsonify({"error": "Chat not found"}), 404


@app.route('/rename_chat/<chat_id>', methods=['POST'])
def rename_chat_route(chat_id):
    """Rename a chat in ChromaDB."""
    data = request.get_json()
    if "heading" not in data:
        return jsonify({"error": "Missing 'heading' field"}), 400

    new_heading = data["heading"]
    chat = get_chat(chat_id)

    if not chat:
        return jsonify({"error": f"No chat found with ID '{chat_id}'"}), 404

    logging.info(f"Renaming chat_id: {chat_id} to new heading: {new_heading}")
    rename_chat(chat_id, new_heading)
    return jsonify({"chat_id": chat_id, "user_id": chat["user_id"], "heading": new_heading, "created_at": chat["created_at"]})



@app.route('/get_user_chats/<user_id>', methods=['GET'])
def get_user_chats_route(user_id):
    """Retrieve all chats for a specific user from ChromaDB."""
    logging.info(f"Fetching all chats for user_id: {user_id}")
    chats = get_user_chats(user_id)
    return jsonify({"user_id": user_id, "chats": chats})


@app.route('/delete_user_chats/<user_id>', methods=['DELETE'])
def delete_user_chats_route(user_id):
    """Delete all chats for a user."""
    logging.info(f"Deleting all chats for user_id: {user_id}")
    delete_user(user_id)
    return jsonify({"message": f"All chats deleted for user '{user_id}'"})


@app.route('/store_contexts_from_directory', methods=['POST'])
def store_contexts_route():
    # Define the path to your directory containing JSON files
    directory_path = 'resources/contexts'  # Change this to your actual directory path

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        return jsonify({"error": "Directory not found."}), 404

    logging.info(f"Storing contexts from directory: {directory_path}")
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)

        # Check if the file is a JSON file
        if filename.endswith('.json'):
            # Call the function to store contexts from the file
            result_message = store_contexts_from_file(file_path)

            # If there's an error, return it immediately
            if result_message != "Contexts stored successfully.":
                if result_message == "File not found.":
                    return jsonify({"error": result_message}), 404
                elif result_message == "Invalid JSON format.":
                    return jsonify({"error": result_message}), 400
                else:
                    return jsonify({"error": result_message}), 500

    # If all files are processed successfully
    return jsonify({"message": "All contexts stored successfully."}), 200


@app.route('/delete_all_contexts', methods=['DELETE'])
def delete_all_contexts():
    """Delete all contexts from the collection."""
    logging.info("Deleting all contexts from the collection")
    delete_context_collection()
    return jsonify({"message": "All contexts deleted successfully."})


@app.route('/retrieve_context_from_query', methods=['POST'])
def retrieve_context_from_query_route():
    """Retrieve the most relevant context based on similarity search."""

    data = request.get_json()
    if "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query = data["query"]

    try:
        logging.info(f"Retrieving context for query: {query}")
        context = retrieve_context(query)
        return jsonify({"query": query, "context": context})

    except Exception as e:
        return jsonify({"error": "Failed to retrieve context", "details": str(e)}), 500


# Update Context with URLs and ignored keywords while web scraping
@app.route('/api/update_context_url', methods=['POST'])
def update_context_url():
    try:
        data = request.get_json()

        if "urls" not in data or "ignored_keywords" not in data:
            return jsonify({"error": "'urls' and 'ignored_keywords' fields are required"}), 400

        urls = data["urls"]
        ignored_keywords = data["ignored_keywords"]

        logging.info(f"Updating context URLs: {len(urls)} urls, {len(ignored_keywords)} ignored keywords")

        # Add your processing logic here.

        return jsonify({"message": f"Received {len(urls)} URLs and {len(ignored_keywords)} ignored keywords for processing."})

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/get_context_collection_count', methods=['GET'])
def get_context_collection_count_route():
    """Get the count of contexts in the collection."""
    count = get_context_collection_count()
    return jsonify({"count": count})

@app.route('/debug_context_collection', methods=['GET'])
def debug_context_collection_route():
    """Debug the context collection state and show structure."""
    try:
        from db import context_collection
        count = get_context_collection_count()
        
        # Get query parameters for pagination and detail level
        limit = int(request.args.get('limit', 5))
        offset = int(request.args.get('offset', 0))
        show_full_content = request.args.get('show_content', 'false').lower() == 'true'
        
        # Try to get a sample of items with metadata
        sample_items = context_collection.get(
            limit=limit,
            offset=offset,
            include=["metadatas", "documents"]
        )
        
        # Process the sample items to show structure
        structured_items = []
        if sample_items and sample_items.get('ids'):
            for i, item_id in enumerate(sample_items['ids']):
                metadata = sample_items.get('metadatas', [{}])[i] if i < len(sample_items.get('metadatas', [])) else {}
                document = sample_items.get('documents', [''])[i] if i < len(sample_items.get('documents', [])) else ''
                
                # Parse JSON fields if they exist
                try:
                    ngos = json.loads(metadata.get('ngos', '[]')) if metadata.get('ngos') else []
                except:
                    ngos = metadata.get('ngos', [])
                
                try:
                    sanctuary = json.loads(metadata.get('sanctuary', '[]')) if metadata.get('sanctuary') else []
                except:
                    sanctuary = metadata.get('sanctuary', [])
                
                item_structure = {
                    "id": item_id,
                    "url": metadata.get('url', 'N/A'),
                    "ngos": ngos,
                    "sanctuaries": sanctuary,
                    "content_length": len(metadata.get('content', '')),
                    "document_length": len(document) if document else 0
                }
                
                # Add full content if requested (truncated for readability)
                if show_full_content:
                    content = metadata.get('content', '')
                    item_structure['content_preview'] = content[:500] + "..." if len(content) > 500 else content
                
                structured_items.append(item_structure)
        
        # Get unique organizations and sanctuaries for overview
        all_orgs = set()
        all_sanctuaries = set()
        
        if sample_items and sample_items.get('metadatas'):
            for metadata in sample_items['metadatas']:
                try:
                    ngos = json.loads(metadata.get('ngos', '[]'))
                    if isinstance(ngos, list):
                        all_orgs.update([ngo for ngo in ngos if ngo])
                except:
                    pass
                
                try:
                    sanctuaries = json.loads(metadata.get('sanctuary', '[]'))
                    if isinstance(sanctuaries, list):
                        all_sanctuaries.update([s for s in sanctuaries if s])
                except:
                    pass
        
        return jsonify({
            "collection_info": {
                "total_count": count,
                "collection_exists": context_collection is not None,
                "collection_name": "context_embeddings"
            },
            "query_params": {
                "limit": limit,
                "offset": offset,
                "show_full_content": show_full_content
            },
            "sample_overview": {
                "items_returned": len(structured_items),
                "unique_organizations": list(all_orgs),
                "unique_sanctuaries": list(all_sanctuaries),
                "organization_count": len(all_orgs),
                "sanctuary_count": len(all_sanctuaries)
            },
            "sample_items": structured_items,
            "usage_note": "Use ?limit=N&offset=N for pagination, ?show_content=true to see content previews"
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/context_structure', methods=['GET'])
def get_context_structure():
    """Get detailed structure and statistics of the context database."""
    try:
        from db import context_collection
        
        # Get query parameters
        limit = int(request.args.get('limit', 10))
        search_org = request.args.get('search_org', '')
        search_sanctuary = request.args.get('search_sanctuary', '')
        
        total_count = get_context_collection_count()
        
        # Get all items for analysis (or limited sample for large databases)
        analysis_limit = min(total_count, 100)  # Analyze first 100 items for stats
        all_items = context_collection.get(
            limit=analysis_limit,
            include=["metadatas"]
        )
        
        # Analyze the structure
        all_orgs = set()
        all_sanctuaries = set()
        all_urls = set()
        content_lengths = []
        url_domains = {}
        
        if all_items and all_items.get('metadatas'):
            for metadata in all_items['metadatas']:
                # Extract organizations
                try:
                    ngos = json.loads(metadata.get('ngos', '[]'))
                    if isinstance(ngos, list):
                        all_orgs.update([ngo.strip() for ngo in ngos if ngo and ngo.strip()])
                except:
                    pass
                
                # Extract sanctuaries
                try:
                    sanctuaries = json.loads(metadata.get('sanctuary', '[]'))
                    if isinstance(sanctuaries, list):
                        all_sanctuaries.update([s.strip() for s in sanctuaries if s and s.strip()])
                except:
                    pass
                
                # Extract URLs and domains
                url = metadata.get('url', '')
                if url:
                    all_urls.add(url)
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        url_domains[domain] = url_domains.get(domain, 0) + 1
                    except:
                        pass
                
                # Content length analysis
                content = metadata.get('content', '')
                content_lengths.append(len(content))
        
        # Filter items based on search criteria
        filtered_items = []
        if all_items and all_items.get('ids'):
            for i, item_id in enumerate(all_items['ids'][:limit]):
                metadata = all_items.get('metadatas', [{}])[i] if i < len(all_items.get('metadatas', [])) else {}
                
                # Apply filters
                if search_org:
                    ngos_str = metadata.get('ngos', '[]')
                    if search_org.lower() not in ngos_str.lower():
                        continue
                
                if search_sanctuary:
                    sanctuary_str = metadata.get('sanctuary', '[]')
                    if search_sanctuary.lower() not in sanctuary_str.lower():
                        continue
                
                # Parse and format the item
                try:
                    ngos = json.loads(metadata.get('ngos', '[]'))
                except:
                    ngos = [metadata.get('ngos', '')] if metadata.get('ngos') else []
                
                try:
                    sanctuaries = json.loads(metadata.get('sanctuary', '[]'))
                except:
                    sanctuaries = [metadata.get('sanctuary', '')] if metadata.get('sanctuary') else []
                
                filtered_items.append({
                    "id": item_id,
                    "url": metadata.get('url', 'N/A'),
                    "organizations": [org for org in ngos if org],
                    "sanctuaries": [s for s in sanctuaries if s],
                    "content_length": len(metadata.get('content', '')),
                    "content_preview": metadata.get('content', '')[:200] + "..." if len(metadata.get('content', '')) > 200 else metadata.get('content', '')
                })
        
        # Calculate statistics
        avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        
        return jsonify({
            "database_overview": {
                "total_items": total_count,
                "analyzed_items": analysis_limit,
                "analysis_complete": analysis_limit >= total_count
            },
            "statistics": {
                "total_organizations": len(all_orgs),
                "total_sanctuaries": len(all_sanctuaries),
                "total_unique_urls": len(all_urls),
                "avg_content_length": round(avg_content_length, 2),
                "min_content_length": min(content_lengths) if content_lengths else 0,
                "max_content_length": max(content_lengths) if content_lengths else 0
            },
            "organizations": sorted(list(all_orgs)),
            "sanctuaries": sorted(list(all_sanctuaries)),
            "url_domains": dict(sorted(url_domains.items(), key=lambda x: x[1], reverse=True)),
            "filtered_items": filtered_items,
            "query_info": {
                "limit": limit,
                "search_org": search_org,
                "search_sanctuary": search_sanctuary,
                "results_count": len(filtered_items)
            },
            "usage": {
                "endpoints": {
                    "/context_structure": "Get overview and filtered view of contexts",
                    "/debug_context_collection": "Get raw debug info with pagination"
                },
                "parameters": {
                    "limit": "Number of items to return (default: 10)",
                    "search_org": "Filter by organization name (partial match)",
                    "search_sanctuary": "Filter by sanctuary name (partial match)"
                }
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to analyze context structure: {str(e)}"})


def extract_sources_from_context(context_metadatas):
    """
    Extract unique source URLs from context metadata.
    Returns a list of unique URLs that can be used for source attribution.
    """
    if not context_metadatas:
        return []
    
    sources = []
    seen_urls = set()
    
    for ctx in context_metadatas:
        url = ctx.get('url', '').strip()
        if url and url not in seen_urls:
            sources.append(url)
            seen_urls.add(url)
    
    return sources

def manually_append_sources(answer: str, context_sources: list, question_type: str = None) -> str:
    """
    Manually append sources from context to the answer.
    This ensures we only include actual sources from the retrieved context,
    not hallucinated sources from the AI.
    """
    # Don't add sources for greeting, gratitude, capability, or off-topic questions
    if question_type in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['GRATITUDE'], 
                        QUESTION_TYPES['CAPABILITY'], QUESTION_TYPES['OFF_TOPIC']]:
        return answer
    
    # Don't add sources if no context sources are available
    if not context_sources:
        return answer
    
    # Remove any existing sources that might have been hallucinated by AI
    answer = re.sub(r'\n\s*#+\s*Sources?:?\s*.*$', '', answer, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    answer = re.sub(r'\n\s*Sources?:\s*.*$', '', answer, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    answer = re.sub(r'https?://[^\s,;\)\]\}]+', '', answer)  # Remove any URLs from the answer
    
    # Clean up the answer
    answer = answer.rstrip()
    
    # Add our manual sources section
    if context_sources:
        if answer and not answer.endswith('\n'):
            answer += '\n\n'
        
        answer += "### Sources\n"
        for source in context_sources:
            answer += f"- {source}\n"
    
    return answer

def build_structured_context(context_metadatas):
    """
    Given a list of context metadata dicts, return a clean string with context information.
    Expects each dict to have at least 'content', and optionally 'sanctuary', 'ngos', 'url'.
    Includes additional deduplication to ensure clean output.
    """
    if not context_metadatas:
        return ""
    
    context_blocks = []
    seen_organizations = set()
    
    for ctx in context_metadatas:
        # Build a clean content block
        content_lines = []
        
        # Parse NGOs to check for duplicates
        ngos_data = ctx.get('ngos', '[]')
        try:
            ngos_list = json.loads(ngos_data) if isinstance(ngos_data, str) else ngos_data
            if not isinstance(ngos_list, list):
                ngos_list = [ngos_data] if ngos_data else []
        except (json.JSONDecodeError, TypeError):
            ngos_list = [ngos_data] if ngos_data else []
        
        # Check if this organization info was already included
        current_orgs = [ngo.lower().strip() for ngo in ngos_list if ngo]
        
        # Enhanced duplicate detection: check for partial matches and common abbreviations
        skip_this_context = False
        for current_org in current_orgs:
            for seen_org in seen_organizations:
                # Exact match
                if current_org == seen_org:
                    skip_this_context = True
                    break
                # Check if one is contained in the other (handles abbreviations and variations)
                elif (current_org in seen_org or seen_org in current_org) and len(current_org) > 3 and len(seen_org) > 3:
                    skip_this_context = True
                    break
            if skip_this_context:
                break
                
        if skip_this_context:
            continue  # Skip this context block as organization already covered
        
        # Add sanctuary info if available
        sanctuary_data = ctx.get('sanctuary', '[]')
        try:
            sanctuaries = json.loads(sanctuary_data) if isinstance(sanctuary_data, str) else sanctuary_data
            if not isinstance(sanctuaries, list):
                sanctuaries = [sanctuary_data] if sanctuary_data else []
        except (json.JSONDecodeError, TypeError):
            sanctuaries = [sanctuary_data] if sanctuary_data else []
        
        if sanctuaries and any(s for s in sanctuaries):
            clean_sanctuaries = [s for s in sanctuaries if s]
            content_lines.append(f"Sanctuary: {', '.join(clean_sanctuaries)}")
        
        # Add NGO info if available
        if current_orgs:
            content_lines.append(f"Organization: {', '.join([ngo for ngo in ngos_list if ngo])}")
        
        # Add the main content, but clean it up to remove redundant NGO mentions
        if ctx.get('content'):
            content = ctx['content']
            # Remove the redundant "Name of the NGO:" prefix that might be in stored content
            content = re.sub(r'^.*?Name of the NGO:.*?Content of the full page:\s*', '', content, flags=re.DOTALL)
            content = re.sub(r'^.*?URL to the Content which NGO maintains:.*?Content of the full page:\s*', '', content, flags=re.DOTALL)
            content = re.sub(r'^.*?Sanctuaries which the NGO are maintaining:.*?Content of the full page:\s*', '', content, flags=re.DOTALL)
            content_lines.append(content.strip())
        
        # Add source URL at the end if available
        if ctx.get('url'):
            content_lines.append(f"Source: {ctx['url']}")
        
        # Only add this block if it has meaningful content
        if len(content_lines) > 1:  # More than just URL
            context_blocks.append('\n'.join(content_lines))
            # Mark these organizations as seen
            seen_organizations.update(current_orgs)
    
    return '\n\n---\n\n'.join(context_blocks)

def filter_sources(answer: str, context: str, question_type: str = None, context_sources: list = None) -> str:
    """
    Clean up the AI response by removing any hallucinated sources or unwanted content.
    This function focuses on cleaning rather than adding sources, since we handle
    source addition manually via manually_append_sources().
    """
    # Remove any context structure patterns that might be echoed
    answer = re.sub(r'Context \d+:.*?\|.*?\n', '', answer, flags=re.MULTILINE)
    answer = re.sub(r'Sanctuary: \[.*?\] \| NGOs: \[.*?\] \| Source:.*?\n', '', answer, flags=re.MULTILINE)
    answer = re.sub(r'Name of the NGO:.*?Content of the full page:\s*', '', answer, flags=re.DOTALL)
    
    # Enhanced filtering based on question type
    if question_type in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['GRATITUDE'], QUESTION_TYPES['CAPABILITY']]:
        # For greetings, gratitude, and capability questions, remove any technical content that might have leaked
        answer = re.sub(r'The organization "[^"]*" working at [^.]*\.[^.]*\.', '', answer, flags=re.DOTALL)
        answer = re.sub(r'Organization: [^.]*\n', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'Sanctuary: [^.]*\n', '', answer, flags=re.MULTILINE)
        # Remove specific NGO/sanctuary references
        answer = re.sub(r'[^.]*\b(Kalpavriksh|Bhimashankar|Wildlife Sanctuary|Maharashtra|WWF|Greenpeace|Sangram|Simlipal)\b[^.]*\.', '', answer, flags=re.IGNORECASE)
        # Remove any URLs for these question types
        answer = re.sub(r'https?://[^\s,;\)\]\}]+', '', answer)
        answer = re.sub(r'Sources?:.*?(\n|$)', '', answer, flags=re.IGNORECASE)
    
    elif question_type == QUESTION_TYPES['OFF_TOPIC']:
        # For off-topic questions, keep the response minimal and remove technical details
        answer = re.sub(r'Organization: [^.]*\n', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'Sanctuary: [^.]*\n', '', answer, flags=re.MULTILINE)
        # Remove any sources for off-topic questions
        answer = re.sub(r'Sources?:.*?(\n|$)', '', answer, flags=re.IGNORECASE)
    
    else:
        # For technical questions, remove any hallucinated sources but keep the content
        # Remove specific organizational content that shouldn't appear in capability responses
        if re.search(r'\b(what\s+(can|do)\s+you\s+do|capabilities|functions|help)\b', answer, re.IGNORECASE):
            answer = re.sub(r'The organization "[^"]*" working at [^.]*\. They [^.]*\.', '', answer, flags=re.DOTALL)
            answer = re.sub(r'Organization: [^.]*\n', '', answer, flags=re.MULTILINE)
            answer = re.sub(r'Sanctuary: [^.]*\n', '', answer, flags=re.MULTILINE)
            answer = re.sub(r'[^.]*\b(Kalpavriksh|Bhimashankar|Wildlife Sanctuary|Maharashtra)\b[^.]*\.', '', answer, flags=re.IGNORECASE)
    
        # Remove any AI-generated sources section (we'll add our own manually)
        answer = re.sub(r'\n\s*Sources?:\s*.*$', '', answer, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        # Remove any URLs that might be hallucinated by the AI
        answer = re.sub(r'https?://[^\s,;\)\]\}]+', '', answer)
    
    # Clean up extra whitespace and empty lines
    answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)  # Multiple empty lines to double
    answer = re.sub(r'^\s+|\s+$', '', answer)  # Leading/trailing whitespace
    
    return answer.strip()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify service status"""
    try:
        health_status = model_manager.check_health()
        primary_healthy = any(health_status.values()) if health_status else False
        
        return jsonify({
            "status": "healthy" if primary_healthy else "degraded",
            "providers": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Model services available" if primary_healthy else "Model services unavailable - using fallback responses"
        }), 200 if primary_healthy else 503
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """
    Get API statistics including cache status and request frequency.
    Useful for debugging frontend polling issues.
    """
    current_time = time.time()
    cache_stats = {}
    request_stats = {}
    
    # Calculate cache hit rates and request frequencies
    for key, (data, cache_time) in ENDPOINT_CACHE.items():
        cache_age = current_time - cache_time
        cache_stats[key] = {
            "cache_age_seconds": round(cache_age, 2),
            "is_valid": cache_age < CACHE_DURATION
        }
    
    for key, last_time in LAST_REQUEST_TIME.items():
        time_since_last = current_time - last_time
        request_stats[key] = {
            "last_request_seconds_ago": round(time_since_last, 2),
            "request_frequency": "high" if time_since_last < MIN_REQUEST_INTERVAL else "normal"
        }
    
    return jsonify({
        "cache_stats": cache_stats,
        "request_stats": request_stats,
        "cache_config": {
            "cache_duration": CACHE_DURATION,
            "min_request_interval": MIN_REQUEST_INTERVAL
        },
        "total_cached_endpoints": len(ENDPOINT_CACHE),
        "total_tracked_endpoints": len(LAST_REQUEST_TIME)
    })

@app.route('/model/status', methods=['GET'])
def model_status():
    """Dedicated endpoint for model service status"""
    try:
        health_status = model_manager.check_health()
        primary_provider = model_manager.primary_provider
        fallback_provider = model_manager.fallback_provider
        
        return jsonify({
            "providers_health": health_status,
            "primary_provider": type(primary_provider).__name__.replace('Provider', '').lower() if primary_provider else None,
            "fallback_provider": type(fallback_provider).__name__.replace('Provider', '').lower() if fallback_provider else None,
            "fallback_enabled": model_manager.fallback_provider is not None,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

if __name__ == '__main__':
    # Test logging immediately
    print("🧪 STARTUP: Testing logging configuration...")
    logging.info("🧪 STARTUP: Logging is working correctly!")
    print("🧪 STARTUP: Starting Flask application...")
    app.run(host="localhost", debug=True, port=5000)