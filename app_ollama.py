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

from utils import retrieve_context
from db import store_chat, get_chat, list_chats, delete_chat, store_user, get_user, delete_user, store_context, retrieve_context, get_user_chats, rename_chat, \
    list_users, store_contexts_from_file, delete_context_collection, get_context_collection_count, list_links, add_link, delete_link, get_structured_context
    
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"

FRONTEND_URL = "http://localhost:5173"  # Update this to your frontend URL if different
LMSTUDIO_CHAT_API_URL = "http://localhost:1234/v1/chat/completions"
MAX_HISTORY = 5  # Configurable conversation history size

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": FRONTEND_URL}}, supports_credentials=True)

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

# Default system contexts
system_contexts = [
    {"role": "system", "content": """
[General Role]
You are an AI assistant focused on wildlife, biodiversity, conservation, sanctuaries, and related topics in India.

[Formatting Instructions]
- Use bullet points or numbered lists when possible.
- Double-check your formatting before responding.

[Domain Restrictions]
Only answer questions related to wildlife, biodiversity, conservation, sanctuaries, and similar topics in India. If a question is about wildlife, biodiversity, or conservation in other countries, politely respond that you can only answer questions about India unless the user explicitly asks for information about other countries.

[Question Type Handling]
- GREETING questions: Respond warmly and briefly explain your capabilities without using any provided context.
- CAPABILITY questions: Clearly explain your role and abilities related to wildlife, biodiversity, and conservation in India without using any provided context.
- TECHNICAL questions (wildlife/conservation): Use provided context when available to give detailed, accurate answers.
- OFF_TOPIC questions: Politely redirect the conversation back to wildlife, conservation, or environmental topics in India.

[NGO Mention Policy]
Only mention NGOs provided in the context and related to wildlife sanctuaries.

[Source Attribution]
Only cite sources that are present in the provided context and directly support your answer. If no relevant sources are available in the context, do not mention any sources or links.

[Table Formatting]
Summarize information in tables whenever possible for technical questions.

[Directness and Response Style]
- Do not use phrases like 'based on the context' or 'as mentioned in the context.' Just provide the answer directly.
- NEVER echo or repeat the reference material structure in your response.
- Do not include phrases like "Context 1:", "Context 2:", etc. in your answers.
- Do not repeat the detailed formatting from reference materials.
- Extract and synthesize information naturally without referencing how it was provided to you.
- Focus on answering the user's question directly using the information available.
- For greetings and capability questions, do not use any technical context even if provided.
"""}
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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
def get_user_details_route(user_id):
    logging.info(f"Fetching user details for user_id: {user_id}")
    user_details = get_user(user_id)
    return jsonify(user_details)

def build_ollama_prompt(system_contexts, rag_context, conversation_history, question_type=None):
    # Concatenate system instructions, RAG context, and chat history into a single prompt string
    prompt_parts = []
    for ctx in system_contexts:
        prompt_parts.append(ctx["content"])
    
    # Include context based on question type
    if rag_context.strip() and question_type not in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['CAPABILITY']]:
        if question_type == QUESTION_TYPES['WILDLIFE_TECHNICAL']:
            prompt_parts.append("Use the following wildlife and sanctuary information to answer the question:")
        elif question_type == QUESTION_TYPES['CONSERVATION_TECHNICAL']:
            prompt_parts.append("Use the following conservation and environmental information to answer the question:")
        else:
            prompt_parts.append("Use the following reference information to answer questions when relevant:")
        
        prompt_parts.append(rag_context)
        prompt_parts.append("Now, please answer the user's question based on this information without mentioning the reference material or context explicitly.")
    elif question_type == QUESTION_TYPES['GREETING']:
        prompt_parts.append("This is a greeting. Respond warmly and briefly explain your capabilities related to wildlife and conservation in India. Do not reference any previous conversation topics.")
    elif question_type == QUESTION_TYPES['CAPABILITY']:
        prompt_parts.append("This is a question about your capabilities. Explain what you can help with regarding wildlife, biodiversity, conservation, and sanctuaries in India. Do not reference any previous conversation topics.")
    elif question_type == QUESTION_TYPES['OFF_TOPIC']:
        prompt_parts.append("This question is off-topic. Politely redirect to wildlife, conservation, or environmental topics in India.")
    
    # Handle conversation history based on question type
    if question_type in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['CAPABILITY']]:
        # For greetings and capability questions, limit conversation history to avoid influence from technical discussions
        # Only include the last greeting/capability exchange if any, or start fresh
        filtered_history = filter_conversation_history_for_non_technical(conversation_history)
        for msg in filtered_history:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
    else:
        # For technical questions, include full conversation history for context
        for msg in conversation_history:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
    
    prompt_parts.append("Assistant:")  # Prompt Ollama to generate the next assistant reply
    return "\n".join(prompt_parts)

def filter_conversation_history_for_non_technical(conversation_history):
    """
    For greeting/capability questions, filter conversation history to avoid technical influence.
    Keep only the most recent non-technical exchanges or start fresh.
    """
    if not conversation_history:
        return []
    
    # Look for the most recent greeting/capability exchange
    filtered_history = []
    
    # Go through history in reverse to find the last greeting/capability interaction
    for i in range(len(conversation_history) - 1, -1, -1):
        msg = conversation_history[i]
        if msg["role"] == "user":
            # Classify this historical question
            historical_classification = intelligent_question_classifier(msg["content"])
            historical_question_type = historical_classification['type'] if isinstance(historical_classification, dict) else historical_classification
            if historical_question_type in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['CAPABILITY']]:
                # Found a greeting/capability question, include this exchange
                if i + 1 < len(conversation_history):  # Check if there's a response
                    filtered_history = [conversation_history[i], conversation_history[i + 1]]
                else:
                    filtered_history = [conversation_history[i]]
                break
    
    # If no recent greeting/capability found, start with empty history for clean response
    return filtered_history

def correct_typo_from_mistral(prompt):
    typo_prompt = f"Correct the typo in the prompt if any available. If there are no typo, you can return the same prompt: {prompt}"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": typo_prompt
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        return prompt

def generate_heading_from_mistral(prompt):
    heading_prompt = f"Generate a short title for this conversation: {prompt}"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": heading_prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        heading = response.json().get("response", "").strip()
        # Remove surrounding single or double quotes if present
        if (heading.startswith('"') and heading.endswith('"')) or (heading.startswith("'") and heading.endswith("'")):
            heading = heading[1:-1].strip()
        return heading
    else:
        return "General Conversation"

# Question classification constants
QUESTION_TYPES = {
    'GREETING': 'greeting',
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
    1. GREETING - Greetings, thanks, or general pleasantries
    2. CAPABILITY - Questions about what the assistant can do, its features, or capabilities
    3. WILDLIFE_TECHNICAL - Questions about specific animals, species, habitats, behavior, or wildlife-related topics
    4. CONSERVATION_TECHNICAL - Questions about conservation efforts, protected areas, environmental protection, or sustainability
    5. GENERAL_ENVIRONMENTAL - General environmental questions that may need some context
    6. OFF_TOPIC - Questions completely unrelated to wildlife, conservation, or environment

    Question: "{question}"
    
    Respond in this exact format:
    PRIMARY: [category]
    CONFIDENCE: [0-100]
    MIXED_INTENT: [yes/no]
    SECONDARY: [category1,category2] (only if mixed intent is yes)"""
    
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": classification_prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            response_text = response.json().get("response", "").strip()
            return parse_enhanced_classification(response_text)
        else:
            return create_default_classification()
    except Exception as e:
        logging.warning(f"AI classification failed: {e}")
        return create_default_classification()

def parse_enhanced_classification(response_text):
    """Parse the enhanced AI classification response"""
    classification_map = {
        'GREETING': QUESTION_TYPES['GREETING'],
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

def is_obvious_technical(text):
    """Fast pattern matching for obvious technical questions"""
    technical_keywords = [
        'tiger', 'elephant', 'leopard', 'sanctuary', 'national park', 'wildlife',
        'conservation', 'biodiversity', 'species', 'habitat', 'ecosystem',
        'forest', 'reserve', 'protection', 'endangered', 'ngo', 'organization'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in technical_keywords)

def intelligent_question_classifier(question, conversation_history=None):
    """Enhanced hybrid approach with mixed intent detection and pronoun resolution"""
    
    # Step 1: Basic pronoun resolution if conversation history available
    resolved_question = question
    if conversation_history:
        resolved_question = resolve_basic_pronouns(question, conversation_history)
        if resolved_question != question:
            logging.info(f"Pronoun resolution: '{question}' -> '{resolved_question}'")
    
    # Step 2: Fast checks first (using original question for pattern matching)
    if is_obvious_greeting(question):
        return {
            'type': QUESTION_TYPES['GREETING'],
            'confidence': 95,
            'is_mixed': False,
            'context_hint': 'fast_greeting'
        }
    
    if is_obvious_capability(question):
        return {
            'type': QUESTION_TYPES['CAPABILITY'],
            'confidence': 95,
            'is_mixed': False,
            'context_hint': 'fast_capability'
        }
    
    if is_obvious_technical(question):
        # Use AI for technical subcategorization
        ai_result = classify_question_with_ai(resolved_question)
        if ai_result['primary_type'] in [QUESTION_TYPES['WILDLIFE_TECHNICAL'], QUESTION_TYPES['CONSERVATION_TECHNICAL']]:
            return {
                'type': ai_result['primary_type'],
                'confidence': min(ai_result['confidence'], 90),  # Cap confidence for pattern-detected technical
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
    
    # Step 3: For ambiguous cases, use full AI classification
    ai_result = classify_question_with_ai(resolved_question)
    return {
        'type': ai_result['primary_type'],
        'confidence': ai_result['confidence'],
        'is_mixed': ai_result['is_mixed'],
        'context_hint': 'ai_classification'
    }

def is_obvious_greeting(text):
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

def is_obvious_technical(text):
    """Fast pattern matching for obvious technical questions"""
    technical_keywords = [
        'tiger', 'elephant', 'leopard', 'sanctuary', 'national park', 'wildlife',
        'conservation', 'biodiversity', 'species', 'habitat', 'ecosystem',
        'forest', 'reserve', 'protection', 'endangered', 'ngo', 'organization'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in technical_keywords)

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
    return question_type in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['CAPABILITY']]


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
        user_question = request.args.get("question")
        user_id = request.args.get("user_id")
        chat_id = request.args.get("chat_id")

        if not user_question or not user_id:
            return Response("data: " + json.dumps({"error": "'question' and 'user_id' are required"}) + "\n\n", content_type="text/event-stream")

        logging.info(f"Streaming question from user_id: {user_id}, chat_id: {chat_id}, question: {user_question}")

        chat = get_chat(chat_id) if chat_id else None

        if not chat:
            chat_id = str(uuid.uuid4())
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

        # Enhanced intelligent question classification with mixed intent detection
        classification_result = intelligent_question_classifier(user_question, conversation_history[:-1])
        question_type = classification_result['type'] if isinstance(classification_result, dict) else classification_result
        
        # Log enhanced classification details
        if isinstance(classification_result, dict):
            logging.info(f"Question classified as: {question_type} (confidence: {classification_result['confidence']}%, mixed: {classification_result['is_mixed']}) for question: {user_question}")
            if classification_result.get('context_hint'):
                logging.info(f"Classification method: {classification_result['context_hint']}")
        else:
            logging.info(f"Question classified as: {question_type} for question: {user_question}")
        
        if should_use_context(classification_result):
            # For technical questions, build retrieval query normally
            retrieval_query = build_retrieval_query(conversation_history[:-1], user_question, max_turns=MAX_HISTORY)
            context_metadatas = get_structured_context(retrieval_query)
            rag_context = build_structured_context(context_metadatas)
            logging.info(f"Retrieved context for {question_type} question, context length: {len(rag_context)}")
        else:
            # For greetings/capability questions, don't retrieve any context
            rag_context = ""
            logging.info(f"Skipped context retrieval for {question_type} question")

        prompt = build_ollama_prompt(system_contexts, rag_context, conversation_history, question_type)

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True
        }

        def generate():
            buffer = ""
            try:
                with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'error': 'Failed to get response', 'details': response.text})}\n\n"
                        return

                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                data = json.loads(line)
                                token = data.get("response", "")
                                buffer += token
                                yield f"data: {json.dumps({'content': token})}\n\n"
                            except json.JSONDecodeError:
                                continue

                # Filter sources in the final answer using the context and question type
                filtered_buffer = filter_sources(buffer, rag_context, question_type)
                conversation_history.append({"role": "assistant", "content": filtered_buffer})
                store_chat(user_id, chat_id, chat["heading"], conversation_history)

                logging.info(f"Ollama streaming response for user_id: {user_id}, chat_id: {chat_id}, length: {len(filtered_buffer)}")

            except Exception as e:
                yield f"data: {json.dumps({'error': 'Internal server error', 'details': str(e)})}\n\n"

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
def list_chats_route():
    """List all chats for a user with filtering, sorting, search, and pagination."""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "'user_id' is required"}), 400

        # Get optional query parameters safely
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        sort_order = request.args.get("sort_order", "desc").lower()
        search_query = request.args.get("search", "").strip()

        # Ensure `page` and `limit` are integers and handle invalid values
        try:
            page = int(request.args.get("page", 1) or 1)
            limit = int(request.args.get("limit", 20) or 20)
        except ValueError:
            return jsonify({"error": "Invalid 'page' or 'limit' value"}), 400

        logging.info(f"Listing chats for user_id: {user_id}, page: {page}, limit: {limit}")

        # Fetch filtered, sorted, paginated chats
        chat_data = list_chats(user_id, start_date, end_date, sort_order, search_query, page, limit)

        return jsonify({
            "user_id": user_id,
            "page": page,
            "limit": limit,
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


def build_structured_context(context_metadatas):
    """
    Given a list of context metadata dicts, return a clean string with context information.
    Expects each dict to have at least 'content', and optionally 'sanctuary', 'ngos', 'url'.
    """
    if not context_metadatas:
        return ""
    
    context_blocks = []
    for ctx in context_metadatas:
        # Build a clean content block
        content_lines = []
        
        # Add sanctuary info if available
        if ctx.get('sanctuary'):
            sanctuaries = ctx['sanctuary'] if isinstance(ctx['sanctuary'], list) else [ctx['sanctuary']]
            content_lines.append(f"Sanctuary: {', '.join(sanctuaries)}")
        
        # Add NGO info if available
        if ctx.get('ngos'):
            ngos = ctx['ngos'] if isinstance(ctx['ngos'], list) else [ctx['ngos']]
            content_lines.append(f"Organization: {', '.join(ngos)}")
        
        # Add the main content
        if ctx.get('content'):
            content_lines.append(ctx['content'])
        
        # Add source URL at the end if available
        if ctx.get('url'):
            content_lines.append(f"Source: {ctx['url']}")
        
        context_blocks.append('\n'.join(content_lines))
    
    return '\n\n---\n\n'.join(context_blocks)

def filter_sources(answer: str, context: str, question_type: str = None) -> str:
    """
    Removes any sources/links from the answer that are not present in the context.
    If no valid sources remain, removes the sources section entirely.
    Handles URLs with trailing punctuation and both http/https links.
    Also removes any context structure that might have been echoed by the AI.
    Enhanced with question type awareness for better filtering.
    """
    # Remove any context structure patterns that might be echoed
    answer = re.sub(r'Context \d+:.*?\|.*?\n', '', answer, flags=re.MULTILINE)
    answer = re.sub(r'Sanctuary: \[.*?\] \| NGOs: \[.*?\] \| Source:.*?\n', '', answer, flags=re.MULTILINE)
    answer = re.sub(r'Name of the NGO:.*?Content of the full page:\s*', '', answer, flags=re.DOTALL)
    
    # Enhanced filtering based on question type
    if question_type in [QUESTION_TYPES['GREETING'], QUESTION_TYPES['CAPABILITY']]:
        # For greetings and capability questions, remove any technical content that might have leaked
        answer = re.sub(r'The organization "[^"]*" working at [^.]*\.[^.]*\.', '', answer, flags=re.DOTALL)
        answer = re.sub(r'Organization: [^.]*\n', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'Sanctuary: [^.]*\n', '', answer, flags=re.MULTILINE)
        # Remove specific NGO/sanctuary references
        answer = re.sub(r'[^.]*\b(Kalpavriksh|Bhimashankar|Wildlife Sanctuary|Maharashtra|WWF|Greenpeace)\b[^.]*\.', '', answer, flags=re.IGNORECASE)
        # Remove any URLs for these question types
        answer = re.sub(r'https?://[^\s,;\)\]\}]+', '', answer)
        answer = re.sub(r'Sources?:.*?(\n|$)', '', answer, flags=re.IGNORECASE)
    
    elif question_type == QUESTION_TYPES['OFF_TOPIC']:
        # For off-topic questions, keep the response minimal and remove technical details
        answer = re.sub(r'Organization: [^.]*\n', '', answer, flags=re.MULTILINE)
        answer = re.sub(r'Sanctuary: [^.]*\n', '', answer, flags=re.MULTILINE)
    
    else:
        # For technical questions, apply normal source filtering
        # Remove specific organizational content that shouldn't appear in capability responses
        if re.search(r'\b(what\s+(can|do)\s+you\s+do|capabilities|functions|help)\b', answer, re.IGNORECASE):
            answer = re.sub(r'The organization "[^"]*" working at [^.]*\. They [^.]*\.', '', answer, flags=re.DOTALL)
            answer = re.sub(r'Organization: [^.]*\n', '', answer, flags=re.MULTILINE)
            answer = re.sub(r'Sanctuary: [^.]*\n', '', answer, flags=re.MULTILINE)
            answer = re.sub(r'[^.]*\b(Kalpavriksh|Bhimashankar|Wildlife Sanctuary|Maharashtra)\b[^.]*\.', '', answer, flags=re.IGNORECASE)
    
        # Find all URLs in the answer (http or https, ignore trailing punctuation)
        urls_in_answer = re.findall(r'https?://[^\s,;\)\]\}]+', answer)
        # Normalize URLs by stripping trailing punctuation
        def clean_url(url):
            return url.rstrip('.,;:!?)]}')
        cleaned_urls_in_answer = [clean_url(url) for url in urls_in_answer]
        valid_urls = [url for url in cleaned_urls_in_answer if url in context]
        # If there are valid URLs, keep only those in the answer
        if valid_urls:
            for orig_url, clean_url_val in zip(urls_in_answer, cleaned_urls_in_answer):
                if clean_url_val not in valid_urls:
                    answer = answer.replace(orig_url, '')
        else:
            # Remove any 'Sources:' or similar sections if no valid URLs
            answer = re.sub(r'Sources?:.*?(\n|$)', '', answer, flags=re.IGNORECASE)
    
    # Clean up extra whitespace and empty lines
    answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)  # Multiple empty lines to double
    answer = re.sub(r'^\s+|\s+$', '', answer)  # Leading/trailing whitespace
    
    return answer.strip()

if __name__ == '__main__':
    app.run(host="localhost", debug=True, port=5000)