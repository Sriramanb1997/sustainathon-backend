from flask import Flask, session, request, jsonify, url_for, redirect, Response
from flask_cors import CORS
import requests
import os
import uuid
from datetime import datetime
from authlib.integrations.flask_client import OAuth
import json
import time

from utils import retrieve_context
from db import store_chat, get_chat, list_chats, delete_chat, store_user, get_user, delete_user, store_context, retrieve_context, get_user_chats, rename_chat, \
    list_users, store_contexts_from_file, delete_context_collection, get_context_collection_count, list_links, add_link, delete_link
    
FRONTEND_URL = "http://localhost:5173"  # Update this to your frontend URL if different
CHAT_API_URL = "http://localhost:1234/v1/chat/completions"
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
    {"role": "system", "content": "You are a helpful and resourceful AI assistant."},
    # {"role": "system", "content": "You must answer the query only from the context provided to you. Do not use external information. Contexts are provided to you in order of highest importance. Prioritize them while answering in order. But this is not applicable to casual greetings and others. But once the user starts asking questions, you must follow these rules."},
    {"role": "system", "content": "Contexts are web scraped texts and will contain some words which are irrelevant and repetitive. Please ignore them while understanding the context."},
    {"role": "system", "content": "Provide bullet points or numbers wherever possible. Use markdown formatting for better readability."},
    {"role": "system", "content": "You should answer only related to questions on wildlife, biodiversity, conservation, sanctuaries, and related topics across India."},
    {"role": "system", "content": "If the user greets or provides friendly gestures like hello and thanks, You must not consider the context. You can respond with a friendly gesture with what all you are capable of doing."},
    {"role": "system", "content": "When mentioning NGOs, you must consider only NGOs provided in the context which is related to wildlife sanctuaries. Do not consider any other NGOs."},
    {"role": "system", "content": "Always provide sources at the end. Sources should be only from the context provided to you. Mention the source name and link from the link which is given to you in the context."},
    {"role": "system", "content": "Do not mention 'based on the context', or 'As mentioned in the context', etc. Just provide the answer."},
    {"role": "system", "content": "Summarize using tabular format wherever possible. Use markdown formatting for better readability."},
]

@app.route('/login')
def login():
    redirect_uri = url_for('authorize', _external=True)
    print(redirect_uri)
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

    # Redirect the user to frontend with user info or token
    return redirect(f'{FRONTEND_URL}/chat?user_id={user_id}')


@app.route('/user/<user_id>', methods=['GET'])
def get_user_details_route(user_id):
    user_details = get_user(user_id)
    return jsonify(user_details)


def correct_typo_from_mistral(prompt):
    """Ask Mistral to return corrected typo based on the first prompt."""
    typo_prompt = f"Correct the typo in the prompt if any available. If there are no typo, you can return the same prompt: {prompt}"
    payload = {
        "model": "mistral",
        "messages": [{"role": "user", "content": typo_prompt}]
    }

    response = requests.post(CHAT_API_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    else:
        return prompt


def generate_heading_from_mistral(prompt):
    """Ask Mistral to generate a chat heading based on the first prompt."""
    heading_prompt = f"Generate a short title for this conversation. Do not add any single or double quotes: {prompt}"

    payload = {
        "model": "mistral",
        "messages": [{"role": "user", "content": heading_prompt}]
    }

    response = requests.post(CHAT_API_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    else:
        return "General Conversation"


@app.route('/ask', methods=['POST'])
def ask_question_route():
    try:
        data = request.get_json()
        args = request.args.to_dict()

        if "question" not in data or "user_id" not in args:
            return jsonify({"error": "'question' and 'user_id' fields are required"}), 400

        user_question = data["question"]
        user_id = args["user_id"]
        chat_id = data.get("chat_id")

        # Fetch existing chat or create a new one
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

        # Retrieve chat heading & history
        chat_heading = chat["heading"]
        conversation_history = chat["messages"]

        # Add user's question to chat history
        conversation_history.append({"role": "user", "content": user_question})

        # Retrieve RAG context
        rag_context = retrieve_context(user_question)

        # Prepare API request payload
        payload = {
            "model": "mistral",
            "messages": system_contexts + [{"role": "system", "content": "Context for the query :" + rag_context}] + conversation_history
        }

        # Call LM Studio API
        response = requests.post(CHAT_API_URL, json=payload)

        if response.status_code != 200:
            return jsonify({"error": "Failed to get response", "details": response.text}), response.status_code

        # Debugging: Print API Response
        response_json = response.json()
        print("LM Studio Response:", response_json)

        # Validate response structure
        choices = response_json.get("choices", [])
        if not choices or not isinstance(choices, list) or "message" not in choices[0]:
            return jsonify({"error": "Invalid response from model", "details": str(response_json)}), 500

        assistant_reply = choices[0]["message"].get("content", "")

        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        store_chat(user_id, chat_id, chat["heading"], conversation_history)

        return jsonify({
            "chat_id": chat_id,
            "user_id": user_id,
            "heading": chat_heading,
            "created_at": chat["created_at"],
            "messages": conversation_history
        })

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route('/ask_with_stream', methods=['GET'])
def ask_question_with_stream_route():
    try:
        user_question = request.args.get("question")
        user_id = request.args.get("user_id")
        chat_id = request.args.get("chat_id")

        if not user_question or not user_id:
            return Response("data: " + json.dumps({"error": "'question' and 'user_id' are required"}) + "\n\n", content_type="text/event-stream")

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

        rag_context = retrieve_context(user_question)

        payload = {
            "model": "mistral",
            "messages": system_contexts + [{"role": "system", "content": "Context for the query: " + rag_context}] + conversation_history,
            "stream": True
        }

        def generate():
            buffer = ""
            try:
                with requests.post(CHAT_API_URL, json=payload, stream=True) as response:
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'error': 'Failed to get response', 'details': response.text})}\n\n"
                        return

                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            decoded_line = line.strip()
                            if decoded_line.startswith("data: "):
                                decoded_line = decoded_line[6:].strip()

                            try:
                                data = json.loads(decoded_line)
                                token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")

                                if token.strip():
                                    buffer += token
                                    yield f"data: {json.dumps({'content': token})}\n\n"

                            except json.JSONDecodeError:
                                continue

                conversation_history.append({"role": "assistant", "content": buffer})
                store_chat(user_id, chat_id, chat["heading"], conversation_history)

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

    delete_chat(chat_id)
    return jsonify({"message": "Chat deleted successfully"})


@app.route('/list_users', methods=['GET'])
def list_users_route():
    """Retrieve all users."""
    try:
        users = list_users()
        return jsonify({"users": users})

    except Exception as e:
        return jsonify({"error": "Failed to retrieve users", "details": str(e)}), 500

@app.route('/links', methods=['GET'])
def list_links_api():
    """Retrieve all users."""
    try:
        links = list_links()
        return jsonify({"links": links})

    except Exception as e:
        return jsonify({"error": "Failed to retrieve links", "details": str(e)}), 500

@app.route('/link', methods=['POST'])
def add_link_api():
    """Retrieve all users."""
    try:
        data = request.get_json()
        add_link(data)
        return jsonify({"status": "ok"})

    except Exception as e:
        return jsonify({"error": "Failed to save link", "details": str(e)}), 500

@app.route('/link/<link_id>', methods=['DELETE'])
def delete_link_api(link_id):
    """Retrieve all users."""
    try:
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
            limit = int(request.args.get("limit", 2000) or 2000)
        except ValueError:
            return jsonify({"error": "Invalid 'page' or 'limit' value"}), 400

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

    rename_chat(chat_id, new_heading)
    return jsonify({"chat_id": chat_id, "user_id": chat["user_id"], "heading": new_heading, "created_at": chat["created_at"]})



@app.route('/get_user_chats/<user_id>', methods=['GET'])
def get_user_chats_route(user_id):
    """Retrieve all chats for a specific user from ChromaDB."""
    chats = get_user_chats(user_id)
    return jsonify({"user_id": user_id, "chats": chats})


@app.route('/delete_user_chats/<user_id>', methods=['DELETE'])
def delete_user_chats_route(user_id):
    """Delete all chats for a user."""
    delete_user(user_id)
    return jsonify({"message": f"All chats deleted for user '{user_id}'"})


@app.route('/store_contexts_from_directory', methods=['POST'])
def store_contexts_route():
    # Define the path to your directory containing JSON files
    directory_path = 'resources/contexts'  # Change this to your actual directory path

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        return jsonify({"error": "Directory not found."}), 404

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

        # Add your processing logic here.

        return jsonify({"message": f"Received {len(urls)} URLs and {len(ignored_keywords)} ignored keywords for processing."})

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/get_context_collection_count', methods=['GET'])
def get_context_collection_count_route():
    """Get the count of contexts in the collection."""
    count = get_context_collection_count()
    return jsonify({"count": count})


if __name__ == '__main__':
    app.run(host="localhost", debug=True, port=5000)