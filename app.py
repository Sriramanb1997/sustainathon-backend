from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import uuid
from datetime import datetime

from utils import retrieve_context
from db import store_chat, get_chat, list_chats, delete_chat, store_user, get_user, delete_user, store_context, retrieve_context, get_user_chats, rename_chat

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

CHAT_API_URL = "http://localhost:1234/v1/chat/completions"
MAX_HISTORY = 5  # Configurable conversation history size

# Default system contexts
system_contexts = [
    {"role": "system", "content": "You are a helpful and resourceful AI assistant."},
    {"role": "system", "content": "Provide bullet points or numbers wherever possible. Use markdown formatting for better readability."},
    {"role": "system", "content": "You should answer only related to questions on wildlife, biodiversity, conservation and related topics across India."},
    {"role": "system", "content": "Always provide sources at the end. Prioritize sourcing research papers and trusted blog sources. No need to provide link to the sources, just mention the source name."},
    {"role": "system", "content": "Summarize using tabular format wherever possible. Use markdown formatting for better readability."},
]


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
            "messages": system_contexts + [{"role": "system", "content": rag_context}] + conversation_history
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


@app.route('/delete_chat/<chat_id>', methods=['DELETE'])
def delete_chat_route(chat_id):
    """Delete a specific chat."""
    data = request.args.to_dict()
    if "user_id" not in data:
        return jsonify({"error": "'user_id' field is required"}), 400

    delete_chat(chat_id)
    return jsonify({"message": "Chat deleted successfully"})


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
            limit = int(request.args.get("limit", 10) or 10)
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
    return jsonify(chat) if chat else jsonify({"error": "Chat not found"}), 404


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


if __name__ == '__main__':
    app.run(host="localhost", debug=True, port=5000)