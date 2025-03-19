from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import uuid
from datetime import datetime

from utils import retrieve_context

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

CHAT_API_URL = "http://localhost:1234/v1/chat/completions"
MAX_HISTORY = 5  # Configurable conversation history size

# Store multiple chats (chat_id -> {heading, history, created_at})
chats = {}

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
def ask():
    try:
        # Parse input JSON
        data = request.get_json()
        if "question" not in data:
            return jsonify({"error": "Missing 'question' field"}), 400

        user_question = data["question"]
        chat_id = data.get("chat_id")

        # If no chat_id, create a new chat session
        if not chat_id:
            chat_id = str(uuid.uuid4())  # Generate a unique chat ID
            heading = generate_heading_from_mistral(user_question)
            chats[chat_id] = {
                "heading": heading,
                "messages": [],
                "created_at": datetime.utcnow().isoformat()  # Store UTC timestamp
            }

        # Retrieve chat session
        chat = chats.get(chat_id)

        if not chat:
            return jsonify({"error": "Chat ID not found"}), 404

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

        # Get assistant's response
        assistant_reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # Return chat_id, heading, response, and created date
        return jsonify({
            "chat_id": chat_id,
            "heading": chat_heading,
            "created_at": chat["created_at"],
            "messages": conversation_history
        })

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/delete_chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a specific chat by chat_id."""
    data = request.get_json()

    if not chat_id or chat_id not in chats:
        return jsonify({"error": "Chat ID not found"}), 404

    del chats[chat_id]  # Remove chat from storage
    return jsonify({"message": "Chat deleted successfully"})

@app.route('/list_chats', methods=['GET'])
def list_chats():
    try:
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        sort_order = request.args.get("sort_order", "desc").lower()  # Default: descending
        search_query = request.args.get("search", "").strip().lower()  # Case-insensitive search

        # Pagination parameters (Optional)
        page = request.args.get("page")
        limit = request.args.get("limit")

        # Convert to integers if provided
        page = int(page) if page else None
        limit = int(limit) if limit else None

        # Convert start/end dates to datetime objects
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        # Filter chats
        filtered_chats = []
        for chat_id, chat in chats.items():
            chat_dt = datetime.fromisoformat(chat["created_at"])

            # Apply date filters
            if start_dt and chat_dt < start_dt:
                continue
            if end_dt and chat_dt > end_dt:
                continue

            # Apply partial search filter
            if search_query and search_query not in chat["heading"].lower():
                continue

            filtered_chats.append({
                "chat_id": chat_id,
                "heading": chat["heading"],
                "created_at": chat["created_at"]
            })

        # Sort by created_at (asc or desc)
        filtered_chats.sort(key=lambda x: x["created_at"], reverse=(sort_order == "desc"))

        # Apply pagination only if page & limit are provided
        total_chats = len(filtered_chats)
        total_pages = (total_chats + limit - 1) // limit if limit else 1

        if page and limit:
            start_index = (page - 1) * limit
            end_index = start_index + limit
            paginated_chats = filtered_chats[start_index:end_index]
        else:
            paginated_chats = filtered_chats  # Return all chats if no pagination

        return jsonify({
            "page": page if page else None,
            "limit": limit if limit else None,
            "total_chats": total_chats,
            "total_pages": total_pages if limit else 1,
            "chats": paginated_chats
        })

    except Exception as e:
        return jsonify({"error": "Invalid request", "details": str(e)}), 400

@app.route('/get_chat/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Retrieve a chat by chat_id."""
    chat = chats.get(chat_id)

    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    return jsonify({
        "chat_id": chat_id,
        "heading": chat["heading"],
        "messages": chat["messages"],
        "created_at": chat["created_at"]
    })


@app.route('/rename_chat/<chat_id>', methods=['POST'])
def rename_chat(chat_id):
    """Rename a chat by chat_id."""
    chat = chats.get(chat_id)

    data = request.get_json()
    if "heading" not in data:
        return jsonify({"error": "Missing 'heading' field"}), 400

    new_heading = data["heading"]

    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    chat["heading"] = new_heading

    return jsonify({
        "chat_id": chat_id,
        "heading": chat["heading"],
        "messages": chat["messages"],
        "created_at": chat["created_at"]
    })


if __name__ == '__main__':
    app.run(host="localhost", debug=True, port=5000)