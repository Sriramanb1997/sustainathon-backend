import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
import json
import re

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create collections
chats_collection = chroma_client.get_or_create_collection(name="chats")
users_collection = chroma_client.get_or_create_collection(name="users")
context_collection = chroma_client.get_or_create_collection(name="context_embeddings")
link_collection = chroma_client.get_or_create_collection(name="links")

# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def store_chat(user_id, chat_id, heading, messages):
    """Store or update a chat session in ChromaDB."""
    existing_chat = chats_collection.get(ids=[chat_id])

    if existing_chat and existing_chat.get("metadatas"):
        chats_collection.update(
            ids=[chat_id],
            metadatas=[{"chat_id": chat_id, "user_id": user_id, "heading": heading, "created_at": datetime.utcnow().isoformat()}],
            documents=[str(messages)]
        )
    else:
        chats_collection.add(
            ids=[chat_id],
            metadatas=[{"chat_id": chat_id, "user_id": user_id, "heading": heading, "created_at": datetime.utcnow().isoformat()}],
            documents=[str(messages)]
        )



def get_chat(chat_id):
    """Retrieve a chat session from ChromaDB."""
    chat = chats_collection.get(ids=[chat_id])

    if not chat or "metadatas" not in chat or not chat["metadatas"]:
        return None

    metadata = chat["metadatas"][0]
    documents = chat.get("documents", [])

    messages = []

    if documents:
        raw_data = documents[0]

        try:
            # Try parsing normally
            fixed_data = re.sub(r"(?<=\{|\[|\s)'|(?<=,)'|'(?=\s|:|,|]|})", '"', raw_data)
            messages = json.loads(fixed_data)
        except json.JSONDecodeError:
            messages = []  # If decoding still fails, return empty list


    return {
        "chat_id": chat_id,
        "user_id": metadata.get("user_id", ""),
        "heading": metadata.get("heading", "Untitled Chat"),
        "created_at": metadata.get("created_at", ""),
        "messages": messages
    }


def list_chats(user_id, start_date=None, end_date=None, sort_order="desc", search_query=None, page=1, limit=10):
    """Retrieve user chats with filtering, sorting, and pagination directly in ChromaDB."""
    try:
        results = chats_collection.get(where={"user_id": user_id})

        if not results or "metadatas" not in results or not results["metadatas"]:
            return {"total_chats": 0, "total_pages": 1, "chats": []}

        # Convert to list of dicts
        chats = [
            {
                "chat_id": res["chat_id"],
                "user_id": res["user_id"],
                "heading": res["heading"],
                "created_at": res["created_at"]
            }
            for res in results["metadatas"]
        ]

        # Convert dates for filtering
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        # Apply filters
        filtered_chats = [
            chat for chat in chats
            if (not start_dt or datetime.fromisoformat(chat["created_at"]) >= start_dt)
               and (not end_dt or datetime.fromisoformat(chat["created_at"]) <= end_dt)
               and (not search_query or search_query.lower() in chat["heading"].lower())
        ]

        # Sort chats
        filtered_chats.sort(key=lambda x: x["created_at"], reverse=(sort_order == "desc"))

        # Pagination
        total_chats = len(filtered_chats)
        total_pages = max((total_chats + limit - 1) // limit, 1)
        paginated_chats = filtered_chats[(page - 1) * limit: page * limit]

        return {
            "total_chats": total_chats,
            "total_pages": total_pages,
            "chats": paginated_chats
        }

    except Exception as e:
        print(f"Error listing chats for user {user_id}: {str(e)}")
        return {"total_chats": 0, "total_pages": 1, "chats": []}



def delete_chat(chat_id):
    """Delete a chat session."""
    chats_collection.delete(ids=[chat_id])


def store_user(user_id, user_data):
    users_collection.add(
        ids=[user_id],
        metadatas=[{
            "first_name": user_data.get("first_name", ""),
            "last_name": user_data.get("last_name", ""),
            "email": user_data.get("email", ""),
            "user_id": user_id,
            "profile_picture": user_data.get("profile_picture", "")
        }],
        documents=["User Data"]
    )


def list_users():
    """Retrieve all users from the users collection."""
    try:
        results = users_collection.get()
        if not results or "metadatas" not in results or not results["metadatas"]:
            return []

        # Convert to list of user dicts
        users = [
            {
                "user_id": res["user_id"],
                "first_name": res.get("first_name", ""),
                "last_name": res.get("last_name", ""),
                "email": res.get("email", ""),
                "profile_picture": res.get("profile_picture", "")
            }
            for res in results["metadatas"]
        ]

        return users

    except Exception as e:
        print(f"Error retrieving users: {str(e)}")
        return []


def get_user(user_id):
    try:
        user = users_collection.get(ids=[user_id])
        return user.get("metadatas", [{}])[0]
    except Exception as e:
        print(f"Error retrieving user {user_id}: {str(e)}")
        return None

def delete_user(user_id):
    users_collection.delete(ids=[user_id])
    chats_collection.delete(where={"user_id": user_id})

def list_links():
    """Retrieve all users from the users collection."""
    try:
        results = link_collection.get()
        return results["documents"]

    except Exception as e:
        print(f"Error retrieving users: {str(e)}")
        return []

def add_link(link):
    try:
        link_collection.add(
            ids=[str(uuid.uuid4())],
            metadatas=[{"created_at": datetime.utcnow().isoformat()}],
            documents=[link]
        )
    except Exception as e:
        print(f"Error adding link - ", str(e))
        return None

def delete_link(link_id):
    link_collection.delete(ids=[link_id])

def store_context(link, text):
    """Store context embeddings."""
    embedding = embedder.encode(text).tolist()
    context_collection.add(
        ids=[str(uuid.uuid4())],
        metadatas=[{"link": link, "text": text}],
        embeddings=[embedding]
    )


def rename_chat(chat_id, new_heading):
    """Rename a chat without overwriting it."""
    chat = get_chat(chat_id)
    if chat:
        store_chat(chat["user_id"], chat_id, new_heading, chat["messages"])


def get_user_chats(user_id):
    """Retrieve all chats for a user."""
    return list_chats(user_id)  # Simply reuse list_chats()


def retrieve_context(query):
    """Retrieve the most relevant context based on similarity search."""
    try:
        query_embedding = embedder.encode(query).tolist()
        results = context_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["metadatas"]
        )

        if not results or "metadatas" not in results or not results["metadatas"]:
            return ""

        return "\n\n".join([res.get("text", "") for res in results["metadatas"] if "text" in res])

    except Exception as e:
        print(f"Error retrieving context for query '{query}': {str(e)}")
        return ""
