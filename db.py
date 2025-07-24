import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime
import json
import re
import functools
import hashlib
import logging
import spacy
import os

# Initialize ChromaDB client with absolute path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Create collections
chats_collection = chroma_client.get_or_create_collection(name="chats")
users_collection = chroma_client.get_or_create_collection(name="users")
context_collection = chroma_client.get_or_create_collection(name="context_embeddings")
link_collection = chroma_client.get_or_create_collection(name="links")


# Load Sentence Transformer model for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")


def store_chat(user_id, chat_id, heading, messages):
    """Store or update a chat session in ChromaDB."""
    existing_chat = chats_collection.get(ids=[chat_id])

    if existing_chat and existing_chat.get("metadatas"):
        chats_collection.update(
            ids=[chat_id],
            metadatas=[{"chat_id": chat_id, "user_id": user_id, "heading": heading, "created_at": datetime.utcnow().isoformat()}],
            documents=[json.dumps(messages)]
        )
    else:
        chats_collection.add(
            ids=[chat_id],
            metadatas=[{"chat_id": chat_id, "user_id": user_id, "heading": heading, "created_at": datetime.utcnow().isoformat()}],
            documents=[json.dumps(messages)]
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
            messages = json.loads(raw_data)
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
        if limit is None:
            # No pagination limit - return all chats
            total_pages = 1
            paginated_chats = filtered_chats
        else:
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
        link_id = str(uuid.uuid4())
        link['id'] = link_id
        print(link_id)
        link_collection.add(
            ids=[link_id],
            metadatas=[{"created_at": datetime.utcnow().isoformat()}],
            documents=[str(link)]
        )
    except Exception as e:
        print(f"Error adding link - ", str(e))
        return None

def delete_link(link_id):
    link_collection.delete(ids=[link_id])

def store_context(link, text):
    """Store context embeddings."""
    try:
        global context_collection
        # Ensure we have a valid collection reference
        if context_collection is None:
            context_collection = chroma_client.get_or_create_collection(name="context_embeddings")
        
        embedding = embedder.encode(text).tolist()
        context_collection.add(
            ids=[str(uuid.uuid4())],
            metadatas=[{"link": link, "text": text}],
            embeddings=[embedding]
        )
        logging.info(f"Successfully stored context for link: {link}")
    except Exception as e:
        logging.error(f"Error storing context for link {link}: {str(e)}")
        raise


def rename_chat(chat_id, new_heading):
    """Rename a chat without overwriting it."""
    chat = get_chat(chat_id)
    if chat:
        store_chat(chat["user_id"], chat_id, new_heading, chat["messages"])


def get_user_chats(user_id):
    """Retrieve all chats for a user."""
    return list_chats(user_id)  # Simply reuse list_chats()


# Helper to hash queries for cache keys
def _hash_query(query):
    return hashlib.sha256(query.encode('utf-8')).hexdigest()

# LRU cache for context retrieval (cache up to 128 unique queries)
@functools.lru_cache(maxsize=128)
def _cached_retrieve_context(query_hash, top_k):
    # This function is only called with hashed queries to avoid issues with unhashable types
    # The actual query string is not passed to the cache directly
    # ...existing code for retrieve_context, but use the global embedder and context_collection...
    try:
        # The query string is not available here, so we need to pass it as a global or refactor
        # Instead, we will use a global variable to store the last query string
        global _last_query_string
        query = _last_query_string
        query_embedding = embedder.encode(query).tolist()
        results = context_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas"]
        )
        if not results or "metadatas" not in results or not results["metadatas"]:
            return ""
        return "\n\n".join([res.get("content", "") for res in results["metadatas"][0] if "content" in res])
    except Exception as e:
        logging.error(f"Error retrieving context for query (cached): {str(e)}")
        return ""

# Global variable to store the last query string for the cache
_last_query_string = None

def retrieve_context(query, top_k=5):
    """Retrieve the most relevant context based on similarity search, with caching."""
    global _last_query_string
    _last_query_string = query
    query_hash = _hash_query(query)
    return _cached_retrieve_context(query_hash, top_k)


def delete_context_collection():
    global context_collection
    chroma_client.delete_collection(name="context_embeddings")
    context_collection = chroma_client.create_collection(name="context_embeddings")


def get_context_collection_count():
    """Get the count of documents in the context collection."""
    try:
        global context_collection
        # Ensure we have a valid collection reference
        if context_collection is None:
            context_collection = chroma_client.get_or_create_collection(name="context_embeddings")
        return context_collection.count()
    except Exception as e:
        logging.error(f"Error getting context collection count: {str(e)}")
        return 0


def store_contexts_objects(text_item):
    """Read a JSON file and store contexts with embeddings in ChromaDB."""
    try:
        global context_collection
        # Ensure we have a valid collection reference
        if context_collection is None:
            context_collection = chroma_client.get_or_create_collection(name="context_embeddings")

        url = text_item["url"]
        sanctuary = text_item["sanctuary"]
        ngos = text_item["ngos"]
        content = text_item["content"]

        # Clean the content before storing
        original_length = len(content)
        content = clean_scraped_content(content)
        cleaned_length = len(content)
        
        # Try lenient cleaning if strict cleaning removes too much
        if cleaned_length < 30 and original_length > 100:
            print(f"Trying lenient cleaning for {url}...")
            content = clean_scraped_content_lenient(text_item['content'])
            cleaned_length = len(content)
            print(f"  After lenient cleaning: {cleaned_length}")
        
        print(f"Content processing for {url}:")
        print(f"  Original length: {original_length}")
        print(f"  After cleaning: {cleaned_length}")
        print(f"  First 100 chars of original: {text_item['content'][:100]}...")
        print(f"  First 100 chars after cleaning: {content[:100]}...")

        if content == "" or len(content) < 30:
            logging.warning(f"Skipping content for URL {url} - content too short or empty (length: {len(content)})")
            print(f"REJECTED: Content too short for {url} (length: {len(content)})")
            return False

        combined_content = f" Name of the NGO: {' '.join(ngos)}. URL to the Content which NGO maintains: '{url}'. Sanctuaries which the NGO are maintaining: {' '.join(sanctuary)}. Content of the full page: {content}"

        embedding = embedder.encode(combined_content).tolist()
        doc_id = str(uuid.uuid4())
        
        context_collection.add(
            ids=[doc_id],
            metadatas=[{
                "url": url,
                "sanctuary": json.dumps(sanctuary),
                "ngos": json.dumps(ngos),
                "content": combined_content
            }],
            embeddings=[embedding]
        )
        
        current_count = context_collection.count()
        logging.info(f"Successfully stored context object for URL: {url} with ID: {doc_id}")
        print(f"Stored context for {url}. Collection count: {current_count}")
        return True

    except KeyError as e:
        logging.error(f"Missing required key in text_item: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error storing context object: {str(e)}")
        return False


def store_contexts_from_file(file_path):
    """Read a JSON file and store contexts with embeddings in ChromaDB."""
    try:
        global context_collection
        # Ensure we have a valid collection reference
        if context_collection is None:
            context_collection = chroma_client.get_or_create_collection(name="context_embeddings")
        
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            contexts = json.load(file)

        # Ensure contexts is a list
        if not isinstance(contexts, list):
            raise ValueError("Invalid data format. Expected an array of objects.")

        stored_count = 0
        # Store each context
        for context in contexts:
            url = context.get("url")
            sanctuary = context.get("sanctuary")
            ngos = context.get("ngos")
            content = context.get("content")

            # Clean the content before storing
            content = clean_scraped_content(content)

            if content == "" or len(content) < 30:
                logging.warning(f"Skipping content for URL {url} - content too short or empty")
                continue

            combined_content = f" Name of the NGO: {' '.join(ngos)}. URL to the Content which NGO maintains: '{url}'. Sanctuaries which the NGO are maintaining: {' '.join(sanctuary)}. Content of the full page: {content}"

            embedding = embedder.encode(combined_content).tolist()
            context_collection.add(
                ids=[str(uuid.uuid4())],
                metadatas=[{
                    "url": url,
                    "sanctuary": json.dumps(sanctuary),
                    "ngos": json.dumps(ngos),
                    "content": combined_content
                }],
                embeddings=[embedding]
            )
            stored_count += 1

        logging.info(f"Successfully stored {stored_count} contexts from file: {file_path}")
        return "Contexts stored successfully."

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return "File not found."
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in file: {file_path}")
        return "Invalid JSON format."
    except Exception as e:
        logging.error(f"Error storing contexts from file {file_path}: {str(e)}")
        return str(e)

# Patch: get context metadata for prompt building
def deduplicate_contexts(metadatas, similarity_threshold=0.3):
    """
    Simple deduplication: Keep only one entry per NGO to prevent repetitive information.
    """
    if not metadatas:
        return []
    
    deduplicated = []
    seen_ngos = set()
    
    for meta in metadatas:
        ngos_json = meta.get("ngos", "[]")
        
        try:
            ngos_list = json.loads(ngos_json) if isinstance(ngos_json, str) else ngos_json
            if not isinstance(ngos_list, list):
                ngos_list = [ngos_json] if ngos_json else []
        except (json.JSONDecodeError, TypeError):
            ngos_list = [ngos_json] if ngos_json else []
        
        # Check if any NGO in this context has already been seen
        ngo_already_seen = any(ngo.lower().strip() in seen_ngos for ngo in ngos_list if ngo)
        
        if not ngo_already_seen:
            deduplicated.append(meta)
            # Mark all NGOs in this context as seen
            for ngo in ngos_list:
                if ngo:
                    seen_ngos.add(ngo.lower().strip())
    
    return deduplicated



def get_structured_context(query, top_k=5):
    """
    Retrieve the most relevant context as a list of metadata dicts for structured prompt building.
    Uses improved entity extraction and specific matching to filter for truly relevant context chunks.
    Includes deduplication logic to prevent repetitive information.
    """
    global _last_query_string
    _last_query_string = query
    try:
        query_embedding = embedder.encode(query).tolist()
        results = context_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(100, top_k * 20),  # fetch many more for better sanctuary matching
            include=["metadatas"]
        )
        if not results or "metadatas" not in results or not results["metadatas"]:
            return []
        metadatas = results["metadatas"][0]
        
        # Improved entity filtering using spaCy with better specificity
        entities = extract_entities_from_question(query)
        if entities:
            filtered = []
            query_lower = query.lower()
            
            # Check for specific sanctuary/location names in the query
            specific_sanctuary_pattern = r'(\w+)(?:\s+(?:wls|wildlife\s+sanctuary|national\s+park|tiger\s+reserve|biosphere|sanctuary|park|reserve))'
            specific_matches = re.findall(specific_sanctuary_pattern, query_lower, re.IGNORECASE)
            
            for meta in metadatas:
                content = meta.get("content", "").lower()
                sanctuary_field = meta.get("sanctuary", "").lower()
                
                # If query mentions a specific sanctuary, check if it matches exactly
                if specific_matches:
                    sanctuary_match = False
                    for specific_match in specific_matches:
                        # Check sanctuary field first (higher priority)
                        if specific_match.lower().strip() in sanctuary_field:
                            sanctuary_match = True
                            break
                        # Then check content
                        elif specific_match.lower().strip() in content:
                            sanctuary_match = True
                            break
                    
                    if sanctuary_match:
                        # Additional check: if query has a specific sanctuary name, 
                        # ensure the main entity (sanctuary name) is present
                        main_entities = [e for e in entities if len(e) > 3 and e not in ['wls', 'sanctuary', 'national', 'park']]
                        if main_entities:
                            entity_match = any(entity in content or entity in sanctuary_field for entity in main_entities)
                            if entity_match:
                                filtered.append(meta)
                        else:
                            filtered.append(meta)
                else:
                    # For general queries, use the original logic but with stricter matching
                    # Require at least 2 entities to match OR one very specific entity
                    matching_entities = [entity for entity in entities if entity in content]
                    if len(matching_entities) >= 2 or any(len(entity) > 5 for entity in matching_entities):
                        filtered.append(meta)
            
            # If no specific matches found, fall back to semantic similarity only
            metadatas = filtered if filtered else metadatas[:top_k]
        
        # Deduplication logic
        deduplicated = deduplicate_contexts(metadatas)
        
        # Return top_k deduplicated results
        return deduplicated[:top_k]
    except Exception as e:
        logging.error(f"Error retrieving context for query (structured): {str(e)}")
        return []

def check_sanctuary_exists(sanctuary_name, state_name=None):
    """
    Check if a specific sanctuary exists in the database.
    Returns True if found, False otherwise.
    """
    try:
        # Skip obviously invalid sanctuary names
        if len(sanctuary_name.strip()) < 3 or sanctuary_name.lower().strip() in ['the', 'all', 'any', 'where', 'what', 'which', 'how', 'many', 'are', 'is']:
            return False
            
        # Search for the exact sanctuary name in the sanctuary field
        search_query = sanctuary_name.lower()
        if state_name:
            search_query += f" {state_name.lower()}"
            
        query_embedding = embedder.encode(search_query).tolist()
        results = context_collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["metadatas"]
        )
        
        if not results or "metadatas" not in results or not results["metadatas"]:
            return False
            
        for meta in results["metadatas"][0]:
            sanctuary_field = meta.get("sanctuary", "").lower()
            content = meta.get("content", "").lower()
            
            # More precise matching - require the sanctuary name to be a significant part, not just a substring
            sanctuary_name_lower = sanctuary_name.lower().strip()
            
            # Check for exact or close match in sanctuary field (must be substantial match)
            if sanctuary_name_lower in sanctuary_field:
                # Ensure it's not just a common word match - the sanctuary name should be substantial
                if len(sanctuary_name_lower) >= 4 and (not state_name or state_name.lower() in sanctuary_field):
                    return True
                    
            # Check for exact mention in content (only if it looks like a proper sanctuary name)
            if len(sanctuary_name_lower) >= 4 and sanctuary_name_lower in content:
                if not state_name or state_name.lower() in content:
                    return True
                    
        return False
    except Exception as e:
        logging.error(f"Error checking sanctuary existence: {str(e)}")
        return False

# Cleaning function for scraped content
def clean_scraped_content(text):
    ignore_keywords = [
        "home", "top of page", "contact", "about us", "privacy", "terms",
        "login", "copyright", "menu", "site map", "newsletter", "subscribe",
        "search", "disclaimer", "cookie", "policy", "advertise", "faq", "help",
        "image", "photo", "logo", "icon", "banner", "picture", "figure", "caption", "graphic"
    ]
    # Regex patterns for common image/caption/alt text
    image_caption_patterns = [
        r"^image(\s*\d*)?:", r"^photo(\s*\d*)?:", r"^figure(\s*\d*)?:", r"^caption:",
        r"^logo:", r"^icon:", r"^banner:", r"^picture:", r"^graphic:"
    ]
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        l = line.strip().lower()
        if any(kw in l for kw in ignore_keywords):
            continue
        if any(re.match(pat, l) for pat in image_caption_patterns):
            continue
        if len(l.split()) < 5:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

# Less aggressive cleaning function for debugging
def clean_scraped_content_lenient(text):
    """Less aggressive content cleaning for debugging purposes."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        l = line.strip()
        # Only remove very short lines (less than 3 words) and empty lines
        if len(l.split()) < 3 or l == "":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def extract_entities_from_question(question):
    """
    Extracts named entities (ORG, GPE, LOC, FAC, PERSON) from the user question using spaCy.
    Returns a list of unique entity strings.
    """
    doc = nlp(question)
    entities = set()
    for ent in doc.ents:
        if ent.label_ in {"ORG", "GPE", "LOC", "FAC", "PERSON"}:
            entities.add(ent.text.lower())
    return list(entities)