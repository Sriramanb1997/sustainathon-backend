# BioSphere - BackEnd
_A Gen AI Powered Centralized Platform for Wildlife, Biodiversity, and Conservation Projects Across India_

*Python based backend application*


## Execution Code

#### Command to clone the code

```
https://github.com/Sriramanb1997/sustainathon-backend.git
```
## Run the application

### Setup

```shell
    python -m venv sustainathon
    source sustainathon/bin/activate
    pip install -r requirements.txt
```

### Run 

```shell
    python app.py
```
## Scrapping Command

```
extractURL_spider.py https://www.thegrasslandstrust.org/
```

# API Endpoints

## 1. Ask a Question

- **Endpoint:** `/ask`
- **Method:** `POST`
- **Description:** Ask a question and receive a response from the AI assistant.
- **Input Format:** JSON
- **Parameters:**
  - **Query Parameters:**
    - `user_id` (required): User ID of the person asking the question.
  - **Body:**
    ```json
    {
      "question": "Your question here",
      "chat_id": "Optional chat ID"
    }
    ```
- **Output Format:** JSON
- **Response:**
  ```json
  {
    "chat_id": "Unique chat ID",
    "user_id": "User ID",
    "heading": "Chat heading",
    "created_at": "Creation timestamp",
    "messages": [
      {"role": "user", "content": "User question"},
      {"role": "assistant", "content": "Assistant response"}
    ]
  }
  ```

## 2. Delete a Chat

- **Endpoint:** `/delete_chat/<chat_id>`
- **Method:** `DELETE`
- **Description:** Delete a specific chat by its ID.
- **Input Format:** URL parameter
- **Parameters:**
  - `chat_id` (required): The ID of the chat to be deleted.
  - **Query Parameters:**
    - `user_id` (required): User ID associated with the chat.
- **Output Format:** JSON
- **Response:**
  ```json
  {
    "message": "Chat deleted successfully"
  }
  ```

## 3. List Chats

- **Endpoint:** `/list_chats`
- **Method:** `GET`
- **Description:** List all chats for a user with filtering, sorting, search, and pagination.
- **Input Format:** URL parameters
- **Parameters:**
  - `user_id` (required): User ID for which to list chats.
  - `start_date` (optional): Filter by start date.
  - `end_date` (optional): Filter by end date.
  - `sort_order` (optional): Sort order (`asc` or `desc`).
  - `search` (optional): Search query in chat headings.
  - `page` (optional): Page number for pagination.
  - `limit` (optional): Number of items per page.
- **Output Format:** JSON
- **Response:**
  ```json
  {
    "user_id": "User ID",
    "page": 1,
    "limit": 10,
    "total_chats": "Total number of chats",
    "total_pages": "Total number of pages",
    "chats": [
      {
        "chat_id": "Chat ID",
        "user_id": "User ID",
        "heading": "Chat heading",
        "created_at": "Timestamp"
      }
    ]
  }
  ```

## 4. Get a Chat

- **Endpoint:** `/get_chat/<chat_id>`
- **Method:** `GET`
- **Description:** Retrieve a chat by ID.
- **Input Format:** URL parameter
- **Parameters:**
  - `chat_id` (required): The ID of the chat to retrieve.
- **Output Format:** JSON
- **Response:**
  ```json
  {
    "chat_id": "Chat ID",
    "user_id": "User ID",
    "heading": "Chat heading",
    "created_at": "Timestamp",
    "messages": [
      {"role": "user", "content": "User message"},
      {"role": "assistant", "content": "Assistant message"}
    ]
  }
  ```

## 5. Rename a Chat

- **Endpoint:** `/rename_chat/<chat_id>`
- **Method:** `POST`
- **Description:** Rename a chat in ChromaDB.
- **Input Format:** JSON
- **Parameters:**
  - `chat_id` (required): The ID of the chat to rename.
  - **Body:**
    ```json
    {
      "heading": "New chat heading"
    }
    ```
- **Output Format:** JSON
- **Response:**
  ```json
  {
    "chat_id": "Chat ID",
    "user_id": "User ID",
    "heading": "New chat heading",
    "created_at": "Timestamp"
  }
  ```

## 6. Get User Chats

- **Endpoint:** `/get_user_chats/<user_id>`
- **Method:** `GET`
- **Description:** Retrieve all chats for a specific user from ChromaDB.
- **Input Format:** URL parameter
- **Parameters:**
  - `user_id` (required): The ID of the user whose chats are to be retrieved.
- **Output Format:** JSON
- **Response:**
  ```json
  {
    "user_id": "User ID",
    "chats": {
      "total_chats": "Total number of chats",
      "total_pages": "Total number of pages",
      "chats": [
        {
          "chat_id": "Chat ID",
          "user_id": "User ID",
          "heading": "Chat heading",
          "created_at": "Timestamp"
        }
      ]
    }
  }
  ```

## 7. Delete User Chats

- **Endpoint:** `/delete_user_chats/<user_id>`
- **Method:** `DELETE`
- **Description:** Delete all chats for a user.
- **Input Format:** URL parameter
- **Parameters:**
  - `user_id` (required): The ID of the user whose chats are to be deleted.
- **Output Format:** JSON
- **Response:**
  ```json
  {
    "message": "All chats deleted for user 'User ID'"
  }
  ```

## 8. Update Context with URLs

- **Endpoint:** `/api/update_context_url`
- **Method:** `POST`
- **Description:** Update context with URLs and ignored keywords while web scraping.
- **Input Format:** JSON
- **Parameters:**
  - **Body:**
    ```json
    {
      "urls": ["http://example.com", "http://anotherexample.com"],
      "ignored_keywords": ["keyword1", "keyword2"]
    }
    ```
- **Output Format:** JSON
- **Response:**
  ```json
  {
    "message": "Received X URLs and Y ignored keywords for processing."
  }
  ```