# Run the application

```shell
    python app.py
```

# API available

```text
    POST http://127.0.0.1:5000/ask
    GET http://127.0.0.1:5000/get_chat/<chat_id>
    GET http://127.0.0.1:5000/list_chats
    DELETE http://127.0.0.1:5000/delete_chat/<chat_id>
    POST http://127.0.0.1:5000/rename_chat/<chat_id>
```

# All API Descriptions
## get_lists()
```text
List all active chats with filtering, sorting, and partial search by heading.
        curl -X GET "http://127.0.0.1:5000/list_chats"
        curl -X GET "http://127.0.0.1:5000/list_chats?page=2&limit=5"
        curl -X GET "http://127.0.0.1:5000/list_chats?search=AI&page=1&limit=3"
        curl -X GET "http://127.0.0.1:5000/list_chats?start_date=2025-03-01T00:00:00Z&end_date=2025-03-10T23:59:59Z"
        curl -X GET "http://127.0.0.1:5000/list_chats?search=AI"
        curl -X GET "http://127.0.0.1:5000/list_chats?start_date=2025-03-01T00:00:00Z&end_date=2025-03-10T23:59:59Z"

    Return format without pagination:
    {
        "page": "all",
        "limit": "all",
        "total_chats": 12,
        "total_pages": 1,
        "chats": [
            {"chat_id": "123e4567-e89b-12d3-a456-426614174000", "heading": "Understanding AI Basics", "created_at": "2025-03-05T12:34:56.789Z"},
            {"chat_id": "789fgh12-x456-78d3-z456-987654321000", "heading": "Exploring Quantum Computing", "created_at": "2025-03-08T13:10:22.456Z"}
        ]
    }

    Return format with pagination:
    {
        "page": 1,
        "limit": 5,
        "total_chats": 12,
        "total_pages": 3,
        "chats": [
            {"chat_id": "123e4567-e89b-12d3-a456-426614174000", "heading": "Understanding AI Basics", "created_at": "2025-03-05T12:34:56.789Z"},
            {"chat_id": "789fgh12-x456-78d3-z456-987654321000", "heading": "Exploring Quantum Computing", "created_at": "2025-03-08T13:10:22.456Z"}
        ]
    }
```
# sustainathon-backend
