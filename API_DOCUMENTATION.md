# Chat Analytics API Documentation

This document provides details about the endpoints available in the Chat Analytics API.

**Base URL:** (Your API base URL, e.g., `http://localhost:8000`)

---

## Authentication

No authentication is required for the current endpoints.

---

## Endpoints

### 1. Upload Chat File

- **Endpoint:** `/upload`
- **Method:** `POST`
- **Description:** Uploads a chat file (e.g., WhatsApp export) for analysis. The file content should be base64 encoded. This initializes the analytics engine for subsequent requests (except `/all`).
- **Request Body:**
  ```json
  {
    "filename": "your_chat.txt",
    "file_content": "<base64_encoded_string_of_file_content>"
  }
  ```
- **Response (Success):**
  ```json
  {
    "message": "File uploaded and processed successfully"
  }
  ```
- **Response (Error):**
  ```json
  {
    "detail": "Error message description"
  }
  ```
  (Status Code: 500)

---

### 2. Get Frequent Words

- **Endpoint:** `/frequent-words`
- **Method:** `GET`
- **Description:** Generates a bar chart visualization of the most frequent words used by each user (minimum 4 characters). Requires a file to be uploaded via `/upload` first.
- **Query Parameters:**
  - `n` (integer, optional, default: 5): The number of top frequent words to display per user.
- **Response (Success):**
  ```json
  {
    "message": "Analysis completed",
    "visualization": "<base64_encoded_png_image_data>"
  }
  ```
- **Response (Error):**
  - Status Code 400: `{"detail": "Please upload a file first"}`
  - Status Code 500: `{"detail": "Error message description"}`

---

### 3. Get Message Length Analysis

- **Endpoint:** `/message-length`
- **Method:** `GET`
- **Description:** Provides statistics on average message length per user and a visualization of average message length over time. Requires a file to be uploaded via `/upload` first.
- **Response (Success):**
  ```json
  {
    "message": "Analysis completed",
    "data": {
      "<user1>": <avg_length1>,
      "<user2>": <avg_length2>,
      ...
    },
    "visualization": "<base64_encoded_png_image_data>"
  }
  ```
- **Response (Error):**
  - Status Code 400: `{"detail": "Please upload a file first"}`
  - Status Code 500: `{"detail": "Error message description"}`

---

### 4. Get Active Time Analysis

- **Endpoint:** `/active-time`
- **Method:** `GET`
- **Description:** Generates a visualization showing chat activity based on the day of the week. Requires a file to be uploaded via `/upload` first.
- **Response (Success):**
  ```json
  {
    "message": "Analysis completed",
    "visualization": "<base64_encoded_png_image_data>"
  }
  ```
- **Response (Error):**
  - Status Code 400: `{"detail": "Please upload a file first"}`
  - Status Code 500: `{"detail": "Error message description"}`

---

### 5. Get User Interaction Graph

- **Endpoint:** `/user-interaction`
- **Method:** `GET`
- **Description:** Creates a network graph visualizing user interactions based on message counts. Requires a file to be uploaded via `/upload` first.
- **Response (Success):**
  ```json
  {
    "message": "Analysis completed",
    "visualization": "<base64_encoded_png_image_data>"
  }
  ```
- **Response (Error):**
  - Status Code 400: `{"detail": "Please upload a file first"}`
  - Status Code 500: `{"detail": "Error message description"}`

---

### 6. Get Topic Modeling

- **Endpoint:** `/topic-modeling`
- **Method:** `GET`
- **Description:** Performs Latent Dirichlet Allocation (LDA) to identify potential topics within the chat messages. Requires a file to be uploaded via `/upload` first.
- **Query Parameters:**
  - `n_topics` (integer, optional, default: 5): The number of topics to identify.
- **Response (Success):**
  ```json
  {
    "message": "Analysis completed",
    "topics": [
      "Topic 1: word1, word2, word3, ...",
      "Topic 2: wordA, wordB, wordC, ...",
      ...
    ]
  }
  ```
- **Response (Error):**
  - Status Code 400: `{"detail": "Please upload a file first"}`
  - Status Code 500: `{"detail": "Error message description"}`

---

### 7. Get Similar Conversations

- **Endpoint:** `/similar-conversations`
- **Method:** `GET`
- **Description:** Clusters messages based on semantic similarity using sentence embeddings and visualizes the clusters. Requires a file to be uploaded via `/upload` first.
- **Query Parameters:**
  - `n_clusters` (integer, optional, default: 5): The number of clusters to create.
- **Response (Success):**
  ```json
  {
    "message": "Analysis completed",
    "visualization": "<base64_encoded_png_image_data>"
  }
  ```
- **Response (Error):**
  - Status Code 400: `{"detail": "Please upload a file first"}`
  - Status Code 500: `{"detail": "Error message description"}`

---

### 8. Get Sentiment Analysis

- **Endpoint:** `/sentiment`
- **Method:** `GET`
- **Description:** Calculates the average sentiment polarity per user and provides a visualization. Requires a file to be uploaded via `/upload` first.
- **Response (Success):**
  ```json
  {
    "message": "Analysis completed",
    "visualization": "<base64_encoded_png_image_data>"
  }
  ```
- **Response (Error):**
  - Status Code 400: `{"detail": "Please upload a file first"}`
  - Status Code 500: `{"detail": "Error message description"}`

---

### 9. Get Conversation Summary

- **Endpoint:** `/conversation-summary`
- **Method:** `GET`
- **Description:** Generates a concise summary of the conversation within a specified date range using a generative AI model (Gemini). Requires a file to be uploaded via `/upload` first.
- **Query Parameters:**
  - `start_date` (string, required, format: `YYYY-MM-DD`): The start date for the summary.
  - `end_date` (string, required, format: `YYYY-MM-DD`): The end date for the summary.
- **Response (Success):**
  ```json
  {
    "message": "Analysis completed",
    "summary": {
      "summary": "A concise paragraph summarizing the conversation..."
    }
  }
  ```
- **Response (Error):**
  - Status Code 400: `{"detail": "Please upload a file first"}`
  - Status Code 400: `{"detail": "No messages found in the specified date range"}`
  - Status Code 400/500: `{"detail": "Failed to generate summary: <error_details>"}`

---

### 10. Get All Analytics

- **Endpoint:** `/all`
- **Method:** `POST`
- **Description:** Uploads a chat file (base64 encoded) and performs all available analyses in a single request. This endpoint handles file processing internally and does not require a prior `/upload` call.
- **Request Body:**
  ```json
  {
    "filename": "your_chat.txt",
    "file_content": "<base64_encoded_string_of_file_content>",
    "n_frequent_words": 5, // Optional, default: 5
    "n_topics": 5, // Optional, default: 5
    "n_clusters": 5, // Optional, default: 5
    "start_date": "YYYY-MM-DD", // Required for summary
    "end_date": "YYYY-MM-DD" // Required for summary
  }
  ```
- **Response (Success):**
  ```json
  {
    "message": "File processed and all analyses completed successfully",
    "data": {
      "frequent_words": {
        "visualization": "<base64_image_data>"
      },
      "message_length": {
         "data": { "<user1>": <avg_length1>, ... },
         "visualization": "<base64_image_data>"
      },
      "active_time": {
        "visualization": "<base64_image_data>"
      },
      "user_interaction": {
        "visualization": "<base64_image_data>"
      },
      "topic_modeling": {
        "topics": [ "Topic 1: ...", ... ]
      },
      "similar_conversations": {
        "visualization": "<base64_image_data>"
      },
      "sentiment": {
        "visualization": "<base64_image_data>"
      },
      "conversation_summary": {
         "summary": "A concise paragraph summarizing the conversation..."
         // or {"error": "..."} if summary fails or no messages in range
      }
    }
  }
  ```
- **Response (Error):**
  ```json
  {
    "detail": "Error message description"
  }
  ```
  (Status Code: 500)

---
