# Speaker Verification API — Frontend Integration Guide

This document explains how frontend applications should interact with the speaker verification backend.

---

## Base URL

    http://127.0.0.1:8000

Adjust host and port according to your environment.

---

## 1. Health Check

**GET** `/`

**Purpose**  
Confirm the server is running.

**Example Response**

    {
      "status": "running"
    }

---

## 2. Enroll a User

**POST** `/enroll/{username}`

**Purpose**  
Register a user by uploading a `.wav` file. The backend extracts an embedding from the audio and stores it under the provided `username`.

**Request**
- Method: `POST`
- URL example: `/enroll/alice`
- Headers: `Content-Type: multipart/form-data`
- Form field:
  - `file` — the `.wav` audio file to upload

**cURL example**

    curl -X POST "http://127.0.0.1:8000/enroll/alice" -F "file=@/path/to/voice_sample.wav"

**Browser (fetch) example**

    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    fetch('http://127.0.0.1:8000/enroll/alice', {
      method: 'POST',
      body: fd
    })
    .then(r => r.json())
    .then(console.log);

**Axios example**

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    axios.post('http://127.0.0.1:8000/enroll/alice', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    }).then(res => console.log(res.data));

**Success Response (200)**

    {
      "status": "enrolled",
      "username": "alice"
    }

**Error Response (400)**

    {
      "detail": "Username already exists"
    }

---

## 3. Verify a User

**POST** `/verify/{username}`

**Purpose**  
Upload a `.wav` file for the specified `username` and compare it against the stored embedding. The API returns a similarity `score` and a `result` of `accepted` or `rejected`.

**Request**
- Method: `POST`
- URL example: `/verify/alice`
- Headers: `Content-Type: multipart/form-data`
- Form field:
  - `file` — the `.wav` audio file to verify

**cURL example**

    curl -X POST "http://127.0.0.1:8000/verify/alice" -F "file=@/path/to/verification_sample.wav"

**Success Response (200)**

    {
      "username": "alice",
      "method": "voice",
      "score": 0.6123,
      "result": "accepted"
    }

**Error Responses**
- `404` — user not enrolled

    {
      "detail": "User not enrolled"
    }

- `400` — no file provided

    {
      "detail": "Must provide voice"
    }

---

## 4. List Users

**GET** `/users`

**Purpose**  
Return the list of enrolled usernames.

**Example Response (200)**

    {
      "count": 3,
      "users": ["alice", "bob", "charlie"]
    }

---

## Integration Notes

- The file form field name is `file`. Do not rename it unless frontend and backend are updated together.
- Preferred audio format: WAV, mono, 16 kHz. Convert client-side if necessary.
- Handle HTTP status codes:
  - `200` success
  - `400` bad request (missing file, duplicate username)
  - `404` not found (user not enrolled)
- The `score` returned by verification is a numeric similarity; the `result` field gives a pass/fail decision.
- For production, serve the API over HTTPS and secure endpoints that modify or retrieve user data (authentication, rate limiting).
- If the frontend must send non-file data together with the file, include additional form fields in the same multipart request.

---


### main.py
This is the entry point of the backend.  
It sets up the ECAPA-TDNN model, loads the trained weights, and starts the FastAPI application.  
The routes defined here are:  
- GET / returns a health check response.  
- POST /enroll/{username} lets a new user enroll with a username, password, and voice sample.  
- POST /verify/{username} verifies an existing user either with their voice or their password.  

### model.py
This file contains the implementation of the ECAPA-TDNN model.  
The model is responsible for turning audio files into speaker embeddings, which are vector representations of a person’s voice.  

### ultils.py
This file contains helper functions used throughout the project:  
- load_parameters(model, path) loads pre-trained model weights.  
- get_embedding(model, file, device) reads a `.wav` file, processes it, and returns a normalized embedding.  
- cosine_score(emb1, emb2) calculates the similarity between two embeddings.  

### database.py
This file implements a very simple JSON-based database. It manages user information in database.json.  
Functions include:  
- add_user(username, password, emb) to add a new user.  
- get_user(username) to fetch user data.  
- verify_password(username, password) to check if a password is correct.  
- get_embedding(username) to retrieve a stored embedding for a user.  

### database.json
This is the actual storage file where user data is saved. Each user has a password and a voice embedding.  
