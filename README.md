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
