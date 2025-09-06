# Part 1: Importing Necessary Libraries
# These are the foundational tools for my SmartRep AI app
# 'os' is a module that provides a way of using operating system dependent functionality like reading or writing to the file system.
# 'google.generativeai' is a library for interacting with Google's generative AI models (like Gemini).
# 'flask' is a micro web framework for Python, used to build web applications and APIs.
# 'flask_cors' is used to handle Cross-Origin Resource Sharing (CORS), which allows the frontend (different domain) to communicate with this backend.
# 'dotenv' loads environment variables from a .env file (commonly used for API keys or secret credentials).
# 'supabase' is a library to interact with Supabase, which provides a PostgreSQL database and authentication services.
# 'functools' provides higher-order functions, like decorators, that act on or return other functions.
# 'langchain_google_genai' is a library for working with Google's AI embeddings inside the LangChain ecosystem.
# 'langchain.text_splitter' is used to break text into smaller chunks for efficient embeddings and processing.
# 'langchain_community.document_loaders' is used to load documents — here specifically PDF files.
# 'tempfile' creates temporary files and directories for handling uploads safely.
import os 
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client
from functools import wraps 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile


# Part 2: Load Environment Variables
# Loads variables from a .env file into environment variables.
# This is safer than hardcoding secrets like API keys into the codebase.
load_dotenv()


# Part 3: Setting Up the Flask Application
# Here we initialize the Flask app and configure CORS.
# CORS is critical so the frontend can call this backend without running into browser security restrictions.
app = Flask(__name__)
# Allow all origins (*) to access endpoints that match /api/*.
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Part 4: Initialize Supabase Client
# Supabase provides the database and authentication layer for this app.
# 'supa_url' is the unique project URL, and the key is stored securely in environment variables.
supa_url: str = "https://sckbcgxrebtmxozplmke.supabase.co"
supa_key: str = os.getenv("SUPABASE_SERVICE_KEY")
supabase_client: Client = create_client(supa_url, supa_key)


# Part 5: Configure the Gemini API client
# Gemini is Google's family of generative AI models.
# We check if the key exists; if not, we print fatal error messages so the developer knows how to fix it.
api_key = os.getenv("GEMINI_API_KEY")

# If the API key is missing, I print an error message and set model to None.
if not api_key:
    print("--- FATAL ERROR: GEMINI_API_KEY not found in environment. ---")
    print("--- Please ensure you have a .env file in the 'backend' directory with your key. ---" )
    print("--- The content should be exactly: GEMINI_API_KEY='Your_key_Here' ---")
    model = None
# Otherwise, I attempt to configure the Gemini client. 
else: 
    print("API Key found. Attempting to configure Gemini client...")
    # If configuration fails, I catch the exception and print an error message.
    try:
        # Configure the genai client with the API key.
        genai.configure(api_key=api_key)
        # Load a lightweight Gemini model variant for fast responses; This is the model that will be used for generating responses.
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini model configured successfully.")
    except Exception as e:
        print("--- ERROR CONFIGURING GEMINI API ---")
        print(f"Error: {e}")
        model = None


# Part 6: Create a Decorator for Token Authentication
# This function acts as a security gatekeeper.
# It wraps around protected routes and ensures the request includes a valid Supabase auth token.
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # Extract Bearer token from Authorization header.
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({'message': 'Token missing!'}), 401
        try:
            supabase_client.auth.get_user(token)
        except:
            return jsonify({'message': 'Token is invalid or expired!'}), 401
        return f(*args, **kwargs)
    return decorated


# Part 7: Define Flask Routes

# Root route — a simple health check to confirm backend is running.
@app.route("/")
def index():
    return "SmartRepAI Backend is running!"  


# Upload Knowledge Route
# This route handles uploading of PDF knowledge documents for a user.
# The PDFs are split into chunks, embedded into vectors, and stored in Supabase for retrieval later.
@app.route("/api/upload-knowledge", methods=["POST"])
@token_required
def upload_knowledge():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Get user associated with the token
        token = request.headers['Authorization'].split(" ")[1]
        user = supabase_client.auth.get_user(token).user

        # Save the uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            # Load document from PDF
            loader = PyPDFLoader(tmp.name)
            documents = loader.load()

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Generate embeddings for each chunk using Gemini embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        # Prepare structured records for Supabase
        records_to_insert = []
        for chunk in chunks:
            vector = embeddings.embed_query(chunk.page_content)
            records_to_insert.append({
                "user_id": user.id,
                "content": chunk.page_content,
                "embedding": vector
            })

        # Insert into Supabase documents table
        supabase_client.table("documents").insert(records_to_insert).execute()
        
        return jsonify({"message": f"Successfully added knowledge from {file.filename}"}), 200

    except Exception as e:
        print(f"Error in file upload: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Ensure temp file is deleted after processing
        os.remove(tmp.name)


# Chat Handler Route:
# This route handles live user queries against the knowledge base.
# It retrieves context from Supabase (via embeddings) and uses Gemini to answer.
@app.route("/api/chat", methods=["POST"])
@token_required
def chat_handler():
    if model is None:
        return jsonify({"error": "Gemini model is not configured. Check backend logs."}), 500

    data = request.get_json()
    if not data or "history" not in data:
        return jsonify({"error": "Invalid request: 'history' not found"}), 400

    history = data["history"]
    last_message = history[-1]['parts'][0]['text']

    try:
        # Identify user
        token = request.headers['Authorization'].split(" ")[1]
        user = supabase_client.auth.get_user(token).user

        # Create embedding for query
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        query_embedding = embeddings.embed_query(last_message)

        # Retrieve best matching documents using Supabase stored procedure
        matches = supabase_client.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_count': 3,
            'requesting_user_id': user.id
        }).execute()

        # Build context string from retrieved documents
        context_text = ""
        if matches.data:
            context_text = "\n\n".join([item['content'] for item in matches.data])
        
        # Create augmented prompt including retrieved knowledge
        prompt_with_context = f"""
        Based on the following context, please answer the user's question. 
        If the context does not contain the answer, say you don't have information on that topic.

        Context:
        ---
        {context_text}
        ---

        User's Question: {last_message}
        """
        
        # Replace last message in history with augmented prompt
        history[-1]['parts'][0]['text'] = prompt_with_context

        # Send history with context to Gemini
        response = model.generate_content(history)
        return jsonify({"text": response.text})

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return jsonify({"error": str(e)}), 500
    

# Roleplay Feedback Route
# This route is for generating coaching feedback from a simulated roleplay session.
# The AI analyzes a conversation and returns structured coaching tips for the sales rep.
@app.route("/api/roleplay-feedback", methods=["POST"])
@token_required
def roleplay_feedback():
    data = request.get_json()
    if not data or "history" not in data:
        return jsonify({"error": "Invalid request: 'history' not found"}), 400

    try:
        # Identify user
        token = request.headers['Authorization'].split(" ")[1]
        user = supabase_client.auth.get_user(token).user

        # Full conversation transcript
        conversation_history = data["history"]
        
        # Coaching prompt for the AI
        feedback_prompt = f"""
        You are a pharmaceutical sales coach. The following is a transcript of a role-play conversation
        between a sales rep and an AI pretending to be a doctor. Please analyze the rep's performance.
        Provide specific, actionable feedback covering their opening, questioning skills, objection handling, and closing.
        Format the feedback with bullet points.

        CONVERSATION:
        {conversation_history}
        """

        # Generate coaching feedback
        feedback_response = model.generate_content(feedback_prompt)
        feedback_text = feedback_response.text

        # Save feedback in Supabase database
        session_data = {
            "user_id": user.id,
            "persona": data.get("persona", "Unknown"),
            "topic": data.get("topic", "General"),
            "feedback": feedback_text
        }
        supabase_client.table("coaching_sessions").insert(session_data).execute()
        
        # Return feedback to frontend
        return jsonify({"feedback": feedback_text})

    except Exception as e:
        print(f"Error during feedback generation: {e}")
        return jsonify({"error": str(e)}), 5000


# Part 8: Run the Flask Server
# The entry point for running this application.
# When executed directly, it starts the Flask development server on port 5001.
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, port=5001)  # Port 5001 to avoid conflicts with frontend
