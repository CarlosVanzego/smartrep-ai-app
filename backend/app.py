import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client
from functools import wraps
import google.generativeai as genai
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader

# Load environment variables. Render will provide these.
load_dotenv()

# --- Unified Backend for Render ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize Supabase client
supa_url: str = os.getenv("YOUR_SUPABASE_PROJECT_URL")
supa_key: str = os.getenv("SUPABASE_SERVICE_KEY")
supabase_client: Client = create_client(supa_url, supa_key)

# Initialize Gemini model
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

# Decorator to protect routes and validate user tokens
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').split(" ")[-1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            user_response = supabase_client.auth.get_user(token)
            kwargs['user'] = user_response.user
        except Exception as e:
            return jsonify({'message': f'Token is invalid or expired! {e}'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/")
def index():
    if model is None:
        return "SmartRepAI Unified Backend: Error - Gemini API Key is missing.", 500
    return "SmartRepAI Unified Backend is running!"

@app.route("/api/upload-knowledge", methods=["POST"])
@token_required
def upload_knowledge(user):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            loader = PyPDFLoader(tmp.name)
            documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        records_to_insert = []
        for chunk in chunks:
            vector = embeddings.embed_query(chunk.page_content)
            records_to_insert.append({ "user_id": user.id, "content": chunk.page_content, "embedding": vector })

        supabase_client.table("documents").insert(records_to_insert).execute()
        os.remove(tmp.name)
        return jsonify({"message": f"Successfully added knowledge from {file.filename}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
@token_required
def chat_handler(user):
    data = request.get_json()
    history = data["history"]
    last_message = history[-1]['parts'][0]['text']

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        query_embedding = embeddings.embed_query(last_message)

        matches = supabase_client.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_count': 3,
            'requesting_user_id': user.id
        }).execute()

        context_text = "\n\n".join([item['content'] for item in matches.data]) if matches.data else ""
        
        prompt_with_context = f"Context:\n{context_text}\n\nUser Question: {last_message}"
        history[-1]['parts'][0]['text'] = prompt_with_context

        response = model.generate_content(history)
        return jsonify({"text": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Gunicorn is used in production, this is for local testing
    app.run(debug=True, port=5000)

