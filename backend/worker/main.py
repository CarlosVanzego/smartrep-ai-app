import os
import tempfile
from flask import Flask, request, jsonify
from supabase import create_client, Client
import google.generativeai as genai
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader

# --- Cloud Run Environment ---
app = Flask(__name__)

# Initialize Supabase client using environment variables
supa_url = os.environ.get("YOUR_SUPABASE_PROJECT_URL")
supa_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase_client: Client = create_client(supa_url, supa_key)

# Initialize Gemini model
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route("/upload-knowledge", methods=["POST"])
def upload_knowledge_worker():
    if 'file' not in request.files or 'user_id' not in request.form:
        return jsonify({"error": "File or user_id missing"}), 400
    
    file = request.files['file']
    user_id = request.form['user_id']

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
            records_to_insert.append({
                "user_id": user_id,
                "content": chunk.page_content,
                "embedding": vector
            })

        supabase_client.table("documents").insert(records_to_insert).execute()
        os.remove(tmp.name)
        return jsonify({"message": f"Successfully added knowledge from {file.filename}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat_worker():
    data = request.get_json()
    history = data["history"]
    user_id = data["user_id"]
    last_message = history[-1]['parts'][0]['text']

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        query_embedding = embeddings.embed_query(last_message)

        matches = supabase_client.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_count': 3,
            'requesting_user_id': user_id
        }).execute()

        context_text = "\n\n".join([item['content'] for item in matches.data]) if matches.data else ""
        
        prompt_with_context = f"Context:\n{context_text}\n\nUser Question: {last_message}"
        history[-1]['parts'][0]['text'] = prompt_with_context

        response = model.generate_content(history)
        return jsonify({"text": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))