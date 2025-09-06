# # Part 1: Importing Necessary Libraries
# # These are the foundational tools for my smartrep ai app.
# # 'os' is a module that provides a way of using operating system dependent functionality like reading or writing to the file system.
# # 'google.generativeai' is a library for interacting with Google's generative AI models.
# # 'flask' is a micro web framework for Python, used to build web applications.
# # 'flask_cors' is used to handle Cross-Origin Resource Sharing, (CORS) which is necessary for allowing the frontend to communicate with the backend.
# # 'dotenv' is used to load environment variables from a .env file, which is a common practice for managing sensitive information like API keys.
# # 'supabase' is a library for interacting with Supabase, a backend-as-a-service platform that provides a PostgreSQL database and authentication.
# # 'functools' is a module that provides higher-order functions that act on or return other functions.
# # 'langchain_google_genai' is a library for working with Google's generative AI models
# # 'langchain.text_splitter' is used to split text into manageable chunks.
# # 'langchain_community.document_loaders' is used to load documents, in this case, PDF files.
# # 'tempfile' is a module for creating temporary files and directories.
# import os 
# import google.generativeai as genai
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv
# from supabase import create_client, Client
# from functools import wraps 
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# import tempfile



# # Load envrionment variables from .env file
# load_dotenv()

# # Part 2: Setting Up the Flask Application
# # This is where we initialize the Flask app and configure it to allow CORS.
# # CORS is important for allowing the frontend (which might be hosted on a different domain)
# # to make requests to the backend without running into security issues.
# # The Supabase client is also initialized here to interact with the database.

# # Initialize Flask app
# # app is the main application object, Flask is the framwork I am using, __name__ is a special variable that represents the name of the current module.
# app = Flask(__name__)
# # CORS allows the frontend to make requests to the backend without running into security issues; app is the Flask instance; resources is a dictionary that specifies which routes should allow CORS and from which origins.
# # Here, I'm allowing all origins to access the API routes under /api/*
# CORS(app, resources={r"/api/*": {"origins": "*"}})

# # Here I Initialize the Supabase client for backend
# # supa_url
# supa_url: str = "https://sckbcgxrebtmxozplmke.supabase.co"
# supa_key: str = os.getenv("SUPABASE_SERVICE_KEY")
# supabase_client: Client = create_client(supa_url, supa_key)

# # Configure the Gemini API client
# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     print("--- FATAL ERROR: GEMINI_API_KEY not found in environment. ---")
#     print("--- Please ensure you have a .env file in the 'backend' directory with your key. ---" )
#     print("--- The content should be exactly: GEMINI_API_KEY= 'Your_key_Here' ---")
#     model = None
# else: 
#     print("API Key found. Attempting to configure Gemini client...")
#     try:
#       genai.configure(api_key=api_key)
#       model = genai.GenerativeModel('gemini-1.5-flash')
#       print("Gemini model configured successfully.")
#     except Exception as e:
#       print(f"--- ERROR CONFIGURING GEMINI API ---")
#       print(f"Error: {e}")
#       model = None

# # Decorator to protect routes
# def token_required(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         token = None
#         if 'Authorization' in request.headers:
#             token = request.headers['Authorization'].split(" ")[1]
#         if not token:
#             return jsonify({'message': 'Token missing!'}), 401
#         try:
#             supabase_client.auth.get_user(token)
#         except:
#             return jsonify({'message': 'Token is invalid or expired!'}), 401
#         return f(*args, **kwargs)
#     return decorated


# @app.route("/")
# def index():
#     return "SmartRepAI Backend is running!"  

# @app.route("/api/upload-knowledge", methods=["POST"])
# @token_required
# def upload_knowledge():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     try:
#         # Get user from token to associate data with them
#         token = request.headers['Authorization'].split(" ")[1]
#         user = supabase_client.auth.get_user(token).user

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#             file.save(tmp.name)
#             loader = PyPDFLoader(tmp.name)
#             documents = loader.load()

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = text_splitter.split_documents(documents)
        
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
#         records_to_insert = []
#         for chunk in chunks:
#             vector = embeddings.embed_query(chunk.page_content)
#             records_to_insert.append({
#                 "user_id": user.id,
#                 "content": chunk.page_content,
#                 "embedding": vector
#             })

#         supabase_client.table("documents").insert(records_to_insert).execute()
        
#         return jsonify({"message": f"Successfully added knowledge from {file.filename}"}), 200

#     except Exception as e:
#         print(f"Error in file upload: {e}")
#         return jsonify({"error": str(e)}), 500
#     finally:
#         os.remove(tmp.name)
 
# @app.route("/api/chat", methods=["POST"])
# @token_required
# def chat_handler():
#     if model is None:
#         return jsonify({"error": "Gemini model is not configured. Check backend logs."}), 500

#     data = request.get_json()
#     if not data or "history" not in data:
#         return jsonify({"error": "Invalid request: 'history' not found"}), 400

#     history = data["history"]
#     last_message = history[-1]['parts'][0]['text']

#     try:
#         token = request.headers['Authorization'].split(" ")[1]
#         user = supabase_client.auth.get_user(token).user

#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
#         query_embedding = embeddings.embed_query(last_message)

#         matches = supabase_client.rpc('match_documents', {
#             'query_embedding': query_embedding,
#             'match_count': 3,
#             'requesting_user_id': user.id
#         }).execute()

#         context_text = ""
#         if matches.data:
#             context_text = "\n\n".join([item['content'] for item in matches.data])
        
#         prompt_with_context = f"""
#         Based on the following context, please answer the user's question. 
#         If the context does not contain the answer, say you don't have information on that topic.

#         Context:
#         ---
#         {context_text}
#         ---

#         User's Question: {last_message}
#         """
        
#         history[-1]['parts'][0]['text'] = prompt_with_context

#         response = model.generate_content(history)
#         return jsonify({"text": response.text})

#     except Exception as e:
#         print(f"Error during Gemini API call: {e}")
#         return jsonify({"error": str(e)}), 500
    
# @app.route("/api/roleplay-feedback", methods=["POST"])
# @token_required
# def roleplay_feedback():
#     data = request.get_json()
#     if not data or "history" not in data:
#         return jsonify({"error": "Invalid request: 'history' not found"}), 400

#     try:
#         token = request.headers['Authorization'].split(" ")[1]
#         user = supabase_client.auth.get_user(token).user

#         # The full conversation history from the frontend
#         conversation_history = data["history"]
        
#         # The "meta-prompt" to make the AI act as a coach
#         feedback_prompt = f"""
#         You are a pharmaceutical sales coach. The following is a transcript of a role-play conversation
#         between a sales rep and an AI pretending to be a doctor. Please analyze the rep's performance.
#         Provide specific, actionable feedback covering their opening, questioning skills, objection handling, and closing.
#         Format the feedback with bullet points.

#         CONVERSATION:
#         {conversation_history}
#         """

#         # Ask the AI for the feedback
#         feedback_response = model.generate_content(feedback_prompt)
#         feedback_text = feedback_response.text

#         # Save the feedback to the database
#         session_data = {
#             "user_id": user.id,
#             "persona": data.get("persona", "Unknown"),
#             "topic": data.get("topic", "General"),
#             "feedback": feedback_text
#         }
#         supabase_client.table("coaching_sessions").insert(session_data).execute()
        
#         # Return the generated feedback to the frontend
#         return jsonify({"feedback": feedback_text})

#     except Exception as e:
#         print(f"Error during feedback generation: {e}")
#         return jsonify({"error": str(e)}), 5000

# # This is the block that starts the Flask server
# if __name__ == "__main__":
#     print("Starting Flask server...")
#     app.run(debug=True, port=5001) # running the server on port 5001 to avoid conflicts with the frontend 