import os 
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load envrionment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Allow requests from the frontend
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure the Gemini API client
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("--- FATAL ERROR: GEMINI_API_KEY not found in environment. ---")
    print("--- Please ensure you have a .env file in the 'backend' directory with your key. ---" )
    print("--- The content should be exactly: GEMINI_API_KEY= 'Your_key_Here' ---")
    model = None
else: 
    print("API Key found. Attempting to configure Gemini client...")
    try:
      genai.configure(api_key=api_key)
      model = genai.GenerativeModel('gemini-1.5-flash')
      print("Gemini model configured successfully.")
    except Exception as e:
      print(f"--- ERROR CONFIGURING GEMINI API ---")
      print(f"Error: {e}")
      model = None

@app.route("/")
def index():
    return "SmartRepAI Backend is running!"
    
@app.route("/api/chat", methods=["POST"])
def chat_handler():
    if model is None:
        return jsonify({"error": "Gemini model is not configured. Check backend logs."}), 500

    data = request.get_json()
    # The frontend will send the entire chat history in a 'history' key
    if not data or "history" not in data:
        return jsonify({"error": "Invalid request: 'history' not found"}), 400

    # The history from the frontend is the full conversation
    history = data["history"]

    try:
        # Use the simpler, stateless generate_content method which is more robust 
        response = model.generate_content(history)
        return jsonify({"text": response.text})

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return jsonify({"error": str(e)}), 500

# This is the block that starts the Flask server
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, port=5000)