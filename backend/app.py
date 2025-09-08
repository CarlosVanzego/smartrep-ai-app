import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client
from functools import wraps

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

supa_url: str = os.getenv("YOUR_SUPABASE_PROJECT_URL")
supa_key: str = os.getenv("SUPABASE_SERVICE_KEY")
supabase_client: Client = create_client(supa_url, supa_key)

WORKER_URL = os.getenv("WORKER_URL")

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
    return "SmartRepAI Lean Proxy Backend is running!"

@app.route("/api/chat", methods=["POST"])
@token_required
def chat_handler(user):
    if not WORKER_URL:
        return jsonify({"error": "Worker service is not configured"}), 500

    data = request.get_json()
    data['user_id'] = user.id

    try:
        response = requests.post(f"{WORKER_URL}/chat", json=data)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to worker service: {e}"}), 500

@app.route("/api/upload-knowledge", methods=["POST"])
@token_required
def upload_knowledge(user):
    if not WORKER_URL:
        return jsonify({"error": "Worker service is not configured"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    try:
        files = {'file': (file.filename, file.stream, file.mimetype)}
        data = {'user_id': user.id}
        response = requests.post(f"{WORKER_URL}/upload-knowledge", files=files, data=data)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to worker service: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)