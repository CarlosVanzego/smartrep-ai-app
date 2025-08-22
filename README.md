# SmartRepAI: AI-Powered Sales Training Platform

# Project Description
SmartRepAI is a full-stack web application designed to be an indispensable training and field-readiness tool for pharmaceutical sales representatives. The platform leverages a powerful generative AI to provide a suite of features aimed at enhancing product knowledge, messaging skills, and objection handling. Users can upload their own product documentation to create a personalized knowledge base, practice conversations in a realistic role-play environment, and receive instant, AI-driven coaching feedback.

# Features
- Secure User Authentication: Full sign-up, login, and session management handled by Supabase, ensuring all user data is secure and private.
- AI-Powered Knowledge Base (RAG): Users can upload PDF documents (e.g., clinical studies, package inserts) to create a personalized knowledge base. The AI can then answer questions using only the information from those documents.
- Automated Role-Play Coaching: Users can engage in role-play conversations with the AI. Upon completion, the AI analyzes the entire transcript and provides actionable coaching feedback, which is saved to the user's profile.
- Voice-to-Text Input: The chat interfaces include a speech-to-text feature, allowing for more natural and realistic conversational practice.
- Dynamic Frontend: A responsive, single-page application interface built with HTML, Tailwind CSS, and vanilla JavaScript.

# How to Run the App
This project is split into a Python backend and a vanilla JavaScript frontend. You will need two terminal windows to run them concurrently.

1. Clone the Repository
Bash

git clone [Your Repository URL Here]
cd smartrep-ai-app
2. Set Up the Backend
Navigate to the backend directory:

Bash

cd backend
Create and activate a virtual environment:

Bash

# Create the environment
python3 -m venv venv
# Activate it (Mac/Linux)
source venv/bin/activate
# Activate it (Windows)
# venv\Scripts\activate
Install Python dependencies:

Bash

pip install -r requirements.txt
Create your environment file: In the backend folder, create a file named .env and add your secret keys. Remember to enclose the keys in quotes.

Code snippet

GEMINI_API_KEY="your-google-ai-api-key"
SUPABASE_SERVICE_KEY="your-supabase-service-role-key"
Run the backend server:

Bash

python3 app.py
The backend will now be running on http://127.0.0.1:5000. Leave this terminal running.

3. Set Up and Run the Frontend
Open a new terminal window.

Navigate to the project's root directory.

Set up the frontend: The frontend is a single index.html file that requires your public Supabase keys. Open frontend/index.html in your code editor and add your keys to the script at the bottom:

JavaScript

const supaUrl = 'YOUR_SUPABASE_PROJECT_URL';
const supaAnonKey = 'YOUR_SUPABASE_ANON_KEY';
Run the frontend with Live Server:

If you have the "Live Server" extension in VS Code, right-click on frontend/index.html and select "Open with Live Server".

Alternatively, you can open the index.html file directly in your web browser.

The application will open in your browser, and you can now sign up, log in, and use the features.

# Technologies Used
Backend:

Python 3.13

Flask (for the web server and API)

google-generativeai (for the Gemini API)

supabase (for user authentication and database operations)

langchain, pypdf (for the RAG pipeline)

Frontend:

HTML5

Tailwind CSS

Vanilla JavaScript

Database & Auth:

Supabase (PostgreSQL with pgvector extension)

Deployment:

Vercel
