from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Flask app
app = Flask(__name__)

@app.route('/process/assamese', methods=['POST'])
def process_assamese():
    try:
        # Step 1: Get Assamese text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' parameter"}), 400

        assamese_text = data['text']

        # Step 2: Translate Assamese to English
        translation_url = "http://localhost:5000/translate/assamese-to-english"
        response = requests.post(translation_url, json={"text": assamese_text})

        if response.status_code != 200:
            return jsonify({"error": "Failed to translate Assamese to English", "details": response.text}), response.status_code

        english_text = response.json().get("translated_text")
        print(english_text)
        # Step 3: Use Gemini API with English text
        gemini_prompt = f"You are going to get a prompt that is translated from assamese to english. Try to understand the prompt and respond according to it. Keep the responses short, within 20 words. Be clear and do not use hard english words. Here is the translated    prompt that you will respond to :{english_text}"
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=gemini_prompt
        )

        if not gemini_response or not gemini_response.text:
            return jsonify({"error": "Failed to get response from Gemini API"}), 500

        processed_english_text = gemini_response.text.strip()
        print(processed_english_text)
        # Step 4: Translate English back to Assamese
        reverse_translation_url = "http://localhost:5000/translate/english-to-assamese"
        reverse_response = requests.post(reverse_translation_url, json={"text": processed_english_text})

        if reverse_response.status_code != 200:
            return jsonify({"error": "Failed to translate English to Assamese", "details": reverse_response.text}), reverse_response.status_code

        processed_assamese_text = reverse_response.json().get("translated_text")
        print(processed_assamese_text)
        # Step 5: Return the processed Assamese text
        return jsonify({"processed_text": processed_assamese_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000)