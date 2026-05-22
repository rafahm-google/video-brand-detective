import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')

client = genai.Client(api_key=api_key)
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='hello',
        config=types.GenerateContentConfig(
            http_options={'timeout': 600.0}
        )
    )
    print("Success:", response.text)
except Exception as e:
    print("Error:", type(e), e)
