import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
print("API KEY exists:", bool(api_key))

print("Default client...")
try:
    client = genai.Client(api_key=api_key)
    audio_file = client.files.upload(
        file="dummy.txt",
        config={'display_name': "dummy_chunk_001"}
    )
    print("Upload success with default client:", audio_file.name)
except Exception as e:
    print("Error during upload 1:", type(e), e)

print("Client with timeout...")
try:
    client2 = genai.Client(api_key=api_key, http_options={'timeout': 600.0})
    audio_file2 = client2.files.upload(
        file="dummy.txt",
        config={'display_name': "dummy_chunk_001_v2"}
    )
    print("Upload success with 600 timeout:", audio_file2.name)
except Exception as e:
    print("Error during upload 2:", type(e), e)
