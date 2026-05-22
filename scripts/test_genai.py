import os
from google import genai
import time

try:
    print("Initializing client...")
    client = genai.Client(http_options={'timeout': 600.0})
    print("Success")
except Exception as e:
    print("Error:", e)
