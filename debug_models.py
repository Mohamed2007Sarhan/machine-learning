from google import genai
import os

API_KEY = "AIzaSyA07X02l4V1HwJzVTIa_CpuMqWryVxPxLI"

client = genai.Client(api_key=API_KEY)

print("Listing models...")
try:
    print("Attempting to list models via client.models.list()...")
    # In the new google-genai SDK, listing models is often iterable
    for m in client.models.list():
        print(f"Found model: {m.name}")
        
    print("\n--- End of List ---")

except Exception as e:
    print(f"List failed: {e}")

print("Testing generation with 'gemini-3-pro-preview'...")
try:
    response = client.models.generate_content(
        model='gemini-3-pro-preview',
        contents="Test",
    )
    print("SUCCESS with gemini-3-pro-preview")
    print(f"Response: {response.text}")
except Exception as e2:
    print(f"Manual test failed: {e2}")
