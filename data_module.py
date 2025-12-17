from google import genai
from google.genai import types
import pandas as pd
import io

# Configure Gemini
API_KEY = ""

def preprocess_data(raw_text, topic=""):
    """
    Uses Gemini to process raw text into a structured DataFrame.
    """
    print("Preprocessing data with Gemini (google-genai SDK)...")
    
    try:
        client = genai.Client(api_key=API_KEY)
        
        prompt = f"""
        Analyze the following text and extract a structured dataset suitable for machine learning.
        Output ONLY a CSV format with headers. 
        Ensure the data is clean and relevant to the text.
        If no clear data is found, create a synthetic dataset based on the topic: "{topic}".
        Text:
        {raw_text[:50000]} 
        """ 
        
        # Helper function for generating with retries
        import time
        import re

        def generate_with_retry(model_name, prompt, retries=5):
            for attempt in range(retries):
                try:
                    print(f"Attempting to use model: {model_name} (Attempt {attempt+1}/{retries})")
                    return client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        # Extract wait time if available, defaulting to 20s increasing
                        wait_time = 20 * (attempt + 1)
                        # Try to find specific retry time in error message
                        match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                        if match:
                            wait_time = float(match.group(1)) + 2 # Add buffer
                        
                        print(f"Rate limited on {model_name}. Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                    elif "503" in error_str or "UNAVAILABLE" in error_str:
                         print(f"Model {model_name} overloaded. Waiting 10s...")
                         time.sleep(10)
                    else:
                        raise e # Re-raise if not a temporary error
            raise Exception(f"Failed to generate with {model_name} after {retries} retries.")

        # List of models to try in order of preference
        models_candidate_list = [
            'gemini-3-pro-preview',
            'gemini-2.5-flash', 
            'gemini-2.0-flash', 
            'gemini-flash-latest', # Valid fallback found in listing
        ]
        
        response = None
        last_error = None
        
        for model_name in models_candidate_list:
            try:
                response = generate_with_retry(model_name, prompt, retries=2)
                if response:
                    break
            except Exception as e:
                print(f"Model {model_name} failed completely. Moving to next.")
                last_error = e
                
        if not response:
             print("All models failed.")
             raise last_error
        
        text_data = response.text
        
        # Clean up markdown code blocks
        if "```csv" in text_data:
            text_data = text_data.split("```csv")[1].split("```")[0]
        elif "```" in text_data:
            text_data = text_data.split("```")[1].split("```")[0]
            
        # Parse to DataFrame
        df = pd.read_csv(io.StringIO(text_data.strip()))
        print("Data successfully processed.")
        return df
        
    except Exception as e:
        print(f"Data preprocessing failed: {e}")
        return pd.DataFrame()

