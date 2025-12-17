
import pandas as pd
from model_module import train_hf_model

# Create dummy data
data = {
    'feature1': [1, 2, 3, 4, 5] * 4,
    'feature2': ['a', 'b', 'c', 'd', 'e'] * 4,
    'target': [0, 1, 0, 1, 0] * 4
}
df = pd.DataFrame(data)

print("Starting HF Training Test...")
try:
    trainer, _, _, metrics = train_hf_model(df, 'target', model_name="distilbert-base-uncased")
    print("HF Training Successful!")
    print(metrics)
except Exception as e:
    print(f"HF Training Failed: {e}")
