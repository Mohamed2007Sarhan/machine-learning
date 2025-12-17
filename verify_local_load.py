
import joblib
from sklearn.linear_model import LogisticRegression
from model_module import load_local_model

# Save dummy model
model = LogisticRegression()
joblib.dump(model, "test_model.pkl")

# Load it back
loaded = load_local_model("test_model.pkl")

if loaded:
    print("Local Model Loading Successful!")
else:
    print("Local Model Loading Failed.")
