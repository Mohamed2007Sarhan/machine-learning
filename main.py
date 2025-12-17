import sys
import pandas as pd
from search_module import interactive_approval
from data_module import preprocess_data
from model_module import analyze_dataset, select_models, train_and_evaluate

def main():
    print("=================================================")
    print("   Machine Learning Workflow Generator (MLWG)    ")
    print("=================================================")
    
    # 1. Input and Web Research
    topic = input("\nEnter a topic or query to research (e.g., 'California Housing Prices csv'): ")
    raw_content = interactive_approval(topic)
    
    if not raw_content:
        print("No content approved. Exiting.")
        return

    # 2. Data Preprocessing
    print("\n[Step 2] Processing data with Gemini...")
    df = preprocess_data(raw_content, topic=topic)
    
    if df.empty:
        print("Failed to extract data. Exiting.")
        return
        
    print("\n--- Data Preview ---")
    print(df.head())
    print("--------------------")

    # 3. Model Selection
    print("\n[Step 3] analyzing dataset...")
    size_category, problem_type, target_col = analyze_dataset(df)
    
    available_models = select_models(size_category, problem_type)
    print(f"\nRecommended Models for {size_category} / {problem_type}:")
    model_names = list(available_models.keys())
    for idx, name in enumerate(model_names):
        print(f"{idx + 1}. {name}")
        
    choice = input(f"\nSelect a model (1-{len(model_names)}): ")
    try:
        selected_model_name = model_names[int(choice) - 1]
        model_instance = available_models[selected_model_name]
    except (ValueError, IndexError):
        print("Invalid selection. Defaulting to first option.")
        selected_model_name = model_names[0]
        model_instance = available_models[selected_model_name]
        
    # 4 & 5. Training and Evaluation
    trained_model, X_test, y_test = train_and_evaluate(df, target_col, selected_model_name, model_instance)
    
    # Interactive Testing
    while True:
        try:
            test_input = input("\nDo you want to test the model with custom input? (yes/no): ").lower()
            if test_input not in ['yes', 'y']:
                break
                
            print(f"Enter values for features (comma separated): {list(X_test.columns)}")
            values_str = input("> ")
            values = [float(x.strip()) for x in values_str.split(',')]
            
            if len(values) != len(X_test.columns):
                print(f"Error: Expected {len(X_test.columns)} values, got {len(values)}")
                continue
                
            prediction = trained_model.predict(pd.DataFrame([values], columns=X_test.columns))
            print(f"Prediction: {prediction[0]}")
            
        except Exception as e:
            print(f"Error during testing: {e}")

    print("\nWorkflow completed.")

if __name__ == "__main__":
    main()
