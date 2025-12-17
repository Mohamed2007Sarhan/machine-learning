import streamlit as st
import pandas as pd
from search_module import search_for_topic, aggregate_content
from data_module import preprocess_data
from model_module import analyze_dataset, select_models, train_and_evaluate

st.set_page_config(page_title="Auto-ML Workflow", layout="wide")

st.title("🤖 ML Workflow Generator")
st.markdown("Automate research, data collection, and model training.")

# Session State for persistency
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = ""
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_features' not in st.session_state:
    st.session_state.model_features = []
if 'search_multiplier' not in st.session_state:
    st.session_state.search_multiplier = 1 # To track retries/rejection

# --- Step 1: Research ---
st.header("1. 🔍 Research & Data Gathering")
topic = st.text_input("Enter a topic (e.g., 'California Housing Prices dataset'):")

col_search, col_reset = st.columns([1, 5])
with col_search:
    if st.button("Search & Aggregate"):
        st.session_state.step = 1 # Reset if new search
        st.session_state.search_multiplier = 1
        with st.spinner("Searching and scraping..."):
            urls = search_for_topic(topic, max_results=20)
            if urls:
                content = aggregate_content(urls)
                st.session_state.raw_data = content
                st.success("Data found!")
            else:
                st.error("No results found.")

if st.session_state.raw_data:
    st.subheader("Results Review")
    
    # Manual Input Section
    st.write("Does this look good? You can add your own data below if needed.")
    manual_input = st.text_area("Add Custom Data/Context (Optional):", height=100, placeholder="Paste your own CSV content or extra info here...")
    
    # Text Editor for extracted content (ReadOnlyish but viewable)
    st.text_area("Extracted Content (editable):", value=st.session_state.raw_data, height=400, key="editor_raw_data")
    
    col_approve, col_reject = st.columns(2)
    
    with col_approve:
        if st.button("✅ Approve & Process"):
            # Append manual input if any
            if manual_input:
                st.session_state.raw_data += f"\n\n--- User Manual Input ---\n{manual_input}"
            # Update with edited text area content
            if st.session_state.editor_raw_data != st.session_state.raw_data:
                 st.session_state.raw_data = st.session_state.editor_raw_data
                 
            st.session_state.step = 2
            st.rerun()

    with col_reject:
        if st.button("❌ Reject & Find More"):
            st.session_state.search_multiplier += 1
            with st.spinner(f"Trying harder (Attempt {st.session_state.search_multiplier})..."):
                # Pass a larger max_results or offset logic if supported, 
                # for now we search for Variation
                refined_query = f"{topic} dataset examples table" 
                urls = search_for_topic(refined_query, max_results=20 * st.session_state.search_multiplier)
                if urls:
                    new_content = aggregate_content(urls)
                    st.session_state.raw_data = new_content
                    st.success(f"Fetched new data (Attempt {st.session_state.search_multiplier})!")
                    st.rerun()
                else:
                    st.error("No new results found even with expanded search.")

# --- Step 2: Preprocessing ---
if st.session_state.step >= 2:
    st.header("2. 🧹 Data Preprocessing (Gemini)")
    
    if st.session_state.df.empty:
        with st.spinner("Gemini is structuring your data..."):
            df = preprocess_data(st.session_state.raw_data, topic=topic)
            if not df.empty:
                st.session_state.df = df
                st.success("Data processed successfully!")
            else:
                st.error("Failed to generate dataframe. Try adding manual data.")
    
    if not st.session_state.df.empty:
        st.subheader("Dataset Preview")
        edited_df = st.data_editor(st.session_state.df) # Allow manual fixes
        st.session_state.df = edited_df
        
        size_cat, prob_type, target = analyze_dataset(edited_df)
        st.info(f"Detected: **{size_cat}** dataset, **{prob_type}** problem. Target: `{target}`")
        
        # --- Step 3: Model Selection ---
        st.header("3. 🧠 Model Selection & Training")
        
        # Model Source Selection
        model_source = st.radio("Select Model Source:", ["Standard (Sklearn/XGBoost)", "Hugging Face (Transformers)", "Load Local Model"])
        
        if model_source == "Standard (Sklearn/XGBoost)":
            models = select_models(size_cat, prob_type)
            model_name = st.selectbox("Choose a model:", list(models.keys()))
            
            if st.button("Train Model"):
                with st.spinner("Training..."):
                    model_inst = models[model_name]
                    trained_model, X_test, y_test, metrics = train_and_evaluate(st.session_state.df, target, model_name, model_inst)
                    
                    st.session_state.trained_model = trained_model
                    st.session_state.model_features = X_test.columns.tolist()
                    
                    st.success("Training Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    if 'Test Accuracy' in metrics:
                        col1.metric("Test Accuracy", f"{metrics['Test Accuracy']:.4f}")
                    if 'Test MSE' in metrics:
                        col2.metric("Test MSE", f"{metrics['Test MSE']:.4f}")
                    if 'Test R2' in metrics:
                        col3.metric("Test R2", f"{metrics['Test R2']:.4f}")

        elif model_source == "Hugging Face (Transformers)":
            st.info("Uses Hugging Face for tabular/text classification. Treats row data as text sequence.")
            hf_model_name = st.text_input("Enter HF Model ID:", "distilbert-base-uncased")
            
            if st.button("Train HF Model"):
                from model_module import train_hf_model
                with st.spinner(f"Fine-tuning {hf_model_name}... (This may take time)"):
                    try:
                        trained_model, _, _, metrics = train_hf_model(st.session_state.df, target, hf_model_name)
                        st.session_state.trained_model = trained_model # Trainer object
                        st.session_state.model_features = "text_mode" # Marker
                        st.success("HF Training Complete!")
                        st.write(metrics)
                    except Exception as e:
                        st.error(f"HF Training Failed: {e}")

        elif model_source == "Load Local Model":
            uploaded_file = st.file_uploader("Upload Model File (.pkl, .joblib)", type=['pkl', 'joblib'])
            if uploaded_file is not None:
                import joblib
                try:
                    loaded_model = joblib.load(uploaded_file)
                    st.session_state.trained_model = loaded_model
                    st.session_state.model_features = st.session_state.df.drop(columns=[target]).columns.tolist() # Guess features
                    st.success("Model Loaded Successfully!")
                except Exception as e:
                    st.error(f"Failed to load model: {e}")

# --- Step 4: Interactive Chat & Inference ---
if st.session_state.trained_model:
    st.header("4. 💬 Chat with your Model")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Display Chat History
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
            
    # Input Area
    st.subheader("New Prediction")
    
    # Text-based input for HF
    if st.session_state.model_features == "text_mode":
        user_input = st.chat_input("Enter text to classify...")
        if user_input:
            # User Message
            st.session_state.chat_history.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)
                
            # Model Prediction
            with st.spinner("Model is thinking..."):
                try:
                    # Simplified inference for HF - reusing session model if possible
                    # Ideally we reload pipeline for robustness, but for now:
                    pred = st.session_state.trained_model.predict([user_input]) 
                    # pred might be complex object depending on trainer, simplifying for demo
                    response = f"**Prediction:** {pred}"
                except Exception as e:
                    response = f"Error: {e}. (Ensure model accepts text list input)"
            
            st.session_state.chat_history.append(("assistant", response))
            with st.chat_message("assistant"):
                st.markdown(response)

    # Form-based input for Tabular (Simulated Chat)
    else:
        with st.form("tabular_chat_form"):
            st.write("Enter values (simulated chat input):")
            input_data = {}
            cols = st.columns(min(len(st.session_state.model_features), 4))
            
            for idx, col in enumerate(st.session_state.model_features):
                with cols[idx % 4]:
                    val = st.text_input(f"{col}", "0")
                    try:
                        input_data[col] = float(val) if val else 0.0
                    except ValueError:
                        input_data[col] = 0.0
            
            submitted = st.form_submit_button("Ask Model")
            
            if submitted:
                # Construct "User Message" representation
                user_msg = "**Input Features:**\n" + "\n".join([f"- {k}: {v}" for k,v in input_data.items()])
                st.session_state.chat_history.append(("user", user_msg))
                # Display immediately (optional, or rerun handled)
                
                # Predict
                input_df = pd.DataFrame([input_data])
                input_df = input_df[st.session_state.model_features]
                pred = st.session_state.trained_model.predict(input_df)
                
                response = f"**Prediction:** {pred[0]:.4f}"
                st.session_state.chat_history.append(("assistant", response))
                
                st.rerun()
