import streamlit as st
from utils.model_utils import load_model, generate_story
import torch

# Set page config
st.set_page_config(
    page_title="Indian Folktales Generator",
    page_icon="📖",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load model with caching"""
    model, tokenizer, device = load_model("fine_tuned_folktales_gpt2")
    return model, tokenizer, device

def main():
    st.title("📖 Indian Folktales Generator")
    st.markdown("Generate creative Indian folktales using AI")
    
    # Load model
    try:
        model, tokenizer, device = load_models()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    # Sidebar controls
    st.sidebar.header("Generation Settings")
    max_length = st.sidebar.slider("Max Length", 50, 500, 200)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.8)
    
    # Prompt input
    prompt = st.text_area(
        "Enter your story prompt:",
        "Once upon a time in a small Indian village,"
    )
    
    # Generation button
    if st.button("Generate Story"):
        if not prompt.strip():
            st.warning("Please enter a prompt")
            return
            
        with st.spinner("Generating your folktale..."):
            try:
                story = generate_story(
                    prompt,
                    model,
                    tokenizer,
                    device,
                    max_length=max_length,
                    temperature=temperature
                )
                
                st.subheader("Generated Folktale")
                st.write(story)
                
                # Download button
                st.download_button(
                    label="Download Story",
                    data=story,
                    file_name="generated_folktale.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error generating story: {str(e)}")

if __name__ == "__main__":
    main()