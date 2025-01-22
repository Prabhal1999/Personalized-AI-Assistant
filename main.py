import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch API keys
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate API keys
if not langchain_api_key or not groq_api_key:
    st.error("API keys are not available. Please ensure LANGCHAIN_API_KEY and GROQ_API_KEY are set in your environment.")
    st.stop()

# Set environment variables for LangChain tracing and project metadata
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Personalized AI Assistant"

# Define a ChatPromptTemplate for the model
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the queries politely in {language}."),
    ("human", "{conversation}")
])

# Function to generate a response
def generate_response(conversation_history, temperature, max_tokens, language):
    try:
        # Format the conversation history
        formatted_history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history
        )

        # Initialize the ChatGroq model
        llm = ChatGroq(model="llama-3.3-70b-versatile")
        output_parser = StrOutputParser()

        # Combine the prompt, model, and output parser
        chain = prompt | llm | output_parser

        # Generate response using the chain
        response = chain.invoke({"conversation": formatted_history, "language": language})
        return response
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        raise RuntimeError("Failed to generate response")

# Streamlit App
st.title("Personalized AI Assistant🔍")

# Sidebar settings
st.sidebar.title("Settings")
temperature = st.sidebar.slider("Temperature (creativity)", min_value=0.0, max_value=1.0, value=0.5)
max_tokens = st.sidebar.slider("Max Tokens (response length)", min_value=50, max_value=500, value=200)
language = st.sidebar.selectbox("Select Language", ["English", "Hindi", "French", "German", "Italian", "Portuguese","Spanish","Thai"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input field
user_input = st.chat_input(placeholder="Type your question here")

# Process user input
if user_input:
    # Append user input to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Generate and append AI response
    with st.spinner("Generating response..."):
        try:
            response = generate_response(
                conversation_history=st.session_state["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
                language=language
            )
            st.session_state["messages"].append({"role": "AI", "content": response})
        except RuntimeError as err:
            st.error(f"An error occurred: {err}")

# Display chat history
with st.container():
    for message in st.session_state["messages"]:  
        if message["role"] == "user":
            st.markdown(f"**You**: {message['content']}")
        else:
            st.markdown(f"**AI**: {message['content']}")


# Hide the "View Source" button and Streamlit footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Sidebar Information
st.sidebar.info(
    "✨ **Welcome to Your Personalized AI Assistant!**\n"
    "This app is designed to provide intelligent, multilingual assistance for your queries. 🌍💬\n\n"
    "🎯 **Key Features:**\n"
    "- Supports multiple languages for global reach.\n"
    "- Tailored responses with adjustable creativity.\n"
    "- Polite, helpful, and always learning. 🤖\n\n"
    "🚀 **What's Next?**\n"
    "Stay tuned for upcoming features like voice input, expanded language support, and smarter, faster responses!"
)
