import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from educhain import Educhain, LLMConfig


# Streamlit Dark Theme
st.set_page_config(
    page_title="Educhain Visual Question Generator",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
    }
     [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    [data-testid="stToolbar"] {
        right: 2rem;
    }
    [data-testid="stTextInput"] > div > div > div {
        background-color: #1f2937;
        color: white;
    }
     [data-testid="stNumberInput"] > div > div > div {
        background-color: #1f2937;
        color: white;
         }
    [data-testid="stNumberInput"] input {
        color: white; /* Fix number input text color */
        }
    [data-testid="stButton"] button {
        background-color: #4CAF50;
        color: white;
    }
    [data-testid="stTextArea"] > div > div > div {
        background-color: #1f2937;
        color: white;
    }
    [data-testid="stSelectbox"] > div > div > div {
        background-color: #1f2937;
        color: white;
    }
     [data-testid="stSidebar"] {
        background-color: #1f2937;
        color: white;
    }
    [data-testid="stSidebar"] h1, h2, h3, h4, h5, h6, p, ul, ol, li {
        color: white;
    }
    [data-testid="stMarkdownContainer"] h1, h2, h3, h4, h5, h6, p, ul, ol, li {
        color: white;
    }
    [data-testid="stMarkdownContainer"] {
        color: white;
    }
   </style>
    """,
    unsafe_allow_html=True,
)


st.title("Educhain Visual Question Generator")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    llm_provider = st.selectbox("Choose LLM Provider", ["Gemini", "OpenRouter"])
    topic_input = st.text_input("Enter Question Topic", "GMAT Statistics")
    num_questions_input = st.number_input("Number of Questions", min_value=1, max_value=1000, value=2, step=1)

    # API Key input
    if llm_provider == "Gemini":
        gemini_api_key = st.text_input("Enter Google API Key", type="password")
    elif llm_provider == "OpenRouter":
        openrouter_api_key = st.text_input("Enter OpenRouter API Key", type="password")
    
generate_button = st.button("Generate Visual Questions")

if generate_button:
    topic = topic_input
    num_questions = num_questions_input
    llm = None
    config = None
    
    with st.spinner(f"Generating {num_questions} visual questions on '{topic}' using {llm_provider}..."):
        try:
            if llm_provider == "Gemini":
                if not gemini_api_key:
                    st.error("Please enter your Google API Key.")
                    st.stop()
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-exp-0827", google_api_key=gemini_api_key)
                config = LLMConfig(custom_model=llm)

            elif llm_provider == "OpenRouter":
                if not openrouter_api_key:
                    st.error("Please enter your OpenRouter API Key.")
                    st.stop()
                openrouter_model_name = "deepseek/deepseek-r1-distill-llama-70b"  # Or let user select in future
                openrouter_base_url = "https://openrouter.ai/api/v1"
                llm = ChatOpenAI(api_key=openrouter_api_key, model_name=openrouter_model_name, base_url=openrouter_base_url)
                config = LLMConfig(custom_model=llm)

            if config:
                client = Educhain(config)
                ques = client.qna_engine.generate_visual_questions(topic=topic, num=num_questions)

                if ques and ques.questions:
                    st.success(f"Successfully generated {len(ques.questions)} visual questions!")
                    for q_data in ques.questions:
                        st.markdown("---")
                        st.subheader(f"Question: {q_data.question}")

                        # Display Visualization - Assuming _generate_and_save_visual is accessible
                        instruction = q_data.graph_instruction.dict()
                        question_text = q_data.question
                        options = q_data.options
                        correct_answer = q_data.answer

                        try:
                            img_base64 = client.qna_engine._generate_and_save_visual(instruction, question_text, options, correct_answer)
                            if img_base64:
                                st.image(f"data:image/png;base64,{img_base64}", width=400) # Adjust width as needed
                            else:
                                st.warning("Visualization could not be generated.")
                        except Exception as viz_err:
                            st.error(f"Error displaying visualization: {viz_err}")

                        options_display = ""
                        for idx, option in enumerate(options, start=1):
                            options_display += f"{chr(64+idx)}. {option}  \n"
                        st.markdown(f"**Options:**  \n{options_display}")
                        st.markdown(f"**Correct Answer:** {correct_answer}")
                        if q_data.explanation:
                            st.markdown(f"**Explanation:** {q_data.explanation}")

                else:
                    st.error("Failed to generate visual questions. Please check settings and try again.")

        except Exception as e:
            st.error(f"An error occurred during question generation: {e}")

st.markdown("---")
st.markdown("Powered by Educhain")
