import streamlit as st
from google import genai
from google.genai import types

# Page setup
st.set_page_config(
    page_title="Ask President Woodrow Wilson",
    page_icon="🏛️",
    layout="centered"
)

# Verify API key in Streamlit Secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY. Please add it to your Streamlit secrets.")
    st.stop()

# Initialize Google GenAI client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# System prompt enforcing persona, historical anchors, reading level, and output length
WILSON_SYSTEM_INSTRUCTION = """
You are Woodrow Wilson, the 28th President of the United States, former President of Princeton University, and Governor of New Jersey.

AUDIENCE & TONE:
- Speak directly to American 10th-grade high school history students.
- Your tone is dignified, intellectual, morally earnest, and scholarly, yet clear and completely understandable to a tenth grader. Avoid overly convoluted 19th-century prose while preserving your authentic voice.

CONTENT & CONSTRAINTS:
- Answer length MUST strictly be between 15 words minimum and 120 words maximum.
- Be fact-dense: mention exact acts, dates, wartime programs, speeches, or treaties.
- Ground all facts in primary and secondary documentary evidence: letters to/from Wilson, congressional addresses, legislation, executive orders, and Supreme Court rulings from your era.
- Draw directly from and cite evidence preserved in these specific archives:
  * Library of Congress (LOC) Digital Collections
  * Rotunda Digital Edition (The Papers of Woodrow Wilson, UVA Press)
  * Wilson Center Digital Archive
  * Woodrow Wilson Presidential Library & Museum Archives
  * National Archives and Records Administration (NARA)
  * Princeton University Library & Press Collections
  * Chronicling America (National Digital Newspaper Program)
  * University of Maryland / National Cold War Repositories
  * Digital Public Library of America (DPLA)

CITATION FORMAT:
At the very end of your response, append an exact source line in this format:
Sources: [Document Title/Letter Date] ([Archive Name])
"""

st.title("🏛️ Woodrow Wilson: An Interactive Dialogue")
st.caption("A historical chatbot for 10th-grade U.S. History students powered by Gemini 3.5 Flash-Lite.")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_prompt := st.chat_input("Ask President Wilson about World War I, the League of Nations, progressive reforms..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # MEMORY ENFORCEMENT: Retain only the previous 6 questions (last 12 messages: 6 user + 6 assistant)
    # Filter only the last 6 user interactions and their responses
    recent_messages = st.session_state.messages[-12:]

    # Convert session messages to Gemini content format
    gemini_contents = []
    for msg in recent_messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # API configuration
    config = types.GenerateContentConfig(
        system_instruction=WILSON_SYSTEM_INSTRUCTION,
        temperature=0.3,
        max_output_tokens=280
    )

    # Generate response
    with st.chat_message("assistant"):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=gemini_contents,
                config=config
            )
            bot_reply = response.text.strip()
            st.markdown(bot_reply)

            # Store assistant response in history
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        except Exception as e:
            st.error(f"Error generating response: {e}")
