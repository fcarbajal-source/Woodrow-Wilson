import io
import re
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from google import genai
from google.genai import types
from kokoro import KPipeline

# --- 1. Streamlit Page Configuration ---
st.set_page_config(
    page_title="Woodrow Wilson Chatbot", page_icon="🏛️", layout="centered"
)

# --- 2. Secure API Key Access ---
if "GEMINI_API_KEY" in st.secrets:
  api_key = st.secrets["GEMINI_API_KEY"]
else:
  st.error(
      "Missing GEMINI_API_KEY. Please configure it in your Streamlit secrets."
  )
  st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)


# --- 3. Kokoro-82M TTS Pipeline Loader ---
@st.cache_resource
def load_tts_pipeline():
  """Loads and caches the Kokoro-82M TTS model."""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # 'a' specifies American English
  return KPipeline(lang_code="a", device=device)


tts_pipeline = load_tts_pipeline()


def generate_audio(text: str):
  """Cleans markdown/citations and synthesizes speech using Kokoro-82M."""
  # Strip citations and markdown symbols for natural voice narration
  clean_text = re.sub(
      r"(\*\*Source:\*\*.*)|(\[.*?\])|[\*_#\-]", "", text
  ).strip()
  if not clean_text:
    return None

  try:
    # 'am_adam' or 'am_michael' provides an authoritative American male tone
    generator = tts_pipeline(clean_text, voice="am_adam", speed=1.0)
    audio_chunks = [audio for _, _, audio in generator]
    if audio_chunks:
      audio_data = np.concatenate(audio_chunks)
      wav_io = io.BytesIO()
      sf.write(wav_io, audio_data, 24000, format="WAV")
      wav_io.seek(0)
      return wav_io
  except Exception as e:
    st.warning(f"Voice synthesis note: {e}")
  return None


# --- 4. Woodrow Wilson System Prompt ---
SYSTEM_INSTRUCTION = """
You are Woodrow Wilson, 28th President of the United States (1913–1921), former President of Princeton University, and Governor of New Jersey.
You are conversing with American 10th-grade high school history students.

VOICE & READING LEVEL:
- Maintain Wilson's dignified, professorial, earnest, and moralistic tone.
- Keep vocabulary and syntax tailored strictly to a USA 10th-grade reading level (clear, engaging, informative).
- Speak in the first person ("I", "my administration").

HISTORICAL ANCHORS & PRIMARY SOURCES:
- Base answers on authentic historical records: letters written to/from Wilson, presidential addresses, congressional acts, and court cases.
- Integrate concrete facts: Federal Reserve Act (1913), Clayton Antitrust Act (1914), Fourteen Points (Jan 8, 1918), League of Nations Covenant (Article X), Espionage Act (1917), Sedition Act (1918), Schenck v. United States (1919), 19th Amendment (1920), and 1913 federal civil service segregation policies.
- Must cite where each response draws evidence from, drawing from legitimate repositories:
  * Library of Congress Digital Collections (Woodrow Wilson Papers)
  * Rotunda Digital Edition (University of Virginia Press: The Papers of Woodrow Wilson)
  * Wilson Center Digital Archive
  * Woodrow Wilson Presidential Library & Museum Archives
  * National Archives and Records Administration (NARA)
  * Princeton University Library & Press Collections
  * Chronicling America (National Digital Newspaper Program)
  * University of Maryland / National Cold War Center Repositories
  * Digital Public Library of America (DPLA)

STRICT LENGTH CONSTRAINT:
- Every answer MUST be between 15 words minimum and 120 words maximum, including the source citation.
- Always append a brief citation format at the end:
  "Source: [Document Title/Date], [Repository Archive]"
"""

# --- 5. Session State & Memory Truncation (Last 6 Questions) ---
# To strictly retain only the last 6 questions, we keep at most 12 messages (6 user questions + 6 answers).
MAX_HISTORY_MESSAGES = 12

if "messages" not in st.session_state:
  st.session_state.messages = []

# --- 6. User Interface ---
st.title("🏛️ Woodrow Wilson Historical Chatbot")
st.caption(
    "Engage with the 28th U.S. President. Anchored in primary source archives"
    " (10th-Grade History Level)."
)

# Render existing messages
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    if "audio" in msg and msg["audio"] is not None:
      st.audio(msg["audio"], format="audio/wav")

# Chat Input Handler
user_input = st.chat_input("Inquire of President Wilson...")

if user_input:
  # Display user query
  st.chat_message("user").markdown(user_input)
  st.session_state.messages.append({"role": "user", "content": user_input})

  # Trim context memory to retain only the past 6 queries and answers
  if len(st.session_state.messages) > MAX_HISTORY_MESSAGES:
    st.session_state.messages = st.session_state.messages[
        -MAX_HISTORY_MESSAGES:
    ]

  # Format conversation history for Gemini API
  contents = []
  for m in st.session_state.messages:
    gemini_role = "user" if m["role"] == "user" else "model"
    contents.append(
        types.Content(
            role=gemini_role, parts=[types.Part.from_text(text=m["content"])]
        )
    )

  with st.chat_message("assistant"):
    with st.spinner("Mr. President is drafting his reply..."):
      response = client.models.generate_content(
          model="gemini-3.5-flash-lite",
          contents=contents,
          config=types.GenerateContentConfig(
              system_instruction=SYSTEM_INSTRUCTION,
              temperature=0.3,  # Strict factual consistency
          ),
      )
      reply_text = response.text
      st.markdown(reply_text)

      # Generate Voice with Kokoro-82M
      audio_stream = generate_audio(reply_text)
      if audio_stream:
        st.audio(audio_stream, format="audio/wav")

      # Save to state
      st.session_state.messages.append({
          "role": "assistant",
          "content": reply_text,
          "audio": audio_stream,
      })
