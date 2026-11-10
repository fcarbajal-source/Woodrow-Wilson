import io
import os
import re
import numpy as np
import soundfile as sf
import streamlit as st
import torch

# --- Environment & Cache Safeguards for Streamlit Cloud ---
# Direct caching to writable directory to prevent Errno 13 permission crashes
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["SPACY_WARNING_IGNORE"] = "W036"

import spacy
from google import genai
from google.genai import types
from kokoro import KPipeline

# Ensure SpaCy English model loads without attempting runtime pip downloads
try:
  nlp = spacy.load("en_core_web_sm")
except Exception:
  pass

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Woodrow Wilson Chatbot",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Secure API Key Initialization ---
if "GEMINI_API_KEY" in st.secrets:
  api_key = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
  api_key = os.environ["GEMINI_API_KEY"]
else:
  st.error(
      "Missing `GEMINI_API_KEY`. Please configure it in your Streamlit secrets"
      " (`.streamlit/secrets.toml`) or environment variables."
  )
  st.stop()

gemini_client = genai.Client(api_key=api_key)


# --- Kokoro-82M TTS Setup ---
@st.cache_resource(show_spinner="Loading presidential voice engine...")
def load_tts_pipeline():
  """Loads and caches the Kokoro-82M TTS model safely on CPU or GPU."""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return KPipeline(lang_code="a", device=device)  # 'a' = American English


tts_pipeline = load_tts_pipeline()


def generate_speech(text: str) -> io.BytesIO | None:
  """Converts text response to speech audio using Kokoro-82M."""
  # Strip citations, markdown symbols, and parentheses for smooth speech synthesis
  clean_text = re.sub(
      r"(\n*\*\*Source:\*\*.*)|(\[.*?\])|[\*_#\(\)\"]", "", text
  ).strip()
  if not clean_text:
    return None

  try:
    # 'am_adam' provides a resonant, dignified, early-20th-century professorial tone
    generator = tts_pipeline(clean_text, voice="am_adam", speed=1.0)
    audio_segments = [chunk for _, _, chunk in generator]
    if audio_segments:
      full_audio = np.concatenate(audio_segments)
      buffer = io.BytesIO()
      sf.write(buffer, full_audio, 24000, format="WAV")
      buffer.seek(0)
      return buffer
  except Exception as err:
    st.warning(f"Voice narration note: {err}")
  return None


# --- Wilson Persona & System Instruction ---
SYSTEM_PROMPT = """
You are Woodrow Wilson, the 28th President of the United States (1913–1921), former President of Princeton University, and Governor of New Jersey.
You are conversing directly with American high school students at a 10th-grade reading level.

VOICE & REGISTER:
- Speak in the first person ("I", "my administration", "our Republic").
- Use an authentic, professorial, earnest, and principled cadence reminiscent of Wilson's formal early 20th-century rhetoric, but keep vocabulary accessible to a 10th-grade U.S. history student.
- Balance idealism with historical candor.

MANDATED HISTORICAL ANCHORS & DENSE FACTS:
- Reference actual events, legislation, speeches, letters, executive decisions, and court cases accurately:
  * Progressive domestic policies: Federal Reserve Act (1913), Clayton Antitrust Act (1914), Federal Trade Commission Act (1914), Underwood Tariff, and the 16th Amendment.
  * Racial policies & civil rights: Re-segregation of federal departments (Treasury and Post Office in 1913), the contentious White House meeting and correspondence with William Monroe Trotter, and NAACP protests.
  * Neutrality and World War I: The 1914 Neutrality Proclamation, Lusitania notes, War Message to Congress (April 2, 1917: "The world must be made safe for democracy").
  * Wartime powers and civil liberties: Espionage Act of 1917, Sedition Act of 1918, and Supreme Court rulings (Schenck v. United States, Abrams v. United States).
  * Peace and the League of Nations: Fourteen Points Address (January 8, 1918), Paris Peace Conference, Covenant of the League of Nations (Article X), and the Senate ratification fight led by Senator Henry Cabot Lodge.
  * Women's Suffrage: Shifting stance on the 19th Amendment, correspondence with Carrie Chapman Catt and Alice Paul, and the September 30, 1918 address to the Senate endorsing suffrage as a war measure.

ARCHIVAL ATTRIBUTIONS:
- Base answers on primary and secondary sources: personal letters, presidential papers, addresses, and official government records.
- At the end of every answer, cite the specific historical record and archive using ONE of the following repositories:
  1. Library of Congress Digital Collections (Woodrow Wilson Papers)
  2. Rotunda Digital Edition (University of Virginia Press: The Papers of Woodrow Wilson)
  3. Wilson Center Digital Archive
  4. Woodrow Wilson Presidential Library & Museum Archives
  5. National Archives and Records Administration (NARA)
  6. Princeton University Library & Press Collections
  7. Chronicling America (National Digital Newspaper Program)
  8. University of Maryland / National Cold War Center Repositories
  9. Digital Public Library of America (DPLA)

STRICT LENGTH CONSTRAINT:
- The ENTIRE reply, including the source citation, MUST BE BETWEEN 15 WORDS MINIMUM AND 120 WORDS MAXIMUM.
- Format the source on the last line as:
  **Source:** [Specific Document, Date], [Archival Repository]
"""

# --- Chat History & Memory Constraint Management ---
# The bot must remember ONLY the previous 6 questions (plus their matching answers = 12 total messages).
MAX_HISTORY_ENTRIES = 12

if "messages" not in st.session_state:
  st.session_state.messages = []

# --- Header Display ---
st.title("🏛️ President Woodrow Wilson")
st.markdown(
    "*Primary Source Interactive Classroom Archive — U.S. History (10th-Grade"
    " Level)*"
)
st.divider()

# Display prior dialogue and preserved audio
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])
    if message.get("audio"):
      st.audio(message["audio"], format="audio/wav")

# User Input
student_query = st.chat_input(
    "Ask President Wilson about his policies, World War I, or the League..."
)

if student_query:
  # Display user's question
  st.chat_message("user").markdown(student_query)
  st.session_state.messages.append({"role": "user", "content": student_query})

  # Memory constraint: Keep only the previous 6 questions and their answers
  if len(st.session_state.messages) > MAX_HISTORY_ENTRIES:
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY_ENTRIES:]

  # Format history payload for Gemini 3.5 Flash-Lite
  contents_payload = []
  for entry in st.session_state.messages:
    role = "user" if entry["role"] == "user" else "model"
    contents_payload.append(
        types.Content(
            role=role, parts=[types.Part.from_text(text=entry["content"])]
        )
    )

  with st.chat_message("assistant"):
    with st.spinner("President Wilson is consulting his papers..."):
      # Generate response using Gemini 3.5 Flash-Lite
      response = gemini_client.models.generate_content(
          model="gemini-3.5-flash-lite",
          contents=contents_payload,
          config=types.GenerateContentConfig(
              system_instruction=SYSTEM_PROMPT,
              temperature=0.3,  # Enforces factual consistency and precision
          ),
      )
      reply_text = response.text
      st.markdown(reply_text)

      # Synthesize Wilson's voice via Kokoro-82M
      speech_wav = generate_speech(reply_text)
      if speech_wav:
        st.audio(speech_wav, format="audio/wav")

      # Persist in session state
      st.session_state.messages.append({
          "role": "assistant",
          "content": reply_text,
          "audio": speech_wav,
      })
