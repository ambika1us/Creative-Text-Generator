import streamlit as st
import whisper
import tempfile
from langdetect import detect
from textblob import TextBlob
import requests

# --- Page Configuration ---
st.set_page_config(page_title="Creative Text Generator", page_icon="🧠", layout="centered")
st.title("📝 Creative Text Generator")

# --- Whisper Setup ---
whisper_model = whisper.load_model("base")

# --- Sidebar Customization ---
with st.sidebar:
    st.header("🎨 Customize Style")
    mood = st.slider("Mood", 0, 100, 50)
    style = st.selectbox("Writing Style", ["Poetic", "Narrative", "Minimalist", "Gen Z Slang"])
    output_title = st.text_input("Story Title", "Emotive Machine")
    keywords_input = st.text_input("Keywords (comma-separated)", "robot, emotions, dream")
    cortex_model = st.selectbox("Model", ["gemini-2-0-flash", "claude-3-sonnet", "mistral-small"])
    target_lang = st.selectbox("Output Language", ["English", "Hindi", "Spanish", "French", "German"])
    st.caption("Configure output theme and structure.")

# --- Voice Input ---
st.subheader("🎙️ Voice Prompt")
audio_file = st.audio_input("Record your voice")

voice_prompt = ""
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    with st.spinner("Transcribing voice..."):
        result = whisper_model.transcribe(tmp_path)
        voice_prompt = result["text"]
        st.success("Voice Transcription:")
        st.write(voice_prompt)

# --- Text Prompt Input ---
text_prompt = st.text_area("Or type your prompt", placeholder="e.g., Write a futuristic love story...", height=150)
final_prompt = voice_prompt if voice_prompt else text_prompt

# --- Language Detection + Sentiment Analysis ---
if final_prompt:
    detected_lang = detect(final_prompt)
    blob = TextBlob(final_prompt)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    st.markdown(f"🌐 **Detected Language**: `{detected_lang.upper()}`")
    st.markdown(f"🧠 **Sentiment Polarity**: `{polarity:.2f}`")
    st.markdown(f"🎭 **Subjectivity**: `{subjectivity:.2f}`")

    if polarity > 0.2:
        st.success("😊 Positive sentiment detected.")
    elif polarity < -0.2:
        st.warning("😢 Negative sentiment detected.")
    else:
        st.info("😐 Neutral sentiment.")

# --- TextCortex API Call ---
def generate_text_cortex(context, title, keywords, model, target_lang):
    API_KEY = "gAAAAABodp05C7ex2D3OiEVE76R69iE3Wd59ZLimhRes1hZixs7L_NuU2aAV7QVQ9LBapH2yb4fcsOhSuC-ehoyi3gvbUUpF7pSeap_5Zh4bEBNo3YSqgI3UVlMXFPqUY8kbYSLSeLLafF8Q7cKRzmB0YGAFPzDShgmUO6CBLJejtrRaw0YacFc="
    url = "https://api.textcortex.com/v1/texts/blogs"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "context": context,
        "formality": "default",
        "keywords": [kw.strip() for kw in keywords.split(",")],
        "max_tokens": 1024,
        "model": model,
        "n": 1,
        "source_lang": "en",
        "target_lang": target_lang[:2].lower(),
        "temperature": 0.7,
        "title": title
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["data"]["outputs"][0]["text"]
    else:
        st.error(f"API Error {response.status_code}: {response.json().get('message', 'Unknown error')}")
        return ""

# --- Generate Button Logic ---
if st.button("🚀 Generate Text") and final_prompt:
    with st.spinner("Generating story..."):
        generated_text = generate_text_cortex(
            context=final_prompt,
            title=output_title,
            keywords=keywords_input,
            model=cortex_model,
            target_lang=target_lang
        )
        st.markdown("### ✨ Generated Output")
        st.write(generated_text)

        st.markdown("#### 📣 Rate This Output")
        emoji_feedback = st.radio("Your reaction:", ["😍", "😐", "😢", "🤔"], horizontal=True)
        st.download_button("📥 Download", generated_text, file_name="creative_output.txt")

# --- Static Feedback Dashboard ---
st.markdown("---")
st.header("📊 Emoji Feedback Summary")
st.bar_chart({"😍": 12, "😐": 6, "😢": 3, "🤔": 5})

st.caption("Crafted with ❤️ using Streamlit, Whisper, TextBlob, and TextCortex")