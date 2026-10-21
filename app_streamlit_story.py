import streamlit as st
from PIL import Image
import io, os
import textwrap
from gtts import gTTS
from transformers import BlipProcessor, BlipForConditionalGeneration
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import google.generativeai as genai
import torch

# Environment Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY environment variable is not set. Please set it before generating stories.")
    st.stop()

# Streamlit Page Setup
st.set_page_config(
    page_title="StoryStudio AI | Multi-Modal Narrative Suite",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

    :root {
        --primary: #4F46E5;
        --primary-hover: #4338CA;
        --primary-light: #EEF2FF;
        --surface: #FFFFFF;
        --surface-subtle: #F8FAFC;
        --text-main: #0F172A;
        --text-muted: #64748B;
        --border-color: #E2E8F0;
        --radius-lg: 16px;
        --radius-md: 12px;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, #F1F5F9 0%, #F8FAFC 80%);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-main);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 12px rgba(0,0,0,0.02);
    }

    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: var(--text-main) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hero Header */
    .hero-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--primary-light);
        color: var(--primary);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 9999px;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 0.6rem;
    }

    .hero-title span {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1.1rem;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 6px;
        border-radius: var(--radius-md);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        color: var(--text-muted);
        background-color: transparent;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: var(--primary) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.7rem 1.8rem !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
    }

    .stDownloadButton > button {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        border-color: var(--primary) !important;
        color: var(--primary) !important;
        background: var(--primary-light) !important;
    }

    /* Story Box */
    .story-output-box {
        background: #FFFFFF;
        border-left: 4px solid var(--primary);
        border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
        padding: 2rem;
        box-shadow: var(--shadow-md);
        font-size: 1.08rem;
        line-height: 1.8;
        color: #1E293B;
        font-family: 'Playfair Display', serif;
    }

    /* Uploader */
    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 2px dashed #CBD5E1;
        border-radius: var(--radius-lg);
        padding: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Voice Definitions
VOICE_OPTIONS = {
    "Default English (US Female)": {"lang": "en", "tld": "com", "slow": False},
    "British Accent (UK Female)": {"lang": "en", "tld": "co.uk", "slow": False},
    "Australian Accent (Female)": {"lang": "en", "tld": "com.au", "slow": False},
    "Indian Accent (Bilingual/Hinglish)": {"lang": "hi", "tld": "co.in", "slow": False},
    "Slow Reading Voice (US Narrator)": {"lang": "en", "tld": "com", "slow": True}
}

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✨ AI StoryStudio Suite</div>
    <div class="hero-title">Intelligent <span>StoryTeller & Audio Studio</span></div>
    <div class="hero-subtitle">
        Generate vivid stories from images or write your own, personalize voice narration accents, and export clean PDFs and audiobooks.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Settings
with st.sidebar:
    st.markdown("### 🎙️ Audio Narration Settings")
    selected_voice_label = st.selectbox(
        "Voice & Accent",
        list(VOICE_OPTIONS.keys()),
        index=0,
        help="Select accent and speed for the generated speech."
    )
    selected_voice = VOICE_OPTIONS[selected_voice_label]

    st.markdown("---")
    st.markdown("### 📖 Generation Parameters")
    tone = st.selectbox(
        "Narrative Tone",
        ["Adventurous", "Whimsical", "Romantic", "Mysterious", "Humorous", "Calm", "Dramatic", "Hinglish Desi Tadka"],
        index=0
    )

    length_label = st.selectbox(
        "Target Word Count",
        ["Short (200-300 words)", "Medium (300-600 words)", "Long (600-1000 words)"],
        index=1
    )



# BLIP & Gemini Model Loaders
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

processor, blip_model = load_models()

genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel(model_name="models/gemini-2.0-flash")

gemini_model = load_gemini_model()

def get_captions(images):
    captions = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(blip_model.device)
        out = blip_model.generate(**inputs, max_new_tokens=40)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

def generate_story_from_captions(captions, tone, length_label):
    length_map = {
        "Short (200-300 words)": (200, 300, 800),
        "Medium (300-600 words)": (300, 600, 1200),
        "Long (600-1000 words)": (600, 1000, 1600)
    }
    min_words, max_words, max_tokens = length_map.get(length_label, (300, 600, 1200))

    prompt = (
        f"You are a gifted author. Write an imaginative, engaging {tone.lower()} story inspired by these image scenes:\n\n"
        + "\n".join([f"- Scene {i+1}: {cap}" for i, cap in enumerate(captions)])
        + f"\n\nMake the story emotionally rich with seamless narrative continuity. "
        + f"Target word count: between {min_words} and {max_words} words."
    )

    # Models to try in sequence if one hits limits
    models_to_try = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash"
]

for m in models_to_try:
        try:
            m_instance = genai.GenerativeModel(model_name=f"models/{m}")
            response = m_instance.generate_content(
                contents=prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.85,
                    top_p=0.95,
                    max_output_tokens=max_tokens
                )
            )
            return response.text.strip()
        except Exception:
            continue
    return "❌ All model options temporarily exceeded quota. Please check your Gemini API key."

def generate_story_from_prompt(user_prompt, tone, length_label):
    length_map = {
        "Short (200-300 words)": (200, 300, 800),
        "Medium (300-600 words)": (300, 600, 1200),
        "Long (600-1000 words)": (600, 1000, 1600)
    }
    min_words, max_words, max_tokens = length_map.get(length_label, (300, 600, 1200))

    full_prompt = (
        f"Write a {tone.lower()} story based on this idea: '{user_prompt}'.\n"
        f"Make it creative, vivid, and approximately {min_words}-{max_words} words long."
    )

    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.8-flash"]
    for m in models_to_try:
        try:
            m_instance = genai.GenerativeModel(model_name=f"models/{m}")
            response = m_instance.generate_content(
                contents=full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.85,
                    top_p=0.95,
                    max_output_tokens=max_tokens
                )
            )
            return response.text.strip()
        except Exception:
            continue
    return "❌ All model options temporarily exceeded quota. Please check your Gemini API key."

def create_pdf(story_text, title="StoryStudio AI — Generated Story", images=None):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    c.setFillColorRGB(0.1, 0.1, 0.15)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 50, title)

    c.setStrokeColorRGB(0.85, 0.88, 0.92)
    c.setLineWidth(1)
    c.line(50, h - 60, w - 50, h - 60)

    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.setFont("Helvetica", 11)

    text = textwrap.wrap(story_text, 85)
    y = h - 90
    for line in text:
        if y < 70:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 11)
            c.setFillColorRGB(0.2, 0.2, 0.2)
        c.drawString(50, y, line)
        y -= 16

    if images:
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.setFillColorRGB(0.1, 0.1, 0.15)
        c.drawString(50, h - 50, "Visual Scene Reference")
        x, y = 50, h - 170
        for img in images:
            img_copy = img.copy()
            img_copy.thumbnail((160, 160))
            c.drawImage(ImageReader(img_copy), x, y, width=img_copy.width, height=img_copy.height)
            x += 180
            if x > w - 180:
                x = 50
                y -= 180
    c.save()
    buffer.seek(0)
    return buffer

def create_audio_with_voice(text, voice_cfg):
    audio_bytes = io.BytesIO()
    tts = gTTS(
        text=text,
        lang=voice_cfg.get("lang", "en"),
        tld=voice_cfg.get("tld", "com"),
        slow=voice_cfg.get("slow", False)
    )
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Tabs for Complete Project Features
tab_images, tab_custom_story, tab_text_prompt = st.tabs([
    "🖼️ Multi-Image StoryTeller",
    "✍️ Paste / Write Your Own Story",
    "💡 Text Prompt to Story"
])

# ----------------- TAB 1: MULTI-IMAGE STORYTELLER -----------------
with tab_images:
    st.markdown("#### 1. Upload Visual Moments")
    uploaded_images = st.file_uploader(
        "Upload photos to be captioned and woven into a story",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="multi_img_uploader"
    )

    if uploaded_images:
        st.markdown("###### Image Gallery")
        cols = st.columns(min(len(uploaded_images), 4))
        for idx, img_file in enumerate(uploaded_images):
            with cols[idx % 4]:
                preview_img = Image.open(img_file)
                st.image(preview_img, use_container_width=True, caption=f"Scene {idx+1}")

    st.markdown("#### 2. Generate Multi-Scene Story")
    if not uploaded_images:
        st.info("👆 Upload at least one image to begin.")
    else:
        if st.button("🪄 Weave Story from Images", key="btn_img_story", use_container_width=True):
            pil_images = [Image.open(img) for img in uploaded_images]
            with st.status("Analyzing visual scenes with BLIP...", expanded=True) as status:
                captions = get_captions(pil_images)
                st.write("##### Scene Captions:")
                for i, cap in enumerate(captions):
                    st.write(f"• **Scene {i+1}**: *\"{cap}\"*")

                status.update(label="Synthesizing narrative with Gemini...", state="running")
                story = generate_story_from_captions(captions, tone, length_label)
                status.update(label="✨ Story Ready!", state="complete", expanded=False)

            st.session_state["image_story"] = story
            st.session_state["image_pil_list"] = pil_images

    if "image_story" in st.session_state:
        st.markdown("### 📖 Generated Story")
        st.markdown(f'<div class="story-output-box">{st.session_state["image_story"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### 📥 Export Story & Voiceover (*{selected_voice_label}*)")
        col1, col2 = st.columns(2)
        with col1:
            pdf_data = create_pdf(st.session_state["image_story"], "AI StoryTeller — Visual Narrative", st.session_state.get("image_pil_list"))
            st.download_button(
                "📄 Download Story PDF",
                data=pdf_data,
                file_name="image_story.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col2:
            with st.spinner(f"Generating audio with {selected_voice_label}..."):
                audio_data = create_audio_with_voice(st.session_state["image_story"], selected_voice)
                st.audio(audio_data, format="audio/mp3")
                st.download_button(
                    f"🔊 Download MP3 ({selected_voice_label.split(' ')[0]})",
                    data=audio_data,
                    file_name="story_audio.mp3",
                    mime="audio/mpeg",
                    use_container_width=True
                )

# ----------------- TAB 2: PASTE / WRITE YOUR OWN STORY -----------------
with tab_custom_story:
    st.markdown("#### ✍️ Custom Story & Audio Studio")
    st.markdown("Paste your own story, manuscript, or chapter to generate voiceover narration in any accent and download as PDF/MP3.")

    default_sample_story = """Wah, ek alien ki kahani, Hindustani tadke ke saath! Chalo sunte hain...

Title: Chai Aur Ek Tara
Himachal ke khoobsurat pahadon mein, Jahan Har subah suraj ki kirnein devdaar ke pedon ko sone sa chamkaati theen, wahaan Raju apni chhoti si chai ki dukaan chalaata tha. राजू की चाय पूरे इलाके में मशहूर थी।
एक सुबह उसने आसमान से गिरती हुई एक रहस्यमयी नीली रोशनी देखी..."""

    user_custom_text = st.text_area(
        "Enter or paste your story here:",
        value=default_sample_story,
        height=260,
        help="Paste any Hinglish, English, or Hindi story text."
    )

    custom_story_title = st.text_input("Story Title (for PDF export):", value="My Custom Story")

    col_btn, _ = st.columns([1, 1])
    with col_btn:
        generate_custom_audio = st.button("🎙️ Generate Audio & PDF for My Story", key="btn_custom_audio", use_container_width=True)

    if generate_custom_audio and user_custom_text.strip():
        st.session_state["custom_story_saved"] = user_custom_text
        st.session_state["custom_title"] = custom_story_title

    if "custom_story_saved" in st.session_state:
        st.markdown("---")
        st.markdown(f"### 🎧 Listen & Export (*Voice: {selected_voice_label}*)")
        c1, c2 = st.columns(2)
        with c1:
            pdf_custom = create_pdf(st.session_state["custom_story_saved"], st.session_state.get("custom_title", "Custom Story"))
            st.download_button(
                "📄 Download as PDF",
                data=pdf_custom,
                file_name="custom_story.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with c2:
            with st.spinner(f"Synthesizing speech with {selected_voice_label}..."):
                custom_audio = create_audio_with_voice(st.session_state["custom_story_saved"], selected_voice)
                st.audio(custom_audio, format="audio/mp3")
                st.download_button(
                    "🔊 Download Narration MP3",
                    data=custom_audio,
                    file_name="custom_story_narration.mp3",
                    mime="audio/mpeg",
                    use_container_width=True
                )

# ----------------- TAB 3: TEXT PROMPT TO STORY -----------------
with tab_text_prompt:
    st.markdown("#### 💡 Prompt-to-Story Generation")
    st.markdown("Describe an idea, character, or situation (like in Cell 4) to generate a complete story from scratch.")

    prompt_input = st.text_input(
        "Enter story prompt / idea:",
        placeholder="e.g., A girl named Megha pursuing engineering who misses her hometown Navratri vibes...",
        key="prompt_input_field"
    )

    if st.button("🚀 Generate Story from Idea", key="btn_prompt_story", use_container_width=True):
        if not prompt_input.strip():
            st.warning("Please enter a story idea first.")
        else:
            with st.spinner("Generating story with Gemini..."):
                prompt_story = generate_story_from_prompt(prompt_input, tone, length_label)
                st.session_state["prompt_story"] = prompt_story
                st.session_state["prompt_idea"] = prompt_input

    if "prompt_story" in st.session_state:
        st.markdown("### 📖 Generated Story")
        st.markdown(f'<div class="story-output-box">{st.session_state["prompt_story"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### 📥 Export Story & Audio (*{selected_voice_label}*)")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pdf_p = create_pdf(st.session_state["prompt_story"], f"Story: {st.session_state.get('prompt_idea', 'AI Story')[:35]}...")
            st.download_button(
                "📄 Download Story PDF",
                data=pdf_p,
                file_name="prompt_story.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col_p2:
            with st.spinner(f"Synthesizing audio with {selected_voice_label}..."):
                audio_p = create_audio_with_voice(st.session_state["prompt_story"], selected_voice)
                st.audio(audio_p, format="audio/mp3")
                st.download_button(
                    "🔊 Download MP3 Audio",
                    data=audio_p,
                    file_name="prompt_story.mp3",
                    mime="audio/mpeg",
                    use_container_width=True
                )
