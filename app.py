import io
import logging

import streamlit as st
from dotenv import load_dotenv

from llm_pipeline import build_image_prompt, generate_image, transcribe_audio


# Load environment variables from a .env file if present
load_dotenv()


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("voice_to_image_app")


# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Voice to Image Agent",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS for modern styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Card-like containers */
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Step indicators */
    .step-indicator {
        display: flex;
        align-items: center;
        margin: 1.5rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-pending {
        background-color: #fff3cd;
        color: #856404;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize Streamlit session state keys used across the app."""
    defaults = {
        "transcript": "",
        "image_prompt": "",
        "image_bytes": None,
        "error_message": "",
        "openai_api_key": "",
        "models_used": {
            "transcription_model": "whisper-1",
            "llm_model": "gpt-5-nano",
            "image_model": "gpt-image-1",
        },
        "image_size": "1024x1024",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar() -> None:
    """Render the configuration sidebar with modern design."""
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='margin: 0; color: #667eea;'>⚙️ Settings</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "Configure the models and parameters for the **Voice → Image** pipeline."
    )

    st.sidebar.markdown("---")

    # API Key section
    st.sidebar.markdown("### 🔑 API Configuration")
    st.session_state.openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        help="Enter your OpenAI API key. If left empty, will use OPENAI_API_KEY from environment variables.",
        key="api_key_input",
    )
    
    if not st.session_state.openai_api_key:
        st.sidebar.warning("⚠️ API key not set. Using environment variable if available.")

    st.sidebar.markdown("---")

    # Models section with icons
    st.sidebar.markdown("### 🎯 Model Configuration")
    
    with st.sidebar.container():
        st.session_state.models_used["transcription_model"] = st.text_input(
            "🎤 Transcription Model",
            value=st.session_state.models_used["transcription_model"],
            help="OpenAI audio transcription model (e.g., `whisper-1`).",
            key="transcription_input",
        )
        
        st.session_state.models_used["llm_model"] = st.text_input(
            "🤖 LLM Model",
            value=st.session_state.models_used["llm_model"],
            help="OpenAI chat model for building the image description.",
            key="llm_input",
        )
        
        st.session_state.models_used["image_model"] = st.text_input(
            "🎨 Image Model",
            value=st.session_state.models_used["image_model"],
            help="OpenAI image generation model (e.g., `gpt-image-1`).",
            key="image_input",
        )

    st.sidebar.markdown("---")
    
    # Image options
    st.sidebar.markdown("### 🖼️ Image Options")
    
    size_index = (
        ["512x512", "768x768", "1024x1024"].index(st.session_state.image_size)
        if st.session_state.image_size in ["512x512", "768x768", "1024x1024"]
        else 2
    )
    
    st.session_state.image_size = st.sidebar.selectbox(
        "Image Resolution",
        options=["512x512", "768x768", "1024x1024"],
        index=size_index,
        help="Higher resolution = better quality but slower generation.",
    )

    st.sidebar.markdown("---")
    
    # Info section
    st.sidebar.markdown("### ℹ️ Information")
    st.sidebar.info(
        "💡 **Tip**: Logs are printed to the terminal where you run `streamlit run app.py`."
    )
    
    # Pipeline status
    if st.session_state.transcript or st.session_state.image_bytes:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Pipeline Status")
        
        status_transcript = "✅" if st.session_state.transcript else "⏳"
        status_prompt = "✅" if st.session_state.image_prompt else "⏳"
        status_image = "✅" if st.session_state.image_bytes else "⏳"
        
        st.sidebar.markdown(f"{status_transcript} Transcription")
        st.sidebar.markdown(f"{status_prompt} Prompt Generation")
        st.sidebar.markdown(f"{status_image} Image Generation")


def run_pipeline(uploaded_audio) -> None:
    """
    Run the full voice → transcript → image prompt → image pipeline.

    Side effects: updates Streamlit session_state keys.
    """
    st.session_state.error_message = ""
    st.session_state.transcript = ""
    st.session_state.image_prompt = ""
    st.session_state.image_bytes = None

    if uploaded_audio is None:
        st.session_state.error_message = "Please upload an audio file first."
        return

    # Convert the uploaded file into a file-like object compatible with OpenAI
    audio_bytes = uploaded_audio.read()
    audio_buffer = io.BytesIO(audio_bytes)
    # OpenAI expects a name with an extension
    audio_buffer.name = uploaded_audio.name or "voice_message.wav"

    transcription_model = st.session_state.models_used["transcription_model"]
    llm_model = st.session_state.models_used["llm_model"]
    image_model = st.session_state.models_used["image_model"]

    # Get API key from session state (empty string will fall back to env var)
    api_key = st.session_state.openai_api_key if st.session_state.openai_api_key else None

    try:
        with st.spinner("Transcribing audio with Whisper..."):
            logger.info("Step 1/3: Transcribing audio.")
            transcript = transcribe_audio(
                audio_buffer,
                model=transcription_model,
                api_key=api_key,
            )
            st.session_state.transcript = transcript

        with st.spinner("Building image prompt with LLM..."):
            logger.info("Step 2/3: Building image prompt from transcript.")
            prompt = build_image_prompt(
                transcript,
                model=llm_model,
                api_key=api_key,
            )
            st.session_state.image_prompt = prompt

        with st.spinner("Generating image from prompt..."):
            logger.info("Step 3/3: Generating image from prompt.")
            image_bytes, metadata = generate_image(
                prompt,
                model=image_model,
                size=st.session_state.image_size,
                api_key=api_key,
            )
            st.session_state.image_bytes = image_bytes

            # Update models_used with the final image metadata as well
            st.session_state.models_used["image_model"] = metadata.get(
                "model", image_model
            )

        logger.info("Pipeline finished successfully.")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        st.session_state.error_message = str(exc)


def render_pipeline_steps() -> None:
    """Render visual pipeline step indicators."""
    steps = [
        ("🎤", "Transcription", st.session_state.transcript),
        ("✍️", "Prompt Building", st.session_state.image_prompt),
        ("🎨", "Image Generation", st.session_state.image_bytes),
    ]
    
    cols = st.columns(3)
    for idx, (icon, label, status) in enumerate(steps):
        with cols[idx]:
            if status:
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 1rem; 
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 10px; color: white;'>
                        <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icon}</div>
                        <div style='font-weight: 600;'>{label}</div>
                        <div style='font-size: 0.8rem; margin-top: 0.5rem;'>✅ Complete</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 1rem; 
                                background: #f0f0f0; border-radius: 10px;'>
                        <div style='font-size: 2rem; margin-bottom: 0.5rem; opacity: 0.5;'>{icon}</div>
                        <div style='font-weight: 600; opacity: 0.7;'>{label}</div>
                        <div style='font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.5;'>⏳ Pending</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def main() -> None:
    """Main entry point for the Streamlit app."""
    init_session_state()
    render_sidebar()

    # Header section
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>🎙️ Voice to Image Agent</h1>
            <p style='font-size: 1.1rem; color: #666; margin-top: -1rem;'>
                Transform your voice into stunning AI-generated images
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pipeline steps visualization
    render_pipeline_steps()
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Main content area
    tab1, tab2 = st.tabs(["🎤 Upload & Generate", "📊 Results & Details"])

    with tab1:
        st.markdown("### Upload Your Voice Message")
        st.markdown(
            "Upload a short audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm`). "
            "The agent will transcribe it and transform it into a beautiful image."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Audio upload section
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "ogg", "webm"],
            accept_multiple_files=False,
            help="Supported formats: WAV, MP3, M4A, OGG, WebM",
        )

        if uploaded_audio is not None:
            st.markdown("**Audio Preview:**")
            st.audio(uploaded_audio, format="audio/wav")
            
            # File info
            file_size_mb = len(uploaded_audio.getvalue()) / (1024 * 1024)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("File Name", uploaded_audio.name)
            with col2:
                st.metric("File Size", f"{file_size_mb:.2f} MB")

        st.markdown("<br>", unsafe_allow_html=True)

        # Run button
        run_button_disabled = uploaded_audio is None
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button(
                "🚀 Run Voice → Image Pipeline",
                type="primary",
                disabled=run_button_disabled,
                use_container_width=True,
            ):
                run_pipeline(uploaded_audio)

        if st.session_state.error_message:
            st.error(f"❌ **Error**: {st.session_state.error_message}")

    with tab2:
        # Results section with better formatting
        if not st.session_state.transcript and not st.session_state.image_bytes:
            st.info("👆 Upload an audio file and run the pipeline to see results here.")
        else:
            # Transcript section
            st.markdown("### 📝 Transcribed Text")
            if st.session_state.transcript:
                st.markdown(
                    f"""
                    <div class='result-card'>
                        <p style='font-size: 1.1rem; line-height: 1.6; color: #333;'>
                            {st.session_state.transcript}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("⏳ Transcript will appear here after transcription.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Prompt section
            st.markdown("### ✍️ Enhanced Image Prompt")
            if st.session_state.image_prompt:
                st.markdown(
                    f"""
                    <div class='result-card'>
                        <p style='font-size: 1rem; line-height: 1.6; color: #555; font-style: italic;'>
                            "{st.session_state.image_prompt}"
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("⏳ The LLM-generated prompt will appear here after transcription.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Image section
            st.markdown("### 🎨 Generated Image")
            if st.session_state.image_bytes:
                col_img1, col_img2, col_img3 = st.columns([1, 3, 1])
                with col_img2:
                    st.image(
                        st.session_state.image_bytes,
                        caption="✨ Generated by the image model",
                        use_container_width=True,
                    )
                
                # Image metadata
                st.markdown("<br>", unsafe_allow_html=True)
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.metric("Model", st.session_state.models_used.get("image_model", "N/A"))
                with col_meta2:
                    st.metric("Size", st.session_state.image_size)
                with col_meta3:
                    img_size_kb = len(st.session_state.image_bytes) / 1024
                    st.metric("File Size", f"{img_size_kb:.1f} KB")
            else:
                st.info("⏳ The generated image will appear here once ready.")

            st.markdown("---")
            
            # Models used section
            st.markdown("### ⚙️ Models Used")
            st.json(st.session_state.models_used)


if __name__ == "__main__":
    main()


