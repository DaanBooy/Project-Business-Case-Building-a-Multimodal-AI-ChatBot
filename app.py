import os
import re
import uuid
from pathlib import Path
import gradio as gr
from gradio.themes.utils import colors

# agent wrapper
from agent import run_agent  # def run_agent(question: str, session_id: str) -> str

WELCOME_MSG = (
    "Welcome to 🌱 **GrowGuide** 🌱\n\n"
    "Your assistant for growing fruits and vegetables at home. "
    "Ask me about seeding, soil conditions, harvesting, or anything that comes to mind about growing your own crops.\n\n"
    "Want more personalized advice? Tell me what city you live in and I’ll find your USDA hardiness climate zone and adjust my advice accordingly.\n\n"
    "Don’t know how to start, or when you should do certain things? I can also create a planning for you! "
    "Simply request a PDF planning and I’ll create an easy, personalized plan for you to follow."
)
# --- OpenAI STT/TTS helpers ---------------------------------------------------
# Uses the official OpenAI SDK (v1.x). Pin in requirements: openai>=1.40.0
try:
    from openai import OpenAI
except Exception as _e:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_AUDIO = bool(OPENAI_API_KEY and OpenAI)  # only enable STT/TTS if key + sdk available

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")            # or "gpt-4o-mini-transcribe"
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")      # or "tts-1"
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "echo")                 # other options are alloy, verse, aria, etc.

client = OpenAI(api_key=OPENAI_API_KEY) if USE_AUDIO else None
TMP_DIR = Path("/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def transcribe_audio(audio_path: str) -> str:
    """Speech -> text with OpenAI. Returns empty string on failure."""
    if not USE_AUDIO or not audio_path:
        return ""
    try:
        with open(audio_path, "rb") as f:
            # New SDK notation:
            # Prefer gpt-4o-mini-transcribe, else whisper-1
            resp = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f
            )
        # whisper-1 returns .text; gpt-4o-mini-transcribe returns .text as well
        return (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        print("STT error:", e)
        return ""


def synthesize_speech(text: str) -> str:
    """Text -> mp3 with OpenAI TTS. Returns a file path or '' on failure."""
    if not USE_AUDIO or not text.strip():
        return ""
    try:
        out_path = TMP_DIR / f"reply_{uuid.uuid4().hex[:8]}.mp3"
        # Stream to file (new SDK)
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text
        ) as resp:
            resp.stream_to_file(out_path)
        return str(out_path)
    except Exception as e:
        print("TTS error:", e)
        return ""


# --- UI logic -----------------------------------------------------------------
PDF_PATH_RE = re.compile(r"(/tmp/[^\s]+\.pdf)")

def ensure_session_id(sess: str | None) -> str:
    return sess or uuid.uuid4().hex

def do_send(user_text: str,
            audio_path: str | None,
            history: list[tuple[str, str]],
            session_id: str | None):
    """
    - If audio provided: transcribe -> user_text
    - Call your agent
    - Append to history
    - If the reply includes a PDF path, surface it as a downloadable file
    """
    session_id = ensure_session_id(session_id)

    # Prefer audio if present
    if audio_path:
        stt_text = transcribe_audio(audio_path)
        user = stt_text or user_text
    else:
        user = (user_text or "").strip()

    if not user:
        return gr.update(), gr.update(), session_id, None, gr.update(value=None)  # no-op

    bot = run_agent(user, session_id)

    # Detect a generated PDF path and expose as a file
    pdf_file = None
    m = PDF_PATH_RE.search(bot or "")
    if m:
        pdf_candidate = m.group(1)
        if Path(pdf_candidate).exists():
            pdf_file = pdf_candidate

    new_hist = history + [(user, bot)]
    return new_hist, "", session_id, pdf_file, gr.update(value=None)  # update chat, clear textbox, keep session, set file (or None)


def do_tts(history: list[tuple[str, str]]):
    """Speak the last bot message."""
    if not USE_AUDIO or not history:
        return None
    last_bot = history[-1][1] if history[-1] and len(history[-1]) > 1 else ""
    if not last_bot.strip():
        return None
    audio_fp = synthesize_speech(last_bot)
    return audio_fp or None


def do_clear():
    """Reset the chat but keep a friendly welcome message."""
    new_session = uuid.uuid4().hex
    return [(None, WELCOME_MSG)], new_session, None

# --- Theme + CSS --------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue=colors.emerald,      # green accents to fit theme
    secondary_hue=colors.emerald,
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)

custom_css = """
footer { visibility: hidden; }
#speech_reply_box { background: #f0fdf4; border-radius: 8px; }  /* soft green bg */
"""

# --- Gradio UI ----------------------------------------------------------------
with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.Markdown("# 🌱 GrowGuide — Your assistant for growing fruits and vegetables at home! 🌱 ")

    with gr.Row():
        chat = gr.Chatbot(height=460, label="Chat", type="tuples", value=[(None, WELCOME_MSG)])
        with gr.Column(scale=1):
            pdf_out = gr.File(label="Generated PDFs", interactive=False)

            # audio preview for TTS playback (green waveform/progress)
            tts_player = gr.Audio(
                label="Speech reply",
                interactive=False,
                elem_id="speech_reply_box",
            )

    with gr.Row():
        txt = gr.Textbox(placeholder="Type your question… (or use the microphone)", scale=3)
        mic = gr.Audio(sources=["microphone"], type="filepath", label="Mic", scale=1)

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        speak_btn = gr.Button("🔊 Speak last reply")
        clear_btn = gr.Button("Clear chat")

    # state
    state_history = gr.State(value=[(None, WELCOME_MSG)])  
    state_session = gr.State(value=uuid.uuid4().hex)

    # events
    send_btn.click(
        fn=do_send,
        inputs=[txt, mic, state_history, state_session],
        outputs=[chat, txt, state_session, pdf_out, mic],
        queue=True,
    )
    txt.submit(
        fn=do_send,
        inputs=[txt, mic, state_history, state_session],
        outputs=[chat, txt, state_session, pdf_out, mic],
        queue=True,
    )

    speak_btn.click(
        fn=do_tts,
        inputs=[state_history],
        outputs=[tts_player],
        queue=True,
    )

    clear_btn.click(
        fn=do_clear,
        inputs=[],
        outputs=[chat, state_session, pdf_out],
    )

    # Keep chat history in sync after each send
    chat.change(lambda h: h, inputs=[chat], outputs=[state_history])

demo.launch()
