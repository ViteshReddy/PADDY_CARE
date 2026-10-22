
import json
import re
import subprocess
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np



# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PaddyCare | Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "paddy_disease_model.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"
PARSER_EXE = ROOT / "symptom_parser.exe"
KNOWLEDGE_BASE_PATH = ROOT / "disease_info.json"

try:
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        DISEASE_INFO = json.load(file)
except (FileNotFoundError, json.JSONDecodeError):
    DISEASE_INFO = {}


# ============================================================
# 2. ADVANCED UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --green: #16845b;
        --green-dark: #075c40;
        --ink: #17352a;
        --muted: #708078;
        --line: #e3ebe5;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 5% 0%,
                rgba(42, 157, 105, 0.07),
                transparent 34%
            ),
            #f7faf7;
    }

    header[data-testid="stHeader"] {
        background: rgba(247, 250, 247, 0.88);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 8rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink);
        letter-spacing: -0.7px;
    }

    /* SIDEBAR: READABLE TEXT */

    [data-testid="stSidebar"] {
        background: #eef5ef !important;
        border-right: 1px solid #dce7dd;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #244735 !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #627568 !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }

    [data-testid="stSidebar"] .stButton > button {
        color: #ffffff !important;
        background: #176b49 !important;
        border: 1px solid #176b49 !important;
        font-weight: 600 !important;
        border-radius: 12px;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #0d5639 !important;
        border-color: #0d5639 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #d1dfd4 !important;
    }

    /* SIDEBAR BRAND */

    .brand-mark {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 28px;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        flex-shrink: 0;
        border-radius: 16px;
        background: linear-gradient(135deg, #16845b, #075c40);
        color: white !important;
        font-size: 25px;
        box-shadow: 0 6px 18px rgba(7, 92, 64, 0.19);
    }

    .brand-name {
        font: 800 22px 'Manrope', sans-serif;
        color: #123b2c !important;
        line-height: 1.15;
    }

    .brand-sub {
        color: #627568 !important;
        font-size: 11px;
        margin-top: 4px;
    }

    /* HERO BANNER */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 34px 36px;
        margin-bottom: 25px;
        border-radius: 26px;
        color: #ffffff;
        background: linear-gradient(
            120deg,
            #073e2d 0%,
            #096444 56%,
            #16845b 100%
        );
        box-shadow: 0 14px 36px rgba(7, 92, 64, 0.14);
    }

    .hero:after {
        content: "🌾";
        position: absolute;
        right: 35px;
        top: 14px;
        font-size: 95px;
        opacity: 0.14;
        transform: rotate(-12deg);
        pointer-events: none;
    }

    .hero-eyebrow {
        display: inline-block;
        color: #e0ffeb !important;
        border: 1px solid rgba(219, 255, 231, 0.28);
        background: rgba(255, 255, 255, 0.08);
        border-radius: 50px;
        padding: 6px 11px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        color: #ffffff !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: clamp(27px, 4vw, 39px);
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -1.2px;
        margin-top: 18px;
        max-width: 620px;
    }

    .hero-description {
        position: relative;
        z-index: 1;
        color: #e1f2e7 !important;
        font-size: 15px;
        line-height: 1.7;
        max-width: 590px;
        margin-top: 12px;
    }

    /* WELCOME CARDS */

    .feature-card {
        border: 1px solid #e0eae2;
        background: rgba(255, 255, 255, 0.94);
        border-radius: 18px;
        padding: 19px;
        min-height: 130px;
        box-shadow: 0 4px 15px rgba(29, 67, 44, 0.025);
    }

    .feature-icon {
        font-size: 23px;
        margin-bottom: 10px;
    }

    .feature-title {
        font: 700 14px 'Manrope', sans-serif;
        color: #183d2d !important;
        margin-bottom: 5px;
    }

    .feature-description {
        color: #65786b !important;
        font-size: 12px;
        line-height: 1.6;
    }

    .chat-heading {
        font: 800 23px 'Manrope', sans-serif;
        color: #173b2b !important;
        letter-spacing: -0.5px;
    }

    .chat-subheading {
        color: #66796d !important;
        font-size: 13px;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    /* CHAT MESSAGES */

    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #e1eae3;
        border-radius: 19px;
        padding: 17px 19px;
        box-shadow: 0 4px 16px rgba(23, 53, 42, 0.025);
        margin-bottom: 13px;
    }

    div[data-testid="stChatInput"] {
        border: 1px solid #d1e1d5;
        border-radius: 20px;
        background: #ffffff;
        box-shadow: 0 9px 30px rgba(23, 53, 42, 0.08);
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #16845b;
        box-shadow: 0 0 0 3px rgba(22, 132, 91, 0.09);
    }

    .stButton > button {
        border-radius: 12px;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* HIGH-CONTRAST TEXT: dark text on light app surfaces */
    .stApp,
    .stApp p,
    .stApp li,
    .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] span,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stExpander"] summary span,
    .stApp [data-testid="stChatMessage"] p,
    .stApp [data-testid="stChatMessage"] span,
    .stApp [data-testid="stChatMessage"] label,
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {
        color: #172b22 !important;
    }

    /* Alert text should remain readable on tinted alert backgrounds */
    div[data-testid="stAlert"],
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        color: #172b22 !important;
    }

    /* Preserve the intentional light text in the dark green hero */
    .hero { color: #ffffff !important; }
    .hero .hero-eyebrow { color: #e0ffeb !important; }
    .hero .hero-title { color: #ffffff !important; }
    .hero .hero-description { color: #e1f2e7 !important; }

    div[data-testid="stExpander"] {
        border: 1px solid #e1eae3;
        border-radius: 14px;
        background: #ffffff;
    }

    .footer-note {
        text-align: center;
        color: #718177 !important;
        font-size: 11px;
        padding-top: 22px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 7rem;
        }

        .hero {
            padding: 25px 22px;
            border-radius: 20px;
        }

        .hero:after {
            right: 8px;
            font-size: 60px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SYMPTOM ANALYSIS
# ============================================================

def baseline_symptom_analysis(text):
    text = text.lower()

    patterns = {
        "brown_spots": r"\bbrown(?:\s+|-)spots?\b|\bbrown\s+lesions?\b",
        "yellowing": r"\byellow(?:ing|ed)?\b",
        "drying": r"\bdry(?:ing)?\b|\bdried\b",
        "blight": r"\bblight\b",
        "rust": r"\brust\b",
        "wilting": r"\bwilt(?:ing|ed)?\b",
        "lesions": r"\blesions?\b",
        "mildew": r"\bmildew\b",
    }

    return {
        key: bool(re.search(pattern, text))
        for key, pattern in patterns.items()
    }


def parse_with_lex_yacc(text):
    if not PARSER_EXE.exists():
        return (
            baseline_symptom_analysis(text),
            "Fallback symptom recognition used; parser executable not found.",
        )

    try:
        result = subprocess.run(
            [str(PARSER_EXE)],
            input=text,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            cwd=str(ROOT),
        )

        if result.returncode != 0:
            return (
                baseline_symptom_analysis(text),
                "Fallback symptom recognition used because the parser returned an error.",
            )

        data = {}

        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip() == "1"

        if not data:
            return (
                baseline_symptom_analysis(text),
                "Fallback recognition used.",
            )

        return data, "Processed by the Lex/Yacc symptom parser."

    except (OSError, subprocess.TimeoutExpired):
        return (
            baseline_symptom_analysis(text),
            "Fallback symptom recognition used.",
        )


# ============================================================
# 4. DISEASE KNOWLEDGE BASE
# ============================================================

def show_disease_information(disease_key):
    info = DISEASE_INFO.get(disease_key)

    if not info:
        st.info("Disease information is not available for this entry yet.")
        return

    st.markdown(f"### 🌿 {info.get('display_name', disease_key)}")

    with st.expander("Symptoms, causes and management", expanded=True):
        st.markdown("**Typical symptoms**")

        for item in info.get("symptoms", []):
            st.write(f"• {item}")

        st.markdown("**Cause**")
        st.write(info.get("cause", "Information unavailable."))

        st.markdown("**Management suggestions**")

        for item in info.get("management", []):
            st.write(f"• {item}")

        st.caption(
            "Educational guidance only. Consult a qualified agricultural "
            "expert before selecting treatment."
        )


# ============================================================
# 5. IMAGE MODEL
# ============================================================

@st.cache_resource
def load_image_model():
    """Load the trained Keras model once and reuse it."""
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        return None, None, (
            "Model files were not found. Check that both files are inside "
            "the project's models folder."
        )
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            class_names = json.load(file)
        if not isinstance(class_names, list) or not class_names:
            return None, None, "class_names.json is empty or invalid."
        return model, class_names, None
    except Exception as exc:
        return None, None, f"The image model could not be loaded: {exc}"


def predict_paddy_image(image_bytes):
    """Predict a class for a paddy image and return its top three scores."""
    model, class_names, error = load_image_model()
    if error:
        return None, error
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize((224, 224))
            image_array = np.asarray(image, dtype=np.float32)
        predictions = model.predict(
            np.expand_dims(image_array, axis=0), verbose=0
        )[0]
        if len(predictions) != len(class_names):
            return None, (
                "The model output count does not match class_names.json. "
                "Check that both files came from the same training run."
            )
        best_index = int(np.argmax(predictions))
        ranking = sorted(
            zip(class_names, [float(value) for value in predictions]),
            key=lambda item: item[1],
            reverse=True,
        )
        return {
            "class_name": class_names[best_index],
            "confidence": float(predictions[best_index]),
            "top_predictions": ranking[:3],
        }, None
    except Exception as exc:
        return None, f"Image prediction failed: {exc}"


# ============================================================
# 6. CONVERSATION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">
            <div class="brand-icon">🌾</div>
            <div>
                <div class="brand-name">PaddyCare</div>
                <div class="brand-sub">CROP INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Your workspace")

    st.caption(
        "Analyze paddy leaf images and symptoms in one conversation."
    )

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("**HOW IT WORKS**")
    st.markdown("🖼️ Leaf image analysis")
    st.markdown("💬 Symptom recognition")
    st.markdown("🌱 Disease information")

    st.divider()

    st.markdown("**CURRENT CAPABILITIES**")
    st.caption("🟢 Chat interface")
    st.caption("🟢 Lex/Yacc symptom parser")
    st.caption("🟢 Image model integration enabled")
    st.caption("🟡 Predictions are preliminary; validate before use")

    st.divider()

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("PaddyCare · Academic project prototype")


# ============================================================
# 8. HERO SECTION — CORRECT HTML RENDERING
# ============================================================

st.markdown(
    """
    <div class="hero">
        <span class="hero-eyebrow">
            SMART AGRICULTURE · AI ASSISTANT
        </span>
        <div class="hero-title">
            Understand your crop.<br>
            Protect your harvest.
        </div>
        <div class="hero-description">
            Share a paddy leaf photo, describe what you observe,
            or send both together. PaddyCare helps organize symptoms
            and is being developed to support disease identification.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. WELCOME SCREEN
# ============================================================

if not st.session_state.messages:
    st.markdown(
        '<div class="chat-heading">'
        'Let’s look at your paddy crop</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="chat-subheading">'
        'Start with an image, a symptom, or a question.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📷</div>
                <div class="feature-title">Leaf inspection</div>
                <div class="feature-description">
                    Attach a photo of a paddy leaf for image analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">Describe symptoms</div>
                <div class="feature-description">
                    Explain spots, yellowing, drying or other signs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌱</div>
                <div class="feature-title">Explore guidance</div>
                <div class="feature-description">
                    Review disease information as the system develops.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 10. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("text"):
            st.markdown(message["text"])

        for image_bytes in message.get("images", []):
            st.image(image_bytes, use_container_width=True)

        if message.get("features") is not None:
            with st.expander("Recognized symptom features"):
                st.json(message["features"])

        if message.get("notice"):
            st.info(message["notice"])

        if message.get("warning"):
            st.caption(message["warning"])


# ============================================================
# 11. CHAT INPUT: TEXT, IMAGE, OR BOTH
# ============================================================

chat_input = st.chat_input(
    "Message PaddyCare...",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png"],
)

if chat_input is not None:
    user_text = (chat_input.text or "").strip()
    uploaded_files = list(chat_input.files or [])

    valid_images = []
    invalid_images = []

    for uploaded_file in uploaded_files:
        try:
            raw = uploaded_file.getvalue()

            with Image.open(BytesIO(raw)) as image:
                image.verify()

            with Image.open(BytesIO(raw)) as image:
                converted = image.convert("RGB")
                output = BytesIO()
                converted.save(output, format="JPEG")

            valid_images.append(output.getvalue())

        except (UnidentifiedImageError, OSError, ValueError):
            invalid_images.append(uploaded_file.name)

    if not user_text and not valid_images:
        st.warning("Type a message, attach a leaf image, or do both.")

    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "text": user_text,
                "images": valid_images,
            }
        )

        with st.chat_message("user"):
            if user_text:
                st.markdown(user_text)

            for image_bytes in valid_images:
                st.image(image_bytes, use_container_width=True)

        response_parts = []
        features = None
        notice = None

        warning = (
            "This is a preliminary academic prototype, not a validated diagnosis. "
            "Do not use its output alone to select pesticides or treatment."
        )

        with st.chat_message("assistant"):
            if invalid_images:
                st.warning(
                    "Could not read these image files: "
                    + ", ".join(invalid_images)
                )

            if user_text:
                features, parser_message = parse_with_lex_yacc(user_text)

                st.markdown("**Symptom analysis**")
                st.caption(parser_message)

                with st.expander("View recognized symptom features"):
                    st.json(features)

                recognized = [
                    key.replace("_", " ")
                    for key, value in features.items()
                    if value
                ]

                if recognized:
                    response_parts.append(
                        "I recognized these symptom features: "
                        + ", ".join(recognized) + "."
                    )
                else:
                    response_parts.append(
                        "I couldn't recognize supported symptom keywords. "
                        "Try mentioning brown spots, yellowing, drying, blight, "
                        "rust, wilting, lesions or mildew."
                    )

            if valid_images:
                st.markdown("**Image analysis**")
                image_results = []

                for image_index, image_bytes in enumerate(valid_images, start=1):
                    prediction, prediction_error = predict_paddy_image(image_bytes)

                    if prediction_error:
                        st.warning(prediction_error)
                        image_results.append(
                            f"Image {image_index}: prediction unavailable."
                        )
                        continue

                    predicted_key = prediction["class_name"]
                    display_name = predicted_key.replace("_", " ").title()
                    confidence_percent = prediction["confidence"] * 100

                    st.markdown(f"#### Image {image_index}: {display_name}")
                    st.metric("Model confidence", f"{confidence_percent:.1f}%")
                    st.caption(
                        "Preliminary model output only. Confidence is not the "
                        "same as diagnostic accuracy."
                    )

                    with st.expander("View top model predictions"):
                        for class_name, score in prediction["top_predictions"]:
                            st.write(
                                f"**{class_name.replace('_', ' ').title()}** — "
                                f"{score * 100:.1f}%"
                            )

                    if predicted_key in DISEASE_INFO:
                        show_disease_information(predicted_key)
                    elif display_name in DISEASE_INFO:
                        show_disease_information(display_name)

                    image_results.append(
                        f"Image {image_index}: the model's top prediction is "
                        f"{display_name} ({confidence_percent:.1f}% confidence)."
                    )

                notice = (
                    "This is an academic prototype trained on a small sample. "
                    "Its predictions may be wrong; do not use them alone to "
                    "choose pesticides or treatment."
                )
                st.info(notice)
                response_parts.extend(image_results)

            response_text = "\n\n".join(response_parts)

            st.markdown(response_text)
            st.caption(warning)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": response_text,
                "images": [],
                "features": features,
                "notice": notice,
                "warning": warning,
            }
        )


# ============================================================
# 12. FOOTER
# ============================================================

st.markdown(
    '<div class="footer-note">'
    'PaddyCare · AI-assisted crop health research · '
    'Prototype under development'
    '</div>',
    unsafe_allow_html=True,
)
