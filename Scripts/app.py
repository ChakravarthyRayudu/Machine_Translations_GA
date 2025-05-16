
import streamlit as st
import os
from dotenv import load_dotenv

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Playshifu - Language Translator", page_icon="🌎")

from streamlit_google_auth import Authenticate

# Load environment variables from .env file
load_dotenv()


# Initialize Google Authenticator
authenticator = Authenticate(
    secret_credentials_path='google_credentials_temp.json',  # path to your credentials
    cookie_name='my_cookie_name',                       # any name for your cookie
    cookie_key='this_is_secret',                        # a random secret key
    # redirect_uri='http://localhost:8501',               # your Streamlit app URL
    redirect_uri='http://43.204.227.170.nip.io:8501',               # your Streamlit app URL

)
 
try:
    authenticator.check_authentification()
except Exception as e:
    st.error("Authentication failed. Please try again.")
    st.stop()

# If not authenticated, show login button
if not st.session_state.get('connected', False):
    st.info("Please log in with Google to use the app.")
    authorization_url = authenticator.get_authorization_url()
    st.link_button('Login with Google', authorization_url)
    st.stop()

# If authenticated, show user info and logout button
user_info = st.session_state['user_info']
st.image(user_info.get('picture'), width=60)
st.success(f"Hello, {user_info.get('name')} ({user_info.get('email')})")

if st.button('Log out'):
    authenticator.logout()
    st.stop()


##############################################################################################################


import streamlit as st
import pandas as pd
from utils import load_language_map, count_translatable_characters
from translate import perform_translation
from fetch_credits import get_deepl_usage
from config import DEEPL_API_KEY, GOOGLE_API_KEY
from streamlit.components.v1 import html
import base64
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher

# --- Semantic Similarity Model (CPU only) ---
@st.cache_resource
def get_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def calculate_semantic_similarity_batch(originals, translated):
    model = get_semantic_model()
    emb_orig = model.encode(originals, convert_to_tensor=True, show_progress_bar=False)
    emb_trans = model.encode(translated, convert_to_tensor=True, show_progress_bar=False)
    return util.pytorch_cos_sim(emb_orig, emb_trans).diagonal().cpu().numpy()

def calculate_text_similarity_batch(originals, translated):
    return [SequenceMatcher(None, o, t).ratio() for o, t in zip(originals, translated)]

st.title("🌎 Playshifu - Language Translator")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; } b
    [data-testid="stMetricLabel"] { font-size: 1rem !important; }
    .css-1d391kg { width: 200px; }
    .stDataFrame div[data-testid="stHorizontalBlock"] {
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)

deepl_lang_map = load_language_map('deepl_languages.json')
google_lang_map = load_language_map('google_languages.json')

# --- DeepL Usage Pie Chart ---
used, limit, remaining = get_deepl_usage(DEEPL_API_KEY)
st.sidebar.header("💳 DeepL Usage")
if used is not None:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    colors = ['#6c5ce7', '#00b894']
    explode = (0.05, 0)
    wedges, texts, autotexts = ax.pie(
        [used, remaining],
        labels=['Used', 'Remaining'],
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100 * limit):,})',
        startangle=90,
        explode=explode,
        shadow=False,
        textprops={'fontsize': 12, 'color': '#2d3436', 'weight': 'bold'},
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white', 'width': 0.4},
        pctdistance=0.8,
        labeldistance=1.15
    )
    centre_circle = plt.Circle((0, 0), 0.65, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.set_title('Credit Utilization', fontsize=15, fontweight='bold', color='#2d3436', pad=20)
    ax.text(0, 0, f'Total\n{limit:,}', ha='center', va='center', fontsize=11, color='#636e72', fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    st.sidebar.pyplot(fig, use_container_width=True)
else:
    st.sidebar.warning("Could not fetch DeepL usage stats.")

# --- File uploader and preview ---
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"], key="csv_uploader")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("🔎 Preview of your uploaded file:")
    st.dataframe(df, use_container_width=True, height=500)

    # Column selection
    if 'English' in df.columns:
        column_to_translate = 'English'
        st.success("✅ Found the 'English' column for translation")
    else:
        columns = df.columns.tolist()
        column_to_translate = st.selectbox("📝 Select a column to translate", columns)
        st.warning("❗ No 'English' column found - please select manually")

    # Language selection
    selected_language_name = st.selectbox("🌐 Select target language", list(deepl_lang_map.keys()))
    target_lang_code_deepl = deepl_lang_map[selected_language_name]
    target_lang_code_google = google_lang_map[selected_language_name]

    if st.button("🚀 Perform Translation"):
        st.session_state.translation_params = {
            'df': df,
            'column': column_to_translate,
            'target_deepl': target_lang_code_deepl,
            'target_google': target_lang_code_google
        }
        st.session_state.show_confirm_buttons = True

    if st.session_state.get('show_confirm_buttons'):
        df[column_to_translate] = df[column_to_translate].astype(str)
        num_characters = count_translatable_characters(df, column_to_translate)

        st.subheader("📊 Translation Details")
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Column", column_to_translate)
        col2.metric("Target Language", f"{selected_language_name} ({target_lang_code_deepl})")
        col3.metric("Characters to Translate", f"{num_characters:,}")

        current_used = used if used else 0
        current_remaining = remaining if remaining else limit
        available_credits = current_remaining - num_characters

        st.write(f"• Available credits: {current_remaining:,} characters")
        st.write(f"• Estimated characters to be translated: {num_characters:,}")
        st.write(f"• Available credits after translation: {available_credits:,} characters")

        proceed = st.button("✅ Proceed with Translation")
        back = st.button("❌ Go Back")

        if proceed:
            st.session_state.show_confirm_buttons = False
            with st.spinner(f"Translating {num_characters:,} characters..."):
                try:
                    df = st.session_state.translation_params['df']
                    col = st.session_state.translation_params['column']
                    tgt_deepl = st.session_state.translation_params['target_deepl']
                    tgt_google = st.session_state.translation_params['target_google']

                    # Perform translation
                    df = perform_translation(df, col, tgt_deepl, tgt_google)

                    # Evaluate translation quality
                    originals = df[col].astype(str).tolist()
                    backtrans = df["Back_Translated"].astype(str).tolist()
                    st.info("Evaluating translation quality (semantic and text similarity)...")
                    df["Semantic_Similarity"] = calculate_semantic_similarity_batch(originals, backtrans)
                    df["Text_Similarity"] = calculate_text_similarity_batch(originals, backtrans)

                    # Show results full-width
                    st.success("✅ Translation and evaluation completed successfully!")
                    st.subheader("📋 Translated DataFrame Preview")
                    st.dataframe(df, use_container_width=True, height=400)

                    # Manual download button directly below the table
                    csv = df.to_csv(index=False).encode('utf-8')
                    download_filename = "translated_evaluated_output.csv"
                    st.download_button(
                        label="⬇️ Download Translated CSV",
                        data=csv,
                        file_name=download_filename,
                        mime="text/csv"
                    )

                    # Optional: Trigger invisible auto-download (if really necessary)
                    b64 = base64.b64encode(csv).decode()
                    download_link = f"""
                        <html>
                            <body>
                                <a id="download_link" href="data:file/csv;base64,{b64}" download="{download_filename}" style="display:none;"></a>
                                <script>document.getElementById('download_link').click();</script>
                            </body>
                        </html>
                    """
                    html(download_link)

                except Exception as e:
                    st.error(f"🚨 Translation or evaluation failed: {str(e)}")
                    print(e)

        if back:
            st.session_state.show_confirm_buttons = False
            st.rerun()

        # Button to start fresh with a new CSV
        if uploaded_file is not None and not st.session_state.get('show_confirm_buttons'):
            st.subheader("🔄 Start Again with a New CSV")
            restart_button = st.button("Start Over with a New CSV")
            if restart_button:
                st.session_state.clear()
                st.rerun()
