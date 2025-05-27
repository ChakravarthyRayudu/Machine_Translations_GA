# app.py
import streamlit as st
import os
import shutil
import pandas as pd
from dotenv import load_dotenv
from streamlit_google_auth import Authenticate
from utils import load_language_map, count_translatable_characters, list_backup_sessions
from translate import perform_translation
from fetch_credits import get_deepl_usage
from config import DEEPL_API_KEY, GOOGLE_API_KEY
from streamlit.components.v1 import html
import base64

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Playshifu - Language Translator", page_icon="🌎")

# --- Load Environment Variables ---
load_dotenv()

# --- Google Authentication ---
authenticator = Authenticate(
    secret_credentials_path='google_credentials_temp.json',
    cookie_name='my_cookie_name',
    cookie_key='this_is_secret',
    redirect_uri='http://43.204.227.170.nip.io:8501',
)

try:
    authenticator.check_authentification()
except Exception as e:
    st.error("Authentication failed. Please try again.")
    st.stop()

if not st.session_state.get('connected', False):
    st.info("Please log in with Google to use the app.")
    authorization_url = authenticator.get_authorization_url()
    st.link_button('Login with Google', authorization_url)
    st.stop()

user_info = st.session_state['user_info']
st.image(user_info.get('picture'), width=60)
st.success(f"Hello, {user_info.get('name')} ({user_info.get('email')})")

if st.button('Log out'):
    authenticator.logout()
    st.stop()

# --- Main Tabs Layout ---
tab_translate, tab_backups = st.tabs(["🌍 Translate", "📂 Backups"])

with tab_translate:
    st.title("Playshifu - Language Translator")
    
    # --- Styling ---
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        [data-testid="stMetricLabel"] { font-size: 1rem !important; }
        .css-1d391kg { width: 200px; }
        .stDataFrame div[data-testid="stHorizontalBlock"] {
            overflow-x: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Load Language Maps ---
    deepl_lang_map = load_language_map('deepl_languages.json')
    google_lang_map = load_language_map('google_languages.json')

    # --- Sidebar DeepL Usage ---
    used, limit, remaining, cost = get_deepl_usage(DEEPL_API_KEY)
    st.sidebar.header("💳 DeepL Usage")
    if used is not None:
        st.sidebar.metric("Total Credits Used", f"{used:,}")
        st.sidebar.metric("Total Cost Incurred (USD)", f"${cost:.2f}")
    else:
        st.sidebar.warning("Could not fetch DeepL usage stats.")

    # --- File Upload and Preview ---
    uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"], key="csv_uploader")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        st.subheader("🔎 Preview of your uploaded file:")
        st.dataframe(df, use_container_width=True, height=500)

        # --- Column Selection ---
        if 'English' in df.columns:
            column_to_translate = 'English'
            st.success("✅ Found the 'English' column for translation")
        else:
            columns = df.columns.tolist()
            column_to_translate = st.selectbox("📝 Select a column to translate", columns)
            st.warning("❗ No 'English' column found - please select manually")

        # --- Language Selection ---
        selected_language_name = st.selectbox("🌐 Select target language", list(deepl_lang_map.keys()))
        target_lang_code_deepl = deepl_lang_map[selected_language_name]
        target_lang_code_google = google_lang_map[selected_language_name]

        if st.button("🚀 Perform Translation"):
            st.session_state.translation_params = {
                'df': df,
                'column': column_to_translate,
                'target_deepl': target_lang_code_deepl,
                'target_google': target_lang_code_google,
                'filename': uploaded_file.name
            }
            st.session_state.show_confirm_buttons = True

        if st.session_state.get('show_confirm_buttons'):
            df[column_to_translate] = df[column_to_translate].astype(str)
            num_characters = count_translatable_characters(df, column_to_translate)
            estimated_cost = num_characters / 40000  # $1 per 40,000 chars

            st.subheader("📊 Translation Details")
            col1, col2, col3 = st.columns(3)
            col1.metric("Selected Column", column_to_translate)
            col2.metric("Target Language", f"{selected_language_name} ({target_lang_code_deepl})")
            col3.metric("Characters to Translate", f"{num_characters:,}")

            st.warning(f"Estimated cost for this translation: ${estimated_cost:.2f} USD")

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
                        original_filename = st.session_state.translation_params['filename']

                        df = perform_translation(df, col, tgt_deepl, tgt_google, original_filename)

                        st.success("✅ Translation completed successfully!")
                        st.subheader("📋 Translated DataFrame Preview")
                        st.dataframe(df, use_container_width=True, height=400)

                        csv = df.to_csv(index=False).encode('utf-8')
                        download_filename = "translated_output.csv"
                        st.download_button(
                            label="⬇️ Download Translated CSV",
                            data=csv,
                            file_name=download_filename,
                            mime="text/csv"
                        )

                        b64 = base64.b64encode(csv).decode()
                        download_link = f"""
                            <html><body><a id=\"download_link\" href=\"data:file/csv;base64,{b64}\" download=\"{download_filename}\" style=\"display:none;\"></a><script>document.getElementById('download_link').click();</script></body></html>
                        """
                        html(download_link)

                    except Exception as e:
                        st.error(f"🚨 Translation failed: {str(e)}\n**Partial translations saved in backups section**")
            if back:
                st.session_state.show_confirm_buttons = False
                st.rerun()

            if uploaded_file is not None and not st.session_state.get('show_confirm_buttons'):
                st.subheader("🔄 Start Again with a New CSV")
                restart_button = st.button("Start Over with a New CSV")
                if restart_button:
                    st.session_state.clear()
                    st.rerun()

with tab_backups:
    st.header("📂 Translation Backup Sessions")
    
    backup_sessions = list_backup_sessions()
    if not backup_sessions:
        st.info("No backup sessions found.")
    else:
        st.write("### Available Backup Sessions")
        backup_sessions.sort(reverse=True)
        for session in backup_sessions:
            with st.expander(f"📁 {session}", expanded=False):
                backup_path = os.path.join("backups", session)
                zip_filename = f"{session}.zip"
                zip_path = os.path.join("backups", zip_filename)
                # Only create ZIP if it doesn't exist
                if not os.path.exists(zip_path):
                    shutil.make_archive(
                        os.path.join("backups", session),
                        'zip',
                        backup_path
                    )
                with open(zip_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Full Session as ZIP",
                        data=f,
                        file_name=zip_filename,
                        key=f"zip_{session}"
                    )
                st.write("All files for this session are included in the ZIP.")
