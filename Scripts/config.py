# config.py

import os
from dotenv import load_dotenv

load_dotenv()

DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


