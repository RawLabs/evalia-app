"""API keys & client setup (keep secrets out of repo)."""
import os
import json
import streamlit as st
from core.logging_config import configure_evalia_logger

logger = configure_evalia_logger()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MEMORY_FILE = "evalia_memory.json"

def initialize_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w') as f:
            json.dump([], f)
        logger.info("Initialized memory file: %s", MEMORY_FILE)
