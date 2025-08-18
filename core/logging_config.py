"""Configure logging for Evalia."""
import logging
import logging.handlers
import os
import streamlit as st

def configure_evalia_logger():
    if st.session_state.get("_evalia_logger_configured"):
        return logging.getLogger("EvaliaLogger")

    logger = logging.getLogger("EvaliaLogger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    log_file = "static/evalia_debug.log"
    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s - [File: %(pathname)s, Line: %(lineno)d]")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if os.getenv("ENV") != "STREAMLIT_CLOUD":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    for noisy in ("urllib3", "httpx", "PIL", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    st.session_state["_evalia_logger_configured"] = True
    logger.debug("Evalia logger configured.")
    return logger
