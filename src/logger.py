# FILE: logger.py
from reporting.terminal_html import terminal_html
from settings import app_settings
from pathlib import Path

class ST_TerminalLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ST_TerminalLogger, cls).__new__(cls)
            cls._instance.initialize_logger()
        return cls._instance

    def initialize_logger(self):
        # Ensure output folder exists before calling terminal_html
        Path(app_settings.output_folder).mkdir(parents=True, exist_ok=True)
        
        formatted_datetime, p = terminal_html(app_settings.output_folder)
        self._logger_instance = p
        p.print(f"[logger.py] terminal_html folder = {app_settings.output_folder}")

    def get_logger(self):
        return self._logger_instance

def log_message(message):
    logger = ST_TerminalLogger().get_logger()
    logger.print(message)

def log_image(fig):
    logger = ST_TerminalLogger().get_logger()
    logger.show(fig)


def reset_logger():
    """Reset the singleton logger (useful if output_folder is changed after initialization)."""
    ST_TerminalLogger._instance = None










