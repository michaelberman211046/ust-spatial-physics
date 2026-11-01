"""Append-only text logging for numerical evaluation."""
from pathlib import Path
from settings import app_settings


def log_message(message):
    print(message,flush=True)
    folder=Path(app_settings.output_folder)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'engine.log').open('a',encoding='utf-8') as stream:
        stream.write(str(message)+'\n')


def log_image(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def reset_logger():
    pass
