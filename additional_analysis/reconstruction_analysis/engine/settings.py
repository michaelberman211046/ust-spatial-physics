class AppSettings:
    output_folder: str = "outputs/logs/"
 

app_settings = AppSettings()


def set_output_folder(path: str) -> None:
    """Set output folder for logger artifacts (terminal/HTML logs, figures)."""
    app_settings.output_folder = path










