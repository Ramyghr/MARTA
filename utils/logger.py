import logging
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow bold",
        "error": "red bold",
        "success": "green bold",
        "timestamp": "dim white",
    }
)

_console = Console(theme=_THEME, highlight=False)


class MARTALogger:
    """Module-level singleton logger backed by rich for coloured, timestamped output."""

    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=_console,
                    show_path=False,
                    rich_tracebacks=True,
                    markup=True,
                )
            ],
        )
        # Silence noisy third-party loggers
        for noisy in ("ccxt", "urllib3", "requests", "yfinance", "git", "mlflow", "httpcore", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        self._log = logging.getLogger("marta")
        self._log.setLevel(logging.DEBUG)

    def _ts(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, message: str, *args, **kwargs) -> None:
        self._log.info(f"[cyan]{message}[/cyan]", *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        self._log.warning(f"[yellow bold]{message}[/yellow bold]", *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        self._log.error(f"[red bold]{message}[/red bold]", *args, **kwargs)

    def success(self, message: str, *args, **kwargs) -> None:
        self._log.info(f"[green bold]{message}[/green bold]", *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        self._log.debug(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        self._log.critical(f"[red bold blink]{message}[/red bold blink]", *args, **kwargs)


logger: MARTALogger = MARTALogger()
