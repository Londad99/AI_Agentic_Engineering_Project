"""Project exceptions.

Why this file exists: library code must never raise SystemExit. SystemExit is a
process-level signal, and it inherits from BaseException rather than Exception,
so it slips past `except Exception` handlers. Worse for us, IPython intercepts it
and prints "To exit: use 'exit', 'quit', or Ctrl-D" instead of the message,
turning a clear diagnosis into a confusing warning. Library code raises a normal
Exception; only the CLI entry points in scripts/ translate it into an exit.
"""


class TutorError(Exception):
    """Base class for every error this project raises on purpose."""


class IngestionError(TutorError):
    """No usable text could be extracted from the given path."""


class ConfigError(TutorError):
    """A setting in .env is wrong in a way we can detect before calling the API."""
