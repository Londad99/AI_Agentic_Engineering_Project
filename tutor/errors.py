"""Project exceptions.

Library code never raises SystemExit: it inherits from BaseException so it slips past
`except Exception`, and IPython swallows it, printing "To exit: use 'exit', 'quit'..."
instead of the message. Only the CLI scripts translate these into an exit.
"""


class TutorError(Exception):
    """Base class for every error this project raises on purpose."""


class IngestionError(TutorError):
    """No usable text could be extracted from the given path."""


class ConfigError(TutorError):
    """A setting in .env is wrong in a way we can detect before calling the API."""
