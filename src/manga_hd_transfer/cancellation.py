"""Shared cooperative-cancellation signal for processing and output tasks."""


class PipelineCancelled(RuntimeError):
    """Raised only at safe stage boundaries when the user cancels work."""
