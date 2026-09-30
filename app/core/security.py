"""Small security helpers kept independent of the model runtime."""

from secrets import compare_digest


def token_is_valid(supplied: str | None, configured: str | None) -> bool:
    """Accept only a supplied token matching a configured non-empty secret."""
    if not supplied or not configured:
        return False
    return compare_digest(supplied, configured)
