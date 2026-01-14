from collections.abc import Iterable


def nice_join(
    seq: Iterable,
    separator: str = ", ",
    conjunction: str | None = "or",
    warp_in_single_quote: bool = False,
) -> str:
    """Joins items of a sequence into English phrases using the representation of the items and
    a given conjunction between the last two elements. If the sequence is an Iterable of strings and
    the conjunction is None, this function reproduces the behavior of the built-in-function `join`.

    Args:
        seq (Iterable): a sequence of strings to join
        separator (str, ", "): a delimiter to use between items
        conjunction (str | None, "or"): a conjunction to use between the last two items, or None to
            reproduce basic join behavior
        wrap_in_single_quote (bool, False): if True, all items of the sequence are warped in extra
            single quotes

    Returns:
        str: a joined string

    Examples:

        >>> from mpisppy.utils import nice_join

        >>> nice_join(["a", "b", "c"])
        "a, b or c"
    """
    q = "'" if warp_in_single_quote else ""
    seq = [f"{q}{x!s}{q}" for x in seq]

    if len(seq) <= 1 or conjunction is None:
        return separator.join(seq)
    else:
        return f"{separator.join(seq[:-1])} {conjunction} {seq[-1]}"
