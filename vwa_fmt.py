"""Zahlen im deutschen Format (Dezimalkomma, geschütztes Tausendertrennzeichen für große Zahlen)."""


def de(x: float, nd: int = 2) -> str:
    """Dezimalkomma, `nd` Nachkommastellen."""
    return f"{x:.{nd}f}".replace(".", ",")


def de_g(x: float) -> str:
    """Kürzestschreibweise (0,125; 4) mit Dezimalkomma."""
    return f"{x:g}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma und geschütztem Leerzeichen vor dem Zeichen."""
    return f"{100 * x:.{nd}f}".replace(".", ",") + " %"


def times(x: float, nd: int = 3) -> str:
    """Verhältnis als „1,552fach“-Schreibweise."""
    return de(x, nd) + "fach"


def thousands(n: int) -> str:
    """Ganze Zahl mit schmalem Leerzeichen als Tausendertrenner (ab 10 000; darunter ohne)."""
    return f"{n:,}".replace(",", " ") if n >= 10000 else str(n)
