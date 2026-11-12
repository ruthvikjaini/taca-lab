"""Vocabulary for the multi-turn calculator testbed.

A problem is a left-to-right chain of single-digit binary operations,
e.g. ``3 + 4 * 2 ?`` means ``((3 + 4) * 2)`` = 14. The policy acts one
turn at a time: it either calls the calculator tool for a single binary
operation (``<call> x op y </call>``) or answers (``<ans> digits </ans>``).
The tool evaluates exactly one binary operation per call, so a 2-step
problem needs two calls and a 3-step problem needs three. ``#`` refers
to the previous tool result. Observations (``<obs> = value </obs>`` or
``<obs> ERR </obs>``) are context only and are never scored.
"""

DIGITS = list(range(10))  # token id == digit value, ids 0..9
PLUS = 10
MINUS = 11
TIMES = 12
HASH = 13
CALL = 14
CALL_END = 15
ANS = 16
ANS_END = 17
OBS = 18
OBS_END = 19
EQ = 20
ERR = 21
EOS = 22
PAD = 23
Q = 24
LPAREN = 25
RPAREN = 26

VOCAB_SIZE = 27

OPS = {PLUS: "+", MINUS: "-", TIMES: "*"}
OP_IDS = {PLUS, MINUS, TIMES}

TOKEN_NAMES = (
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    + ["+", "-", "*", "#", "<call>", "</call>", "<ans>", "</ans>",
       "<obs>", "</obs>", "=", "ERR", "EOS", "PAD", "?", "(", ")"]
)


def decode(tokens):
    """Render a token list as a readable string (PAD/EOS dropped)."""
    return " ".join(TOKEN_NAMES[t] for t in tokens if t not in (PAD, EOS))


def int_to_tokens(value):
    """Non-negative int -> digit tokens; negative ints get a leading MINUS."""
    if value < 0:
        return [MINUS] + int_to_tokens(-value)
    return [int(c) for c in str(value)]


def tokens_to_int(tokens):
    """Digit tokens (optionally MINUS-led) -> int, or None if malformed."""
    if not tokens:
        return None
    neg = False
    if tokens[0] == MINUS:
        neg = True
        tokens = tokens[1:]
    if not tokens or any(t not in DIGITS for t in tokens):
        return None
    value = 0
    for t in tokens:
        value = value * 10 + t
    return -value if neg else value
