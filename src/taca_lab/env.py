"""Multi-turn calculator environment.

Problems are left-to-right chains of single-digit binary operations.
The tool is a small expression evaluator (the paper's Python code
execution, in miniature): it evaluates an arithmetic expression over
digits, '+', '-', '*', parentheses and '#' (the previous result) with
standard precedence. A problem can therefore be solved one-shot (one
fully parenthesised call), step-by-step (one binary op per call), or
answered directly. Anything malformed (missing operand, unbalanced
paren, '#' with no previous result, trailing operator) returns an
error observation, and a missing paren silently changes the value
(standard precedence) instead of erroring. The policy may retry after
an error, which is the failure-driven retry mechanism of TACA Sec. 3.4.
Rewards are binary terminal answer correctness.
"""

import numpy as np

from .vocab import (
    ANS, ANS_END, CALL, CALL_END, DIGITS, EOS, EQ, ERR, HASH, LPAREN,
    MINUS, OBS, OBS_END, OP_IDS, OPS, PAD, PLUS, Q, RPAREN, TIMES,
    int_to_tokens, tokens_to_int,
)

MAX_TURNS = 6          # paper: at most six generated turns
MAX_TURN_TOKENS = 22   # a one-shot tier-4 call is 19 tokens + markers


# ---------------------------------------------------------------- problems
def make_problem(rng, tier=None):
    """Return (operands, ops) for a left-to-right chain.

    tier 1: a op b; tier 2: (a op b) op c; tier 3: ((a op b) op c) op d.
    Rejects chains whose intermediate or final value leaves [0, 999].
    """
    if tier is None:
        tier = int(rng.integers(1, 4))
    while True:
        n_ops = tier
        operands = [int(rng.integers(1, 10)) for _ in range(n_ops + 1)]
        ops = [int(rng.choice(list(OP_IDS))) for _ in range(n_ops)]
        try:
            value = fold(operands, ops)
        except ValueError:
            continue
        if 0 <= value <= 999:
            return operands, ops


def fold(operands, ops):
    """Left-to-right evaluation; raises ValueError if an intermediate
    value leaves [0, 999] (keeps answers to at most three digits)."""
    value = operands[0]
    for op, b in zip(ops, operands[1:]):
        if op == 10:
            value = value + b
        elif op == 11:
            value = value - b
        else:
            value = value * b
        if value < 0 or value > 999:
            raise ValueError("out of range")
    return value


def prompt_tokens(operands, ops):
    """Problem statement tokens: ``a op b op c ?``."""
    toks = [operands[0]]
    for op, b in zip(ops, operands[1:]):
        toks += [op, b]
    return toks + [Q]


def answer_value(operands, ops):
    return fold(operands, ops)


# ---------------------------------------------------------------- tool
def _parse_expr(tokens, last_value):
    """Recursive-descent parser for the tool expression language.

    Grammar (standard precedence, as in the paper's Python tool):
        expr  := term (('+'|'-') term)*
        term  := factor (('*') factor)*
        factor:= digit | '#' | '(' expr ')'
    Returns the value, or raises ValueError on any malformed input.
    """
    def parse_expr(pos):
        value, pos = parse_term(pos)
        while pos < len(tokens) and tokens[pos] in (PLUS, MINUS):
            op = tokens[pos]
            rhs, pos = parse_term(pos + 1)
            value = value + rhs if op == PLUS else value - rhs
        return value, pos

    def parse_term(pos):
        value, pos = parse_factor(pos)
        while pos < len(tokens) and tokens[pos] == TIMES:
            rhs, pos = parse_factor(pos + 1)
            value = value * rhs
        return value, pos

    def parse_factor(pos):
        if pos >= len(tokens):
            raise ValueError("unexpected end")
        t = tokens[pos]
        if t in DIGITS:
            return t, pos + 1
        if t == HASH:
            if last_value is None:
                raise ValueError("'#' with no previous result")
            return last_value, pos + 1
        if t == LPAREN:
            value, pos = parse_expr(pos + 1)
            if pos >= len(tokens) or tokens[pos] != RPAREN:
                raise ValueError("unbalanced paren")
            return value, pos + 1
        raise ValueError("bad factor")

    value, pos = parse_expr(0)
    if pos != len(tokens):
        raise ValueError("trailing tokens")
    return value


def execute_call(content, last_value):
    """Evaluate the tool-call expression with standard precedence.

    content: the tokens between <call> and </call> (an arithmetic
    expression over digits, '+', '-', '*', parentheses and '#').
    Returns (value, ok): ok is False for any malformed expression, in
    which case value is None.
    """
    if not content:
        return None, False
    try:
        return _parse_expr(content, last_value), True
    except ValueError:
        return None, False


def observation_tokens(value, ok):
    if ok:
        return [OBS, EQ] + int_to_tokens(value) + [OBS_END]
    return [OBS, ERR, OBS_END]


# ---------------------------------------------------------------- parsing
def parse_turn(turn):
    """Classify one generated turn.

    Returns (kind, payload): kind in {"call", "answer", "garbage"}.
    A call payload is the token list between <call> and </call>; an
    answer payload is the parsed integer (or None if malformed).
    """
    if not turn or turn[0] == EOS:  # EOS / empty
        return "garbage", None
    if turn[0] == CALL:
        if CALL_END in turn:
            end = turn.index(CALL_END)
            return "call", turn[1:end]
        return "garbage", None
    if turn[0] == ANS:
        if ANS_END in turn:
            end = turn.index(ANS_END)
            content = turn[1:end]
            # plain answer: digits only. Derivation answer:
            # <ans> a op b = v </ans> -> the value follows the last "=".
            if EQ in content:
                content = content[content.index(EQ) + 1:]
            return "answer", tokens_to_int(content)
        return "garbage", None
    return "garbage", None


def expert_call_tokens(operands, ops):
    """The canonical tool-use trace for a problem: one call per step,
    ``#`` referring to the previous result, then the final answer."""
    calls = []
    value = operands[0]
    for i, (op, b) in enumerate(zip(ops, operands[1:])):
        x = operands[0] if i == 0 else HASH
        calls.append([CALL, x, op, b, CALL_END])
        value = {10: value + b, 11: value - b, 12: value * b}[op]
    return calls


def oneshot_expr(operands, ops):
    """The fully parenthesised left-to-right expression for a problem,
    e.g. ((3 + 4) * 2) for operands [3,4,2], ops [+ ,*].

    The tool evaluates with standard precedence, so the parentheses are
    what make the one-shot call compute the left-to-right value; a
    missing paren is a real failure mode (wrong value, not an error).
    """
    expr = [operands[0]]
    for op, b in zip(ops, operands[1:]):
        expr = [LPAREN] + expr + [op, b, RPAREN]
    return expr


def expert_trace(operands, ops, rng, mode="tool"):
    """A full expert trajectory as (sequence, response_mask) pieces.

    Returns a list of (tokens, is_response) segments: prompt (context),
    then alternating response turns and observations. Modes:
    - "tool": one binary-op call per step (step-by-step).
    - "oneshot": a single call with the fully parenthesised expression.
    - "direct": answer immediately with a written derivation (tier 1).
    - "recovery": a deliberately malformed first call (missing operand
      for step-by-step, unbalanced paren for one-shot) that errors,
      then the correct calls in the same style.
    """
    prompt = prompt_tokens(operands, ops)
    answer = [ANS] + int_to_tokens(answer_value(operands, ops)) + [ANS_END]
    segments = [(prompt, False)]
    if mode == "direct":
        # Direct answers write the derivation first (<ans> a op b = v
        # </ans>): the extra tokens give the policy compute before it
        # must produce the value, and keep the format parseable.
        deriv = [ANS, operands[0], ops[0], operands[1], EQ] + \
            int_to_tokens(answer_value(operands, ops)) + [ANS_END]
        segments.append((deriv, True))
        return segments
    style = "oneshot" if mode == "oneshot" else "tool"
    if mode == "recovery":
        style = "oneshot" if rng.random() < 0.5 else "tool"
    last_value = None
    if mode == "recovery":
        if style == "tool":
            bad = [CALL, operands[0], ops[0], CALL_END]  # missing operand
        else:
            bad = [CALL] + oneshot_expr(operands, ops)[:-1] + [CALL_END]
        segments.append((bad, True))
        segments.append((observation_tokens(None, False), False))
    if style == "oneshot":
        call = [CALL] + oneshot_expr(operands, ops) + [CALL_END]
        value, ok = execute_call(call[1:-1], last_value)
        assert ok
        segments.append((call, True))
        segments.append((observation_tokens(value, True), False))
    else:
        for call in expert_call_tokens(operands, ops):
            value, ok = execute_call(call[1:-1], last_value)
            assert ok
            segments.append((call, True))
            segments.append((observation_tokens(value, True), False))
            last_value = value
    segments.append((answer, True))
    return segments


def trace_to_arrays(segments):
    """Flatten segments into (tokens, response_mask) for SFT."""
    tokens, mask = [], []
    for toks, is_resp in segments:
        tokens += toks
        mask += [1 if is_resp else 0] * len(toks)
    return tokens, mask


def make_sft_dataset(rng, n=6000, tier_p=(0.25, 0.40, 0.35), direct_p=0.40):
    """SFT examples as (tokens, response_mask) pairs.

    Tier mix set by tier_p. Tier 1: direct_p direct answer, the rest
    split between one-shot (0.5) and step-by-step (0.4) tool use and
    recovery (0.1). Tiers 2-3: one-shot 40%, step-by-step 50%,
    recovery 10%. Direct traces only ever carry the correct answer.
    """
    out = []
    for _ in range(n):
        tier = int(rng.choice([1, 2, 3], p=list(tier_p)))
        operands, ops = make_problem(rng, tier=tier)
        u = rng.random()
        if tier == 1 and u < direct_p:
            mode = "direct"
        elif u < 0.90:
            mode = "oneshot" if rng.random() < 0.45 else "tool"
        else:
            mode = "recovery"
        segments = expert_trace(operands, ops, rng, mode=mode)
        out.append(trace_to_arrays(segments))
    return out
