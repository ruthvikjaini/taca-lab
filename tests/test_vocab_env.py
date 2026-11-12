"""Tests for the vocabulary and the calculator environment."""

import numpy as np

from taca_lab.env import (
    MAX_TURN_TOKENS, execute_call, expert_trace, fold, make_problem,
    make_sft_dataset, observation_tokens, oneshot_expr, parse_turn,
    prompt_tokens, trace_to_arrays,
)
from taca_lab.vocab import (
    ANS, ANS_END, CALL, CALL_END, EQ, ERR, HASH, LPAREN, OBS, OBS_END, PAD,
    Q, RPAREN, decode, int_to_tokens, tokens_to_int,
)


def test_vocab_roundtrip():
    for v in [0, 5, 42, 999, -7]:
        assert tokens_to_int(int_to_tokens(v)) == v
    assert tokens_to_int([]) is None
    assert tokens_to_int([1, 99]) is None
    assert decode([3, 10, 4, 24]) == "3 + 4 ?"


def test_fold_left_to_right():
    # 3 + 4 * 2 is ((3+4)*2) = 14, not 3+(4*2) = 11
    assert fold([3, 4, 2], [10, 12]) == 14
    assert fold([9, 9, 9], [12, 12]) == 729
    try:
        fold([1, 9], [11])
    except ValueError:
        pass
    else:
        raise AssertionError("negative intermediate must be rejected")


def test_make_problem_ranges():
    rng = np.random.default_rng(0)
    for tier in (1, 2, 3):
        for _ in range(50):
            operands, ops = make_problem(rng, tier=tier)
            assert len(ops) == tier
            assert 0 <= fold(operands, ops) <= 999


def test_prompt_tokens():
    assert prompt_tokens([3, 4, 2], [10, 12]) == [3, 10, 4, 12, 2, Q]


def test_execute_call_ok_and_errors():
    v, ok = execute_call([3, 10, 4], None)
    assert ok and v == 7
    v, ok = execute_call([HASH, 12, 2], 7)
    assert ok and v == 14
    # '#' with no previous result
    _, ok = execute_call([HASH, 10, 2], None)
    assert not ok
    # wrong arity / unknown token / bad op
    _, ok = execute_call([3, 10], None)
    assert not ok
    _, ok = execute_call([3, 99, 4], None)
    assert not ok
    _, ok = execute_call([OBS, 10, 4], None)
    assert not ok
    # standard precedence with parentheses: ((3+4)*2) = 14
    v, ok = execute_call([LPAREN, 3, 10, 4, RPAREN, 12, 2], None)
    assert ok and v == 14
    # without parens standard precedence gives 3+(4*2) = 11
    v, ok = execute_call([3, 10, 4, 12, 2], None)
    assert ok and v == 11
    # unbalanced paren / trailing operator error
    _, ok = execute_call([LPAREN, 3, 10, 4, RPAREN, 12], None)
    assert not ok
    _, ok = execute_call([LPAREN, 3, 10, 4, 12, 2], None)
    assert not ok
    # '#' inside a larger expression
    v, ok = execute_call([LPAREN, HASH, 10, 1, RPAREN, 12, 3], 10)
    assert ok and v == 33


def test_oneshot_expr_matches_left_to_right_value():
    for operands, ops in ([3, 4, 2], [10, 12]), ([9, 9, 9], [12, 12]):
        expr = oneshot_expr(operands, ops)
        v, ok = execute_call(expr, None)
        assert ok and v == fold(operands, ops)


def test_observation_tokens():
    assert observation_tokens(42, True) == [OBS, EQ, 4, 2, OBS_END]
    assert observation_tokens(None, False) == [OBS, ERR, OBS_END]


def test_parse_turn():
    kind, payload = parse_turn([CALL, 3, 10, 4, CALL_END])
    assert kind == "call" and payload == [3, 10, 4]
    kind, payload = parse_turn([ANS, 1, 4, ANS_END])
    assert kind == "answer" and payload == 14
    kind, _ = parse_turn([CALL, 3, 10])  # unterminated
    assert kind == "garbage"
    kind, _ = parse_turn([7, 8, 9])
    assert kind == "garbage"
    kind, _ = parse_turn([])
    assert kind == "garbage"


def test_expert_trace_tool_mode():
    rng = np.random.default_rng(0)
    operands, ops = [3, 4, 2], [10, 12]
    segments = expert_trace(operands, ops, rng, mode="tool")
    tokens, mask = trace_to_arrays(segments)
    # prompt is context; every call and the answer are response
    assert tokens[:4] == [3, 10, 4, 12] or tokens[:5] == [3, 10, 4, 12, 2]
    # last segment is the answer with the correct value
    assert tokens[-4:] == [ANS, 1, 4, ANS_END]
    assert mask[-4:] == [1, 1, 1, 1]
    # observations are context
    assert OBS in tokens
    assert mask[tokens.index(OBS)] == 0


def test_expert_trace_direct_and_recovery():
    rng = np.random.default_rng(0)
    segments = expert_trace([3, 4], [10], rng, mode="direct")
    tokens, _ = trace_to_arrays(segments)
    # derivation format: <ans> 3 + 4 = 7 </ans>
    assert tokens[-7:] == [ANS, 3, 10, 4, EQ, 7, ANS_END]
    kind, payload = parse_turn(tokens[4:])
    assert kind == "answer" and payload == 7
    segments = expert_trace([3, 4, 2], [10, 12], rng, mode="recovery")
    tokens, _ = trace_to_arrays(segments)
    # recovery starts with a malformed call then an ERR observation
    assert ERR in tokens
    assert tokens[-4:] == [ANS, 1, 4, ANS_END]


def test_expert_trace_oneshot_mode():
    rng = np.random.default_rng(0)
    segments = expert_trace([3, 4, 2], [10, 12], rng, mode="oneshot")
    tokens, mask = trace_to_arrays(segments)
    # one call with the parenthesised expression, then the answer
    assert tokens.count(CALL) == 1
    assert tokens[-4:] == [ANS, 1, 4, ANS_END]
    kind, payload = parse_turn(tokens[6:6 + 11])  # prompt is 6 tokens
    assert kind == "call"
    v, ok = execute_call(payload, None)
    assert ok and v == 14


def test_sft_dataset_shapes_and_direct_correctness():
    rng = np.random.default_rng(1)
    data = make_sft_dataset(rng, n=200)
    assert len(data) == 200
    n_direct = 0
    for tokens, mask in data:
        assert len(tokens) == len(mask)
        assert 0 < sum(mask) < len(tokens)
        assert len(tokens) <= 60
        if ANS in tokens and CALL not in tokens:
            n_direct += 1
    assert n_direct > 0  # tier-1 direct answers exist
