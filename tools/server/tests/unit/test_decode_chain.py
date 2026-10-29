import os
import tempfile

import pytest
import requests
from utils import *

# --decode-chain: a lone greedy slot sampling on the backend decodes one token ahead. Every case runs on a server with
# chaining on and on one with it off and must give the same tokens; the server log must show that the chain ran.

server = ServerPreset.tinyllama2()

# a sampler list the backend runs whole; temperature 0 ends it on greedy
GREEDY_BACKEND = {"samplers": ["top_k", "temperature"], "temperature": 0.0, "backend_sampling": True}

PROMPT = "I believe the meaning of life is"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def start(decode_chain: int, **kwargs) -> tuple[ServerProcess, str]:
    s = ServerPreset.tinyllama2()
    s.backend_sampling = True
    s.decode_chain = decode_chain
    s.n_slots = 1
    for k, v in kwargs.items():
        setattr(s, k, v)
    fd, path = tempfile.mkstemp(prefix=f"decode-chain-{decode_chain}-", suffix=".log")
    os.close(fd)
    s.log_path = path
    s.start()
    return s, path


def n_chained(log_path: str) -> int:
    n = 0
    with open(log_path, errors="replace") as f:
        for line in f:
            if "decode chaining ends" in line:
                n += int(line.split("decode chaining ends")[1].split(":")[1].split("tokens chained")[0])
    return n


def run_both(fn, **kwargs):
    """fn(server) -> comparable result; returns (on, off, chained tokens on, chained tokens off)"""
    res = {}
    chained = {}
    for mode in (0, 1):
        s, log = start(mode, **kwargs)
        try:
            res[mode] = fn(s)
        except Exception:
            with open(log, errors="replace") as f:
                print(f"--- server log (decode_chain = {mode}), last lines ---")
                print("".join(f.readlines()[-60:]))
            raise
        finally:
            s.stop()
        chained[mode] = n_chained(log)
        os.remove(log)
    return res[1], res[0], chained[1], chained[0]


def complete(s: ServerProcess, prompt, **data):
    body = {"prompt": prompt, "return_tokens": True, "cache_prompt": True, **GREEDY_BACKEND, **data}
    res = s.make_request("POST", "/completion", data=body)
    assert res.status_code == 200, res.body
    return {k: res.body[k] for k in ("content", "tokens", "stop_type", "stopping_word", "truncated")} | \
        {"predicted_n": res.body["timings"]["predicted_n"]}


def complete_stream(s: ServerProcess, prompt, **data):
    body = {"prompt": prompt, "stream": True, "cache_prompt": True, **GREEDY_BACKEND, **data}
    content, tokens, last = "", [], None
    for ev in s.make_stream_request("POST", "/completion", data=body):
        content += ev.get("content", "")
        tokens += ev.get("tokens", [])
        last = ev
    assert last is not None and last["stop"]
    return {"content": content, "tokens": tokens, "stop_type": last["stop_type"], "predicted_n": last["timings"]["predicted_n"]}


@pytest.mark.parametrize("n_predict", [1, 2, 3, 64])
def test_decode_chain_max_tokens(n_predict: int):
    on, off, c_on, c_off = run_both(lambda s: [complete(s, PROMPT, n_predict=n_predict) for _ in range(2)])
    assert on == off
    assert on[0]["predicted_n"] == n_predict and on[0]["stop_type"] == "limit"
    assert c_off == 0
    if n_predict >= 3:
        assert c_on > 0


def test_decode_chain_eos():
    # ignore_eos off, long budget: stories end on EOS
    prompts = ["Once upon a time", "Lily and Tom went to the park", "The little dog was sad", PROMPT]
    on, off, c_on, _ = run_both(lambda s: [complete(s, p, n_predict=300) for p in prompts])
    assert on == off
    assert c_on > 0
    if not any(r["stop_type"] == "eos" for r in on):
        pytest.skip("no prompt ended on EOS: " + str([r["stop_type"] for r in on]))


def test_decode_chain_stop_string():
    def fn(s):
        first = complete(s, PROMPT, n_predict=64)
        words = first["content"].split()
        assert len(words) > 6
        stop = words[5]
        return [first, complete(s, PROMPT, n_predict=64, stop=[stop]), complete_stream(s, PROMPT, n_predict=64, stop=[stop])]
    on, off, c_on, _ = run_both(fn)
    assert on == off
    assert on[1]["stop_type"] == "word"
    assert c_on > 0


def test_decode_chain_stream():
    on, off, c_on, _ = run_both(lambda s: [complete_stream(s, PROMPT, n_predict=48), complete(s, PROMPT, n_predict=48)])
    assert on == off
    assert on[0]["content"] == on[1]["content"]
    assert c_on > 0


def test_decode_chain_multi_turn():
    # each turn's prompt carries the previous answer: the cached slot (incl. the extra decoded token when chained) is reused
    def fn(s):
        out = []
        prompt = PROMPT
        for turn in range(4):
            r = complete(s, prompt, n_predict=24 + 8 * turn)
            out.append(r)
            prompt = prompt + r["content"] + " And then"
        res = s.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 16, **GREEDY_BACKEND})
        out.append({"content": res.body["content"], "cache_n": res.body["timings"]["cache_n"] > 0})
        return out
    on, off, c_on, _ = run_both(fn)
    assert on == off
    assert on[-1]["cache_n"]
    assert c_on > 0


def test_decode_chain_context_limit():
    # no context shift: the slot stops at the context limit (chaining hands over to the ordinary path before it)
    on, off, c_on, _ = run_both(lambda s: [complete(s, PROMPT, n_predict=-1, ignore_eos=True)], n_ctx=128, n_predict=-1)
    assert on == off
    assert on[0]["truncated"] or on[0]["stop_type"] == "limit"
    assert c_on > 0


def test_decode_chain_context_shift():
    on, off, c_on, _ = run_both(lambda s: [complete(s, PROMPT, n_predict=200, ignore_eos=True)], n_ctx=128, n_predict=-1,
                                enable_ctx_shift=True)
    assert on == off
    assert on[0]["predicted_n"] == 200
    assert c_on > 0


def test_decode_chain_not_engaged():
    # host sampling, n_probs, a sampler the backend cannot run whole: chaining stays off and nothing changes
    def fn(s):
        r = []
        r.append(complete(s, PROMPT, n_predict=24, backend_sampling=False))
        res = s.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 8, "n_probs": 3, **GREEDY_BACKEND})
        r.append([t["id"] for t in res.body["completion_probabilities"]])
        # greedy, but top_n_sigma has no backend version: the backend runs a prefix, the CPU picks the token
        r.append(complete(s, PROMPT, n_predict=24, samplers=["top_n_sigma", "top_k", "temperature"], top_n_sigma=1.0))
        r.append(complete(s, PROMPT, n_predict=24, temperature=0.8, seed=7))
        return r
    on, off, c_on, c_off = run_both(fn)
    assert on == off
    assert c_on == 0 and c_off == 0


def test_decode_chain_two_slots():
    # a second request arrives while the first one is chained: the chain ends, both continue on the ordinary path
    def fn(s):
        tasks = [
            ("POST", "/completion", {"prompt": PROMPT, "n_predict": 96, "ignore_eos": True, "return_tokens": True, **GREEDY_BACKEND}),
            ("POST", "/completion", {"prompt": "Once upon a time", "n_predict": 48, "ignore_eos": True, "return_tokens": True, **GREEDY_BACKEND}),
        ]
        results = parallel_function_calls([(s.make_request, t) for t in tasks])
        assert all(r is not None for r in results), "a request failed"
        return [(r.body["content"], r.body["tokens"]) for r in results]
    on, off, _, _ = run_both(fn, n_slots=2)
    assert on == off


def test_decode_chain_cancel():
    # a client that goes away mid-stream: the slot is released, the next request is unaffected
    def fn(s):
        url = f"http://{s.server_host}:{s.server_port}/completion"
        body = {"prompt": PROMPT, "n_predict": 200, "ignore_eos": True, "stream": True, **GREEDY_BACKEND}
        with requests.post(url, json=body, stream=True, timeout=60) as r:
            for i, _ in enumerate(r.iter_lines()):
                if i > 10:
                    break
        return complete(s, "Once upon a time", n_predict=32)
    on, off, _, _ = run_both(fn)
    assert on == off
