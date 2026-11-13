import os
import random
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.drain()
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def _segment(name, n):
    # token ids of a named segment: segments with the same name share an exact prefix
    # (the model has 512 tokens in its vocabulary, the first 3 are special)
    r = random.Random(name)
    return [r.randint(3, 511) for _ in range(n)]


def _complete(tokens, id_slot=-1):
    res = server.make_request("POST", "/completion", data={
        "prompt": tokens,
        "id_slot": id_slot,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body["timings"]["prompt_n"]


def _start_full_pool_server(cache_ram):
    # three conversations of 750 tokens do not fit into a unified pool of 2048 cells together
    global server
    server.n_slots = 3
    server.n_ctx = 2048
    server.n_batch = 256
    server.cache_ram = cache_ram
    server.no_cache_idle_slots = True
    server.start()
    return LogReader(server.log_path)


# the least recently used idle slot makes room and goes to the cache, so a returning conversation is restored
def test_lru_purge_restores_returning_conversation():
    log = _start_full_pool_server(cache_ram=32)
    a, b, c = (_segment(name, 750) for name in "abc")

    assert _complete(b, id_slot=1) == 750
    assert _complete(a, id_slot=0) == 750
    assert _complete(c, id_slot=2) == 750  # the pool is full, so b's slot is purged
    assert "purging slot 1" in log.drain()

    assert _complete(b) <= 4
    assert _complete(a) <= 4


# a restore that fails for lack of room empties the slot's sequence, and the save that makes room can evict
# the very entry that was to be restored: the slot must not keep the tokens of the sequence it has lost
def test_failed_restore_with_evicted_entry():
    log = _start_full_pool_server(cache_ram=1)
    shared = _segment("shared", 125)
    t = shared + _segment("t", 625)
    u = _segment("u", 750)
    w = _segment("w", 500)

    _complete(t, id_slot=0)
    _complete(u, id_slot=1)
    _complete(w, id_slot=2)
    _complete(shared + _segment("v", 250), id_slot=0)  # t goes to the cache
    _complete(w + _segment("w2", 250), id_slot=2)
    log.drain()

    # slot 0 is chosen by the shared prefix, but t does not fit next to the two other slots
    # room is made by purging slot 1, whose save evicts t from the cache
    assert _complete(t + _segment("z", 12)) > 0
    logs = log.drain()
    assert "failed to restore state" in logs
    assert "removing oldest entry" in logs

    assert server.make_request("GET", "/health").status_code == 200


# the saves that make room can evict the entry that is to be restored: no further slots are purged for it then
def test_failed_restore_stops_when_entry_is_evicted():
    log = _start_full_pool_server(cache_ram=1)
    shared = _segment("shared", 100)

    _complete(shared + _segment("t", 800), id_slot=0)
    _complete(shared + _segment("x", 296), id_slot=0)  # t goes to the cache
    _complete(shared + _segment("v", 150), id_slot=0)  # x goes to the cache
    _complete(_segment("u", 100), id_slot=1)
    _complete(_segment("w", 1680), id_slot=2)
    log.drain()

    # t does not fit, purging slot 1 evicts it, and x, the next best match, does not fit either
    _complete(shared + _segment("t", 800) + _segment("z", 12))
    logs = log.drain()
    assert logs.count("failed to restore state") == 2
    assert logs.count("looking for better prompt") == 2

    assert server.make_request("GET", "/health").status_code == 200
