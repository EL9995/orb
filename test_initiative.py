"""
test_initiative.py - checks the speak-up-on-its-own logic with a fake clock (no LLM, no mic).

    python3 test_initiative.py        (or: python3 -m pytest test_initiative.py)
"""

import initiative as ini
from initiative import Initiative, is_brush_off, is_shh

MID = lambda: 0.5          # noqa: E731  rng with no jitter: every wait is exactly WAIT[stage]


def fresh(now=0.0):
    return Initiative("Rook", rng=MID, now=now, chattiness=0.5)   # 0.5 = the WAIT table as is


def test_waits_for_silence():
    it = fresh()
    assert it.due(ini.WAIT["check_in"] - 1) is None
    prompt = it.due(ini.WAIT["check_in"])
    assert prompt and "check in" in prompt and "PASS" in prompt
    assert it.stage == "check_in"


def test_jitter_stays_in_range():
    for r in (0.0, 1.0):
        it = Initiative(rng=lambda r=r: r, now=0, chattiness=0.5)
        base = ini.WAIT["check_in"]
        assert abs(it.next_at - base) <= base * ini.JITTER + 1e-9


def test_no_answer_escalates_up_the_ladder():
    it, t = fresh(), 0.0
    seen = []
    for _ in range(6):
        t = it.next_at
        assert it.due(t)
        seen.append(it.stage)
        it.on_spoke(t + 3)               # it says its line; you say nothing
    assert seen == ["check_in", "give_space", "jab", "self_talk", "self_talk", "self_talk"]


def test_brush_off_escalates_real_answer_resets():
    it = fresh()
    t = it.next_at
    it.due(t)
    it.on_spoke(t + 2)
    it.on_user("yeah", t + 5)            # brush-off
    assert it.stage == "give_space"
    t = it.next_at
    it.due(t)
    it.on_spoke(t + 2)
    it.on_user("I'm fixing my bike, the chain keeps slipping", t + 5)
    assert it.stage == "check_in"


def test_talking_first_keeps_it_at_check_in():
    it = fresh()
    it.on_user("yeah", 10)               # not a reply to anything, so not a brush-off
    assert it.stage == "check_in" and it.next_at == 10 + ini.WAIT["check_in"]


def test_pass_doesnt_escalate_or_count():
    it = fresh()
    t = it.next_at
    it.due(t)
    it.passed(t + 1)
    assert it.stage == "check_in" and not it.fired
    t = it.next_at
    it.due(t)
    assert it.stage == "check_in"        # a PASS is not "ignored"


def test_hourly_cap():
    it, t = fresh(), 0.0
    fired = 0
    while t < 3500:
        t = max(t + 1, it.next_at)
        if it.due(t):
            fired += 1
            it.on_spoke(t)
    assert fired == ini.MAX_PER_HOUR
    assert it.due(3599) is None


def test_shh_pauses_until_you_talk():
    it = fresh()
    assert it.on_user("Shh.", 5) == "shh"
    assert it.due(5 + ini.SHH_MINUTES * 60 - 1) is None
    assert it.on_user("okay, what were you saying", 100) == "shh_off"
    assert it.due(100 + ini.WAIT["check_in"]) is not None


def test_shh_wears_off():
    it = fresh()
    it.on_user("be quiet", 0)
    t = ini.SHH_MINUTES * 60 + 1
    it.on_spoke(t - ini.WAIT["check_in"])
    assert it.due(t) is not None


def test_grader_and_shh_words():
    for s in ["yeah", "Yeah.", "mm", "ok", "not now", "Yeah, I guess.", "nope"]:
        assert is_brush_off(s), s
    for s in ["Yeah, I'm making pasta", "I'm busy with homework tonight", "What do you think?"]:
        assert not is_brush_off(s), s
    for s in ["Shh.", "Shhh!", "Rook, be quiet.", "hush", "quiet please", "Shut up, Rook"]:
        assert is_shh(s, "Rook"), s
    for s in ["It's quiet here.", "yeah", "shh I think the baby is sleeping now so"]:
        assert not is_shh(s, "Rook"), s


def test_chattiness_fader():
    it = fresh()
    it.set_chattiness(1.0)                       # 4x sooner, mid-wait
    assert it.next_at == ini.WAIT["check_in"] / 4
    prompt = it.due(it.next_at)
    assert prompt and "PASS" not in prompt       # at full chattiness it must say something
    assert it.cap() == 4 * ini.MAX_PER_HOUR
    it.set_chattiness(0)
    assert it.due(10 ** 6) is None               # 0 = off


def test_pass_offer_follows_chattiness():
    offered = 0
    for i in range(100):
        it = Initiative(rng=lambda i=i: (i + 0.5) / 100, now=0, chattiness=0.7)
        if "PASS" in it.due(10 ** 4):
            offered += 1
    assert offered == 30


def test_grades_your_reply():
    it = fresh()
    it.due(it.next_at)
    it.on_user("mm", 70)
    assert it.last_grade == "brush-off"
    it.due(it.next_at)
    it.on_user("Long day, I just got back from the gym", 300)
    assert it.last_grade == "engaged"
    it.on_user("hello", 310)
    assert it.last_grade is None                 # not a reply to a nudge


def test_long_silence_keeps_the_conversation_in_view():
    import brain
    U = lambda c, n=False: {"role": "user", "content": c, "nudge": n}          # noqa: E731
    A = lambda c, n=False: {"role": "assistant", "content": c, "nudge": n}     # noqa: E731
    h = [U("I'm making pasta"), A("Salt the water.")]
    for i in range(20):                                    # a long silence: 20 speak-ups
        h += [U(f"nudge {i}", True), A(f"line {i}", True)]
    seen = [m["content"] for m in brain.window(h + [U("back! it was good")])]
    assert seen == ["I'm making pasta", "Salt the water.", "nudge 18", "line 18",
                    "nudge 19", "line 19", "back! it was good"]
    assert all("nudge" not in m for m in brain.window(h))  # the marker never reaches Ollama


def test_sleep_after_no_word_from_you():
    it = fresh()
    it.sleep_after_min = 20
    t = 0.0
    while True:                               # it keeps speaking up; you never answer
        t = max(t + 1, it.next_at)
        if t >= 20 * 60:
            break
        if it.due(t):
            it.on_spoke(t)                    # its own lines don't count: sleep is from *your* last word
        assert not it.should_sleep(t)
    assert it.should_sleep(20 * 60)
    assert "sleep" in it.sleep_prompt(20 * 60) and "standby" in it.sleep_prompt(20 * 60)
    it.sleep()
    assert it.due(10 ** 6) is None and it.off(10 ** 6) and not it.should_sleep(10 ** 6)
    it.wake(30 * 60)
    assert not it.asleep and it.stage == "check_in" and not it.should_sleep(30 * 60 + 60)


def test_sleep_never():
    it = fresh()
    it.sleep_after_min = 0
    assert not it.should_sleep(10 ** 6)


def test_end_and_standby_answers():
    from initiative import is_end, is_standby
    for s in ["End the session.", "Shut it down, Rook", "bye", "Stop.", "Turn it off"]:
        assert is_end(s, "Rook") and not is_standby(s, "Rook"), s
    for s in ["Standby.", "Yeah, leave it on.", "stay", "Keep listening please"]:
        assert is_standby(s, "Rook") and not is_end(s, "Rook"), s
    for s in ["Stay cool. The tip to the face.", "The end of the movie was sad",
              "I need to stop by the store", "what's for dinner"]:
        assert not is_end(s, "Rook") and not is_standby(s, "Rook"), s


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} passed")
