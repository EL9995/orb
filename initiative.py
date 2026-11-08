"""
initiative.py - lets the character speak up on its own when the room goes quiet.

After a stretch of silence (randomized, so it never feels like a timer) the brain gets a nudge
to say something, or to answer PASS and stay quiet. If you don't really engage, it escalates
through a ladder instead of repeating itself:

    check_in   -> asks how you're doing / what you're up to
    give_space -> your answer was a brush-off (or there was none): it backs off, briefly
    jab        -> still nothing: a short dig about the silence
    self_talk  -> it gives up on you and mutters to itself

Talking to it properly ("yeah" doesn't count) resets the ladder. Say "shh" (or "be quiet") to
stop the check-ins until you talk to it again. There's an hourly cap either way.

Chattiness (0..1, the orb page has a fader for it) scales all of that: 0 is off, 0.5 uses the
waits below as they are, 1 speaks up 4x sooner. It also sets how often the model is offered
PASS: at 0.6, 40% of nudges let it stay quiet; at 1 it always has to say something.

Sleep: after SLEEP_AFTER_MIN minutes with no word from you it signs off, asks whether to end the
session or stay on standby, and the brain unloads the model. Talking (or typing) wakes it.
"""

import random
import re
import time
from collections import deque

# ---------------------------------------------------------------- settings
STAGES = ["check_in", "give_space", "jab", "self_talk"]
WAIT = {"check_in": 60, "give_space": 120, "jab": 180, "self_talk": 300}   # seconds at chattiness 0.5
JITTER = 0.3              # each wait is randomly 30% shorter or longer
CHATTINESS = 0.6          # default 0..1 (0 = never speaks up on its own)
MAX_PER_HOUR = 8          # at chattiness 0.5; scales with it
SHH_MINUTES = 30          # "shh" lasts until you talk to it again, or this long
SLEEP_AFTER_MIN = 20      # minutes with no word from you before it goes to sleep (0 = never)
SLEEP_NOTE = ("You're going to sleep now to save power. Sign off in one short line, and ask "
              "whether they want to end the session or leave you on standby to wake by talking.")
STAGE_NOTES = {
    "check_in": "Speak up on your own and check in with the user.",
    "give_space": "Your last check-in got a brush-off or no answer. Say one brief line that "
                  "backs off and gives them space.",
    "jab": "They've ignored you a few times now. Make a short jab about the silence.",
    "self_talk": "They're clearly not talking. Mutter something to yourself, not to them.",
}
PASS_OFFER = " Or, if you'd rather stay quiet this time, reply with just PASS."

# replies that answer without engaging
BRUSH_OFF_WORDS = {
    "yeah", "yea", "yep", "yup", "yes", "ya", "ok", "okay", "k", "sure", "fine", "cool", "right",
    "mhm", "mm", "mmm", "hmm", "hm", "uh", "huh", "um", "no", "nope", "nah", "whatever", "maybe",
    "i", "guess", "dunno", "idk", "not", "now", "later", "busy", "good", "alright", "eh", "meh",
}
SHH_WORDS = re.compile(r"(sh+|shush|hush|quiet)( (sh+|shush|hush))*")        # the whole utterance
SHH_PHRASES = ("be quiet", "stop talking", "quiet mode", "shut up", "zip it", "give it a rest")


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def is_brush_off(text):
    """'yeah', 'mm', 'not now', 'I guess'... short answers made only of filler."""
    w = words(text)
    return len(w) <= 4 and all(x in BRUSH_OFF_WORDS for x in w)


def is_shh(text, name=""):
    """'Shh.', 'Rook, be quiet', 'hush' - short requests for silence."""
    w = [x for x in words(text) if x not in (name.lower(), "please", "just", "okay", "ok", "now")]
    joined = " ".join(w)
    return 0 < len(w) <= 5 and (bool(SHH_WORDS.fullmatch(joined))
                                or any(p in joined for p in SHH_PHRASES))


END_PHRASES = ("end session", "end the session", "end it", "shut down", "shut it down", "log off",
               "turn off", "turn it off", "power off", "power down", "close it", "sign off")
END_WORDS = {"end", "stop", "bye", "goodbye", "quit", "done", "off"}         # only as the whole answer
STANDBY_PHRASES = ("standby", "stand by", "stay on", "leave it on", "leave the mic on", "keep it on",
                   "keep listening", "quick wake", "leave me on", "leave you on", "stay up")
STANDBY_WORDS = {"stay", "sleep", "listen", "wait"}
FILLER = {"please", "just", "yeah", "yes", "ok", "okay", "the", "you", "can", "go", "ahead", "now",
          "sure", "alright", "for"}


def _answer(text, phrases, single, name=""):
    w = [x for x in words(text) if x not in FILLER and x != name.lower()]
    joined = " ".join(w)
    if len(w) == 1 and w[0] in single:
        return True
    return 0 < len(w) <= 5 and any(re.search(rf"\b{re.escape(p)}\b", joined) for p in phrases)


def is_end(text, name=""):
    """'end the session', 'shut down', 'bye' - its question before going to sleep."""
    return _answer(text, END_PHRASES, END_WORDS, name)


def is_standby(text, name=""):
    """'standby', 'leave it on', 'stay' - keep sleeping with the mic on."""
    return _answer(text, STANDBY_PHRASES, STANDBY_WORDS, name) and not is_end(text, name)


def say_duration(secs):
    if secs < 90:
        return f"about {int(round(secs / 10.0) * 10)} seconds"
    mins = round(secs / 60)
    return "about a minute" if mins == 1 else f"about {mins} minutes"


class Initiative:
    """Decides when the character speaks up. The brain calls:
       due(now)              -> a prompt to send the LLM, or None
       passed(now)           -> the LLM answered PASS
       on_spoke(now)         -> the character finished saying something (any reason)
       on_user(text, now)    -> you said something; returns "shh", "shh_off" or None
    """

    def __init__(self, name="", rng=random.random, now=None, chattiness=CHATTINESS):
        self.name = name
        self.rng = rng
        self.chattiness = chattiness
        self.offered_pass = False  # whether the last nudge allowed PASS
        self.last_grade = None     # "brush-off" / "engaged" for your last reply to a nudge
        self.level = 0             # index into STAGES
        self.pending = False       # an initiative is waiting for your reply
        self.fired = deque()       # when it spoke up, for the hourly cap
        self.shh_until = 0.0
        self.last = now if now is not None else time.time()   # last time anyone spoke
        self.last_user = self.last # last time you said something (sleep counts from here)
        self.sleep_after_min = SLEEP_AFTER_MIN
        self.asleep = False
        self.next_at = 0.0
        self.schedule(self.last)

    @property
    def stage(self):
        return STAGES[self.level]

    def next_stage(self):
        """What the next nudge will be (one step up if the last one is still unanswered)."""
        return STAGES[min(self.level + 1, len(STAGES) - 1)] if self.pending else self.stage

    def quiet(self, now):
        return now < self.shh_until

    def scale(self):
        """Wait multiplier: 4x at chattiness 0, 1x at 0.5, 1/4 at 1."""
        return 4 ** (1 - 2 * self.chattiness)

    def cap(self):
        return max(2, min(40, round(MAX_PER_HOUR / self.scale())))

    def schedule(self, now):
        self.anchor = now
        self.jit = 1 + JITTER * (2 * self.rng() - 1)
        self._place()

    def _place(self):
        self.next_at = self.anchor + WAIT[self.next_stage()] * self.scale() * self.jit

    def set_chattiness(self, value):
        """The fader moved: re-time the current wait without restarting it."""
        self.chattiness = max(0.0, min(1.0, float(value)))
        self._place()

    def off(self, now):
        return self.chattiness <= 0 or self.quiet(now) or self.asleep

    # ---- sleep
    def should_sleep(self, now):
        return (not self.asleep and self.sleep_after_min > 0
                and now - self.last_user >= self.sleep_after_min * 60)

    def sleep_prompt(self, now):
        return f"(Nobody has said anything for {say_duration(now - self.last_user)}. {SLEEP_NOTE})"

    def sleep(self):
        self.asleep, self.pending = True, False

    def wake(self, now):
        self.asleep, self.level, self.pending = False, 0, False
        self.last = self.last_user = now
        self.schedule(now)

    def due(self, now):
        if self.off(now) or now < self.next_at:
            return None
        while self.fired and now - self.fired[0] > 3600:
            self.fired.popleft()
        if len(self.fired) >= self.cap():
            self.next_at = self.fired[0] + 3600          # try again once the oldest one ages out
            return None
        if self.pending:                                  # the last one got no answer at all
            self.level = min(self.level + 1, len(STAGES) - 1)
        self.pending = True
        self.fired.append(now)
        self.offered_pass = self.rng() >= self.chattiness
        return (f"(Nobody has said anything for {say_duration(now - self.last)}. "
                f"{STAGE_NOTES[self.stage]}{PASS_OFFER if self.offered_pass else ''})")

    def passed(self, now):
        """It chose silence: no escalation, and it doesn't count toward the cap."""
        if self.fired:
            self.fired.pop()
        self.pending = False
        self.schedule(now)

    def on_spoke(self, now):
        self.last = now
        self.schedule(now)

    def on_user(self, text, now):
        self.last = self.last_user = now
        if is_shh(text, self.name):
            self.shh_until = now + SHH_MINUTES * 60
            self.pending = False
            return "shh"
        was_quiet = self.quiet(now)
        self.shh_until = 0.0
        self.last_grade = None
        if self.pending and is_brush_off(text):
            self.level = min(self.level + 1, len(STAGES) - 1)
            self.last_grade = "brush-off"
        else:
            if self.pending:
                self.last_grade = "engaged"
            self.level = 0                                # a real answer, or you started talking
        self.pending = False
        self.schedule(now)
        return "shh_off" if was_quiet else None
