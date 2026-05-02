import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
from groq import Groq
from src.config import GROQ_MODEL

load_dotenv()
MAX_RETRY_PER_KEY = 3   # số lần retry trên mỗi key trước khi chuyển key
WAIT_BASE = 20          # giây chờ cơ bản khi bị rate limit


def _load_keys() -> list[str]:
    keys = []
    for i in range(1, 20):
        env = "GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"
        k = os.getenv(env, "").strip()
        if k:
            keys.append(k)
    return keys


class LLMWrapper:
    def __init__(self):
        self._keys = _load_keys()
        if not self._keys:
            raise RuntimeError("Không tìm thấy GROQ_API_KEY nào trong .env")
        self._idx = 0
        self._clients = [Groq(api_key=k) for k in self._keys]
        print(f"  [LLM] Loaded {len(self._keys)} Groq key(s)", flush=True)

    def _current(self) -> Groq:
        return self._clients[self._idx]

    def _next_key(self):
        old = self._idx
        self._idx = (self._idx + 1) % len(self._keys)
        print(f"  [LLM] Switching key {old+1} → {self._idx+1}/{len(self._keys)}", flush=True)

    def generate(self, prompt: str) -> str:
        # Thử từng key, mỗi key MAX_RETRY_PER_KEY lần
        total_attempts = len(self._keys) * MAX_RETRY_PER_KEY
        for attempt in range(total_attempts):
            try:
                resp = self._current().chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                is_rate = ("429" in err or "rate" in err.lower()
                           or "quota" in err.lower() or "exhausted" in err.lower())
                if is_rate:
                    key_attempt = attempt % MAX_RETRY_PER_KEY
                    if key_attempt < MAX_RETRY_PER_KEY - 1:
                        # Còn retry trên key hiện tại
                        wait = WAIT_BASE * (key_attempt + 1)
                        print(f"  [LLM] Rate limit key {self._idx+1} "
                              f"(retry {key_attempt+1}/{MAX_RETRY_PER_KEY}), "
                              f"wait {wait}s", flush=True)
                        time.sleep(wait)
                    else:
                        # Hết retry trên key này → chuyển sang key tiếp theo
                        self._next_key()
                        time.sleep(5)
                else:
                    raise
        raise RuntimeError(f"LLM failed sau {total_attempts} attempts trên {len(self._keys)} keys")
