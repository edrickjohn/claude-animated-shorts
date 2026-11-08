"""Turn an ElevenLabs voice-over into everything the picture and the sound need, in one step.

    tools/voice/.venv/Scripts/python.exe tools/voice/prepare_voice.py topics/<slug> [voice/<file>]

The user generates the dub in ElevenLabs and drops the file (mp3/wav/...) into topics/<slug>/voice/. With no file
argument the newest audio file in voice/ other than voice.wav is used. This script:
  1. converts it to voice/voice.wav (48 kHz mono, what audio/sound.py reads),
  2. aligns the words of SCRIPT.md ("## Script" section) to the audio with faster-whisper -> voice/words_whisper.json,
  3. snaps every word that follows a pause to the real speech onset in the waveform, applies the topic's hand fixes
     from voice/overrides.json ({"<word index>": [start, end]}), widens slivers -> voice/words.json,
  4. writes video/words.ts, so every cut in video/story.ts retimes itself.
Then re-run audio/sound.py and render. Words whisper never heard are interpolated and listed: check them by ear.
"""
import sys, json, re, pathlib, difflib, subprocess, shutil
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
FF = r"C:\Users\edric\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
if not pathlib.Path(FF).exists(): FF = shutil.which("ffmpeg") or "ffmpeg"
relp = lambda p: p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else p.as_posix()
AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

topic = (REPO / sys.argv[1]).resolve() if len(sys.argv) > 1 else sys.exit(__doc__)
V = topic / "voice"
if len(sys.argv) > 2: src = (topic / sys.argv[2]).resolve()
else:
    cands = [p for p in V.iterdir() if p.suffix.lower() in AUDIO and p.name not in ("voice.wav",)]
    if not cands: sys.exit(f"no ElevenLabs file in {V} (drop the mp3 there)")
    src = max(cands, key=lambda p: p.stat().st_mtime)

# ---- 1. convert to voice/voice.wav, 48 kHz mono ----
wav = V / "voice.wav"
if src != wav:
    subprocess.run([FF, "-loglevel", "error", "-y", "-i", str(src), "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True)
print("voice:", src.name, "->", relp(wav))
pcm = subprocess.run([FF, "-loglevel", "error", "-i", str(wav), "-ar", "16000", "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout
audio = np.frombuffer(pcm, dtype=np.float32)
print(f"length {len(audio) / 16000:.2f} s")

# ---- 2. whisper alignment against the script words ----
txt = (topic / "SCRIPT.md").read_text(encoding="utf-8").split("## Script", 1)[1]
script = re.findall(r"[A-Za-z0-9']+", txt.replace("\u2019", "'"))
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
from faster_whisper import WhisperModel
m = WhisperModel("small.en", device="cpu", compute_type="int8")
segs, _ = m.transcribe(audio, word_timestamps=True, language="en", vad_filter=False)
hyp = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]
print("heard:", " ".join(h[0] for h in hyp))
H = [norm(h[0]) for h in hyp]; S = [norm(s) for s in script]
times = [None] * len(script)
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, S, H, autojunk=False).get_opcodes():
    if tag == "equal":
        for k in range(i2 - i1): times[i1 + k] = (hyp[j1 + k][1], hyp[j1 + k][2])
    elif tag == "replace" and i2 - i1 == 1:  # one script word = several whisper pieces
        times[i1] = (hyp[j1][1], hyp[j2 - 1][2])
    elif tag == "replace" and i2 - i1 == j2 - j1:  # misheard words, same count: map 1:1
        for k in range(i2 - i1): times[i1 + k] = (hyp[j1 + k][1], hyp[j1 + k][2])
    elif tag == "replace" and j2 - j1 == 1:  # several script words in one whisper token ("25" = twenty five)
        t0, t1 = hyp[j1][1], hyp[j1][2]; n = i2 - i1; d = (t1 - t0) / n
        for k in range(n): times[i1 + k] = (round(t0 + k * d, 3), round(t0 + (k + 1) * d, 3))
missed = [i for i, t in enumerate(times) if t is None]
for i in missed:  # interpolate inside the gap between matched neighbours
    a = i - 1
    while a >= 0 and times[a] is None: a -= 1
    b = i + 1
    while b < len(times) and times[b] is None: b += 1
    t0 = times[a][1] if a >= 0 else 0.0
    t1 = times[b][0] if b < len(times) else t0 + 0.4 * (b - a)
    n = b - a - 1; step = (t1 - t0) / n if n else 0
    k = i - a - 1
    times[i] = (round(t0 + k * step, 3), round(t0 + (k + 1) * step, 3))
w = [{"word": s, "start": round(t[0], 3), "end": round(t[1], 3)} for s, t in zip(script, times)]
(V / "words_whisper.json").write_text(json.dumps(w, indent=0), encoding="utf-8")
print(len(script), "script words,", len(hyp), "heard; interpolated (check by ear):", [(i, script[i]) for i in missed])

# ---- 3. snap words after a pause to the speech onset, hand fixes, slivers ----
toks = re.findall(r"[A-Za-z0-9'\u2019]+[^\sA-Za-z0-9'\u2019]*", txt)
after_pause = [i == 0 or bool(re.search(r"[.,:;!?]$", toks[i - 1])) for i in range(len(toks))]
assert len(toks) == len(w), (len(toks), len(w))
hop = 160  # 10 ms
rms = np.sqrt(np.convolve(audio ** 2, np.ones(hop) / hop, "same"))[::hop]
db = 20 * np.log10(rms + 1e-9)
speech = db > (db.max() - 40)
onsets = [i / 100 for i in range(1, len(speech)) if speech[i] and not speech[max(0, i - 12):i].any()]  # after >=120 ms silence
fixed = []
for i, x in enumerate(w):
    if not after_pause[i]: continue
    lo = w[i - 1]["end"] - 0.02 if i else 0.0
    cand = [o for o in onsets if lo <= o <= x["end"] + 0.05]
    if not cand: continue
    o = round(cand[0], 3)
    if abs(o - x["start"]) > 0.03:
        x["start"] = o
        if x["end"] - x["start"] < 0.12:  # squeezed: borrow from the next word's start
            nxt = w[i + 1]["start"] if i + 1 < len(w) else o + 0.3
            x["end"] = round(min(max(x["end"], o + 0.15), nxt), 3)
        fixed.append(i)
    if x["end"] - x["start"] > 0.9 and i + 1 < len(w):  # interpolated word spanning a pause
        x["end"] = round(min(w[i + 1]["start"], x["start"] + 0.2), 3)
ov = V / "overrides.json"
manual = {int(k): v for k, v in json.loads(ov.read_text(encoding="utf-8")).items() if not k.startswith("_")} if ov.exists() else {}
for i, (s, e) in manual.items(): w[i]["start"], w[i]["end"] = s, e
for i, x in enumerate(w):  # any remaining sliver: extend into the silence after it (min 80 ms)
    nxt = w[i + 1]["start"] if i + 1 < len(w) else x["end"] + 0.1
    if x["end"] - x["start"] < 0.08: x["end"] = round(min(x["start"] + 0.08, nxt), 3)
(V / "words.json").write_text(json.dumps(w, indent=0), encoding="utf-8")
print(len(fixed), "onsets snapped,", len(manual), "overrides; still short:", [(i, x["word"]) for i, x in enumerate(w) if x["end"] - x["start"] < 0.06])

# ---- 4. video/words.ts ----
rel = relp(topic)
with open(topic / "video" / "words.ts", "w", encoding="utf-8") as f:
    f.write(f"// generated by tools/voice/prepare_voice.py from {rel}/voice/words.json; do not edit by hand\n")
    f.write("export const WORDS: [string, number, number][] = [\n")
    for i, x in enumerate(w): f.write(f'  ["{x["word"]}", {x["start"]}, {x["end"]}], // {i}\n')
    f.write("];\n")
print(f"{len(w)} words, last ends {w[-1]['end']} s -> {rel}/video/words.ts. Next: py {rel}/audio/sound.py, then render.")
