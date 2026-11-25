"""Narration for the demo video: one ElevenLabs clip per scene. The key is read from <repo>/elevenlabs/.env
(git-ignored) and never printed. Output: build/demo/vo_<n>.mp3 and durations.json."""
import json, os, subprocess, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "build", "demo")
os.makedirs(OUT, exist_ok=True)
VOICE = "JBFqnCBsd6RMkjVDRZzb"   # George: warm, captivating storyteller

SCENES = [
    ("hero", "This is Rope. It makes AI pictures on the laptop you already have. No cloud, no account, and no big graphics card."),
    ("plan", "Type what you want to see. Before anything runs, Rope reads your laptop and plans the picture: the model, the size, and how long it will take."),
    ("make", "Press generate, and watch the picture form, step by step. You can pause it, or change the prompt halfway."),
    ("laptops", "Every laptop gets its own plan. An old four gigabyte laptop, or a gaming laptop: Rope shows what runs there, and how fast, before you start."),
    ("mine", "Already have a model file, even a big one? Paste its path. Rope checks it, and finds a way to run it."),
    ("end", "Rope is free, it works offline, and it installs in two minutes. Download it at rope dash A I dot vercel dot app."),
]

def key():
    t = open(os.path.join(ROOT, "elevenlabs", ".env"), encoding="utf-8").read().strip()
    return t.split("=", 1)[1].strip().strip('"') if "=" in t else t

def main():
    k = key()
    durs = {}
    for i, (name, text) in enumerate(SCENES):
        path = os.path.join(OUT, f"vo_{i}_{name}.mp3")
        if not os.path.exists(path) or "--again" in sys.argv:
            body = json.dumps({"text": text, "model_id": "eleven_multilingual_v2",
                               "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.25, "use_speaker_boost": True}}).encode()
            req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_44100_128", data=body,
                                         headers={"xi-api-key": k, "Content-Type": "application/json", "Accept": "audio/mpeg"})
            with urllib.request.urlopen(req, timeout=120) as r: open(path, "wb").write(r.read())
        d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip())
        durs[name] = round(d, 2)
        print(f"{name:8} {d:5.2f} s  {text[:60]}...")
    json.dump(durs, open(os.path.join(OUT, "durations.json"), "w"), indent=1)
    print("total", round(sum(durs.values()), 1), "s")

if __name__ == "__main__":
    main()
