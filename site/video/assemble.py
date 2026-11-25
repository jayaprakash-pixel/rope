"""Cut the recorded scenes to the narration: crossfades, narration per scene, a soft music bed, captions (WebVTT).
Output: site/public/demo.mp4 (H.264, faststart), demo-poster.webp, demo.vtt."""
import json, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "build", "demo")
PUB = os.path.join(ROOT, "site", "public")
sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, sys_path)
from narrate import SCENES                                   # noqa: E402

cuts = json.load(open(os.path.join(OUT, "cuts.json")))
durs = json.load(open(os.path.join(OUT, "durations.json")))
clip_of = {"hero": "hero", "plan": "app", "make": "app", "laptops": "laptops", "mine": "mine", "end": "end"}
XF, LEAD = 0.5, 0.35                                         # crossfade, narration starts this long after its scene

names = [n for n, _ in SCENES]
def clip_len(c):                                             # real length: decode (the webm header is not reliable)
    r = subprocess.run(["ffmpeg", "-i", os.path.join(OUT, f"clip_{c}.webm"), "-f", "null", "-"], capture_output=True, text=True).stderr
    h, m, s = r.rsplit("time=", 1)[1].split()[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

seg = []
for n in names:
    c = clip_of[n]; s, L = cuts[c][n]
    if s < 0: s = clip_len(c) + s - 0.2                      # cut from the end of the clip
    seg.append((c, s, L))
starts, t = [], 0.0
for i, (_, _, L) in enumerate(seg):
    starts.append(t)
    t += L - (XF if i < len(seg) - 1 else 0)
total = t

inputs, vf = [], []
for i, (c, s, L) in enumerate(seg):
    inputs += ["-i", os.path.join(OUT, f"clip_{c}.webm")]
    vf.append(f"[{i}:v]trim=start={s:.3f}:duration={L:.3f},setpts=PTS-STARTPTS,fps=30,scale=1920:1080:flags=lanczos,format=yuv420p,settb=AVTB[v{i}]")
acc = "v0"
for i in range(1, len(seg)):
    off = starts[i]
    vf.append(f"[{acc}][v{i}]xfade=transition=fade:duration={XF}:offset={off:.3f}[x{i}]")
    acc = f"x{i}"
vf.append(f"[{acc}]fade=t=in:st=0:d=0.6,fade=t=out:st={total - 1.0:.3f}:d=1.0[vout]")

na = len(seg)
for i, n in enumerate(names):
    inputs += ["-i", os.path.join(OUT, [f for f in os.listdir(OUT) if f.startswith(f"vo_{i}_")][0])]
    ms = int((starts[i] + LEAD) * 1000)
    vf.append(f"[{na + i}:a]aresample=44100,adelay={ms}|{ms},apad[a{i}]")
# music bed: a soft A-minor-9 pad that breathes slowly, far under the voice
pad = ("aevalsrc='0.20*sin(2*PI*110*t)+0.14*sin(2*PI*164.81*t)+0.11*sin(2*PI*261.63*t)+0.09*sin(2*PI*392*t)+0.06*sin(2*PI*493.88*t)"
       f"*(0.6+0.4*sin(2*PI*0.08*t))':s=44100:d={total + 0.5:.2f}")
vf.append(f"{pad},lowpass=f=1400,aecho=0.8:0.7:120|240:0.3|0.2,volume=0.13,afade=t=in:st=0:d=2,afade=t=out:st={total - 2.5:.2f}:d=2.5,aformat=channel_layouts=stereo[bed]")
vf.append("".join(f"[a{i}]" for i in range(len(names))) + f"[bed]amix=inputs={len(names) + 1}:normalize=0:duration=longest,atrim=0:{total:.3f},loudnorm=I=-16:TP=-1.5:LRA=11,aresample=44100[aout]")

mp4 = os.path.join(PUB, "demo.mp4")
cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(vf), "-map", "[vout]", "-map", "[aout]",
       "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-profile:v", "high", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", mp4]
subprocess.run(cmd, check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2.2", "-i", mp4, "-frames:v", "1", "-vf", "scale=1280:-1", "-q:v", "82", os.path.join(PUB, "demo-poster.webp")], check=True)

def ts(x): return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{x % 60:06.3f}"
with open(os.path.join(PUB, "demo.vtt"), "w", encoding="utf-8") as f:
    f.write("WEBVTT\n\n")
    for i, (n, text) in enumerate(SCENES):
        a = starts[i] + LEAD
        f.write(f"{i + 1}\n{ts(a)} --> {ts(a + durs[n])}\n{text.replace('rope dash A I dot vercel dot app', 'rope-ai.vercel.app')}\n\n")
size = os.path.getsize(mp4) / 1e6
print(f"demo.mp4: {total:.1f} s, {size:.1f} MB; scenes start at", [round(s, 1) for s in starts])
