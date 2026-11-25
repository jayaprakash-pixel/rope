import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion, useMotionValue, useSpring, useTransform } from 'motion/react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import {
  ArrowDown, Check, ChevronDown, CircleDashed, Clock, Cpu, Download, Eye, FileBox, Layers, Lock, Maximize2,
  MonitorDown, Play, Plug, ShieldCheck, Sparkles, Wand2,
} from 'lucide-react'
import Nav, { Footer } from './components/Nav'
import plans from './data/plans.json'
import { SHOWCASE } from './data/showcase'
import { DOWNLOAD, SETUP_MB, SETUP_SHA256, VERSION, reducedMotion } from './site'

gsap.registerPlugin(ScrollTrigger)

const fmt = (s: number) => (s < 60 ? `${s} s` : s < 3600 ? `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ''}` : `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`)

/* ------------------------------------------------------------------ hero: the product, plainly */
const rise = (d: number) => ({ initial: { opacity: 0, y: 18 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] as const, delay: d } })

function Hero() {
  return (
    <section className="hero" aria-labelledby="hero-title">
      <div className="wrap hero-center">
        <motion.p className="hero-kicker" {...rise(0.05)}>Rope {VERSION} · free · offline</motion.p>
        <motion.h1 id="hero-title" {...rise(0.12)}>Big AI image models.<br /><span className="grad-text">Everyday laptops.</span></motion.h1>
        <motion.p className="hero-lede" {...rise(0.2)}>Rope reads your laptop, picks the model that fits, and tells you how long a picture will take — before it starts.</motion.p>
        <motion.div className="hero-actions" {...rise(0.28)}>
          <a className="btn btn-primary pill" href={DOWNLOAD}><MonitorDown size={18} /> Download for Windows</a>
          <a className="text-link" href="#demo"><Play size={15} /> Watch the film <span aria-hidden>›</span></a>
        </motion.div>
        <motion.p className="hero-note" {...rise(0.34)}>Windows 10 and 11 · {SETUP_MB} MB · works without a graphics card</motion.p>
      </div>
      <motion.figure className="device" initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1], delay: 0.35 }}>
        <div className="device-screen">
          <img src="/product.webp" width={2880} height={1800} alt="The Rope app on a laptop: a picture of a lighthouse at dusk, made in 16 seconds" fetchPriority="high" />
        </div>
        <div className="device-base" aria-hidden><i /></div>
        <figcaption>Made on a 2021 laptop with 8 GB of RAM, in 16 seconds.</figcaption>
      </motion.figure>
      <div className="wrap hero-facts">
        <div><b>17</b><span>image models,<br />chosen for your laptop</span></div>
        <div><b>4 GB</b><span>of memory is enough<br />to start</span></div>
        <div><b>0</b><span>pictures or prompts<br />leave the laptop</span></div>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ showcase */
function Showcase() {
  if (!SHOWCASE.length) return null
  const row = [...SHOWCASE, ...SHOWCASE]
  return (
    <section className="showcase" aria-label="Pictures made with Rope">
      <div className="marquee">
        {row.map((s, i) => (
          <figure className="shot" key={i} aria-hidden={i >= SHOWCASE.length}>
            <img src={`/showcase/${s.file}`} alt={i < SHOWCASE.length ? s.prompt : ''} loading="lazy" width={512} height={512} />
            <figcaption><b>{s.model}</b>{s.secs} s · laptop with a 4 GB GTX 1650</figcaption>
          </figure>
        ))}
      </div>
      <p className="showcase-note">Every picture above was made by Rope on one ordinary gaming laptop from 2021. No cloud.</p>
    </section>
  )
}

/* ------------------------------------------------------------------ demo video */
function Demo() {
  return (
    <section id="demo" style={{ paddingTop: 40 }}>
      <div className="wrap">
        <div className="kicker reveal">See it work</div>
        <h2 className="reveal">One minute. <span className="grad-text">One real laptop.</span></h2>
        <p className="sub reveal">Recorded on the 2021 laptop Rope is built on: the plan, a picture forming step by step, the planner for other laptops, and a big model file of your own. Nothing is sped up.</p>
        <div className="demo-frame reveal">
          <video controls playsInline preload="metadata" poster="/demo-poster.webp" width={1920} height={1080} aria-label="Rope demo video, 1 minute, with narration and captions">
            <source src="/demo.mp4" type="video/mp4" />
            <track kind="captions" src="/demo.vtt" srcLang="en" label="English" />
          </video>
        </div>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ numbers */
function Count({ to, suffix = '', decimals = 0 }: { to: number; suffix?: string; decimals?: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (reducedMotion()) { el.textContent = to.toFixed(decimals) + suffix; return }
    const o = { v: 0 }
    const tw = gsap.to(o, { v: to, duration: 1.6, ease: 'power3.out', scrollTrigger: { trigger: el, start: 'top 88%', once: true },
      onUpdate: () => { el.textContent = o.v.toFixed(decimals) + suffix } })
    return () => { tw.scrollTrigger?.kill(); tw.kill() }
  }, [to, suffix, decimals])
  return <span ref={ref}>0{suffix}</span>
}

function Problem() {
  return (
    <section id="why">
      <div className="glow" style={{ background: '#7c3aed', left: '-10%', top: '10%' }} />
      <div className="wrap">
        <div className="kicker reveal">The problem</div>
        <h2 className="reveal">AI image tools assume a big graphics card. <span className="grad-text">Most laptops don't have one.</span></h2>
        <p className="sub reveal">They crash with "out of memory", or run for an hour without telling you. Rope measures the laptop first, then plans the run so it fits: the right model, size, steps and decoder — and an honest time.</p>
        <div className="numbers">
          {[
            { v: <Count to={17} />, l: 'image models Rope knows, from 2 GB to 12 GB files' },
            { v: <Count to={4} suffix=" GB" />, l: 'of RAM is enough to make a first picture' },
            { v: <Count to={12} suffix=" B" />, l: 'parameters: FLUX.1-schnell runs on an 8 GB laptop' },
            { v: <Count to={0} suffix=" bytes" />, l: 'of your prompts or pictures leave the laptop' },
          ].map((n, i) => (
            <div className="card reveal" key={i}><div className="num grad-text">{n.v}</div><div className="num-label">{n.l}</div></div>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ how it works */
const STEPS = [
  { t: 'Read the laptop', d: 'Rope measures the processor and its vector units, the RAM that is really free, the graphics card, the disk speed and the power source. Nothing leaves the laptop.',
    term: <><span className="c">$ rope probe</span>{'\n'}<span className="k">cpu    </span> AMD Ryzen 5 5600H · 6 cores / 12 threads · <span className="g">avx2</span>{'\n'}<span className="k">memory </span> 8.0 GiB installed · 7.3 usable · 2.1 free{'\n'}<span className="k">gpu    </span> NVIDIA GeForce GTX 1650 · 4 GB · CUDA 13.2{'\n'}<span className="k">disk   </span> NVMe SSD · 3.3 GB/s read{'\n'}<span className="k">power  </span> on mains</> },
  { t: 'Plan the run', d: 'The planner checks every model against this laptop: does it fit in memory, which decoder, which size, how many steps — and how many seconds. It refuses what cannot fit, and says why.',
    term: <><span className="c">$ rope plan</span>{'\n'}plan (balanced) on this device (417 GFLOPS, 7.3 GiB RAM){'\n'}<span className="k">  model </span> sd-turbo — SD-Turbo (SD 2.1 distilled){'\n'}<span className="k">  image </span> 512x512, 1 step, full VAE{'\n'}<span className="k">  memory</span> predicted peak <span className="y">1.44 GiB</span>{'\n'}<span className="k">  time  </span> <span className="p">~8 s</span> with the GTX 1650{'\n'}<span className="k">  note  </span> weights read from disk while it runs</> },
  { t: 'Make the picture', d: 'The engine is written from scratch in Rust and C, with CUDA for NVIDIA cards. Big files are streamed from disk or shrunk once. You watch a live preview, and you can pause it or change the prompt halfway.',
    term: <><span className="c">denoise</span>  step 1/4  <span className="g">██████░░░░░░░░░░░░</span>  preview{'\n'}<span className="c">denoise</span>  step 2/4  <span className="g">████████████░░░░░░</span>  preview{'\n'}<span className="y">› prompt</span> "…at sunset" <span className="c">(taken at the next step)</span>{'\n'}<span className="c">denoise</span>  step 4/4  <span className="g">██████████████████</span>{'\n'}<span className="c">decode </span>  tiles 4/4</> },
  { t: 'Keep it', d: 'The picture goes to your gallery on this laptop, with the prompt, the seed and the time. Rope then compares the time with its own plan, so the next plan is more honest.',
    term: <><span className="g">✓</span> image 512x512 in <span className="p">8.1 s</span> <span className="c">(plan said 7.0 s)</span>{'\n'}<span className="g">✓</span> saved to the gallery{'\n'}{'\n'}<span className="c">memory: peak 1.09 GiB · nothing sent anywhere</span></> },
]

function How() {
  const [active, setActive] = useState(0)
  const steps = useRef<(HTMLDivElement | null)[]>([])
  useLayoutEffect(() => {
    const triggers = steps.current.map((el, i) => el && ScrollTrigger.create({ trigger: el, start: 'top 60%', end: 'bottom 40%', onToggle: (s) => s.isActive && setActive(i) }))
    return () => triggers.forEach((t) => t?.kill())
  }, [])
  return (
    <section id="how">
      <div className="wrap">
        <div className="kicker reveal">How it works</div>
        <h2 className="reveal">Measure first. <span className="grad-text">Then run.</span></h2>
        <p className="sub reveal">One engine adapts to any laptop: from a 4 GB laptop with no graphics card to a gaming laptop. The same four steps every time.</p>
        <div className="how">
          <div className="how-steps">
            {STEPS.map((s, i) => (
              <div key={s.t} ref={(el) => { steps.current[i] = el }} className={`how-step ${active === i ? 'active' : ''}`}>
                <div className="n">0{i + 1}</div>
                <h3>{s.t}</h3>
                <p>{s.d}</p>
              </div>
            ))}
          </div>
          <div className="how-panel">
            <div className="term" role="img" aria-label={`Example output: ${STEPS[active].t}`}>
              <div className="term-bar"><i /><i /><i /><span>rope · step {active + 1} of 4 · example output</span></div>
              <AnimatePresence mode="wait">
                <motion.pre key={active} initial={{ opacity: 0, y: 12, filter: 'blur(6px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }} exit={{ opacity: 0, y: -12, filter: 'blur(6px)' }} transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}>
                  {STEPS[active].term}
                </motion.pre>
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ planner */
type Plan = { key: string; label: string; sub: string; ram: number; gpu: string | null; models: { id: string; title: string; runs: boolean; tested: boolean; s?: number; size?: number; steps?: number; gpu?: boolean; dl?: number; why?: string }[] }

function Planner() {
  const all = plans as Plan[]
  const [key, setKey] = useState('8gb')
  const p = all.find((x) => x.key === key)!
  const rows = useMemo(() => [...p.models].filter((m) => m.runs).sort((a, b) => (a.s ?? 0) - (b.s ?? 0)), [p])
  const max = Math.log(Math.max(...all.flatMap((x) => x.models.map((m) => m.s ?? 1))))
  return (
    <section id="planner">
      <div className="glow" style={{ background: '#db2777', right: '-12%', top: '30%' }} />
      <div className="wrap">
        <div className="kicker reveal">Will it run on my laptop?</div>
        <h2 className="reveal">Pick a laptop. <span className="grad-text">See the plan.</span></h2>
        <p className="sub reveal">These are the real answers of Rope's planner for four example laptops: every model, its picture size and steps, and the time for one picture. On your laptop, Rope makes this list for you.</p>
        <div className="planner">
          <div className="profiles" role="group" aria-label="Example laptops">
            {all.map((x) => (
              <button key={x.key} className="profile" aria-pressed={x.key === key} onClick={() => setKey(x.key)}>
                <b>{x.label}</b><span>{x.sub}</span>
              </button>
            ))}
          </div>
          <div>
            <motion.div className="plan-list" layout>
              <AnimatePresence initial={false}>
                {rows.map((m) => (
                  <motion.div layout key={m.id} className="plan-row" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} transition={{ type: 'spring', stiffness: 260, damping: 30 }}>
                    <div>
                      <div className="t">{m.title.replace(/ \(.*\)$/, '')}{m.gpu && <span className="tag gpu">GPU</span>}{!m.tested && <span className="tag">untested</span>}</div>
                      <div className="m">{m.size}×{m.size} · {m.steps} step{m.steps === 1 ? '' : 's'}{m.dl ? ` · ${m.dl} GB download` : ''}</div>
                    </div>
                    <div className="bar" aria-hidden><motion.i initial={false} animate={{ width: `${Math.max(4, (Math.log(Math.max(m.s ?? 1, 2)) / max) * 100)}%` }} transition={{ type: 'spring', stiffness: 120, damping: 22 }} /></div>
                    <div className="secs">~{fmt(m.s ?? 0)}</div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </motion.div>
            <p className="planner-note">Times for one picture at the planner's "balanced" quality, model already downloaded. "Untested": the model has the same architecture as a tested one and was checked from its file header, but has not been run on Rope yet. Bars use a log scale.</p>
          </div>
        </div>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ features */
function Tilt({ children, className }: { children: React.ReactNode; className: string }) {
  const x = useMotionValue(0), y = useMotionValue(0)
  const rx = useSpring(useTransform(y, [-0.5, 0.5], [6, -6]), { stiffness: 200, damping: 20 })
  const ry = useSpring(useTransform(x, [-0.5, 0.5], [-6, 6]), { stiffness: 200, damping: 20 })
  return (
    <motion.div className={className} style={{ rotateX: rx, rotateY: ry, transformPerspective: 900 }}
      onPointerMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); x.set((e.clientX - r.left) / r.width - 0.5); y.set((e.clientY - r.top) / r.height - 0.5) }}
      onPointerLeave={() => { x.set(0); y.set(0) }}>
      {children}
    </motion.div>
  )
}

const FEATURES = [
  { i: <Cpu size={22} />, t: 'One engine, any laptop', d: 'Its own engine in Rust and C: AVX2 and SSE4.1 kernels for any 64-bit processor, CUDA for NVIDIA cards. It picks the path at run time.', wide: true },
  { i: <Clock size={22} />, t: 'Honest time, before you start', d: 'Every plan says how long a picture takes. On real laptops the plan was within 25% on 7 of 9 runs.' },
  { i: <Eye size={22} />, t: 'Live preview', d: 'Watch the picture appear step by step, on SD, SDXL and FLUX.' },
  { i: <Wand2 size={22} />, t: 'Steer while it runs', d: 'Pause, resume, or change the prompt halfway. It takes effect at the next step.' },
  { i: <FileBox size={22} />, t: 'Bring your own model', d: 'Paste the path of a model file you already have, even a big one. Rope checks it and finds a way: stream it, shrink it once, or use a smaller size.', wide: true },
  { i: <Layers size={22} />, t: 'LoRA styles', d: 'Add a LoRA style file to an SD or SDXL model, with a strength slider.' },
  { i: <Maximize2 size={22} />, t: '4× upscaler', d: 'Real-ESRGAN in tiles: a 512 picture becomes 2048 inside the same memory limit.' },
  { i: <Lock size={22} />, t: 'Private by design', d: 'Runs offline after the one model download. No account, no telemetry. Model files are checked (SHA-256 and a safety scan) before first use.' },
  { i: <Plug size={22} />, t: 'For developers', d: 'A documented local HTTP API and a C library (librope) with Python and C examples.' },
]

function Features() {
  return (
    <section id="features">
      <div className="wrap">
        <div className="kicker reveal">Features</div>
        <h2 className="reveal">Small laptop. <span className="grad-text">Full studio.</span></h2>
        <div className="features">
          {FEATURES.map((f) => (
            <Tilt key={f.t} className={`card feature reveal ${f.wide ? 'wide' : ''}`}>
              <div className="ic">{f.i}</div><h3>{f.t}</h3><p>{f.d}</p>
            </Tilt>
          ))}
        </div>
        <motion.img className="app-shot reveal" src="/app.webp" alt="The Rope app: the Create screen with the plan for this laptop" loading="lazy" width={1440} height={900} />
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ status (honest matrix) */
const MATRIX = [
  { h: 'Built and tested', c: 'var(--ok)', icon: <Check size={16} color="var(--ok)" />, items: [
    'Image models: SD 1.5, SD 2.x / SD-Turbo, SDXL / SDXL-Turbo, FLUX.1-schnell', 'CPU (AVX2, SSE4.1) and NVIDIA CUDA', 'Planner: model, size, steps, decoder, memory and time',
    'Live preview, pause, prompt change while running', 'LoRA, 4× upscaler, your own model files', 'Browser app, HTTP API, C library', 'Windows installer (per user, no admin rights)'] },
  { h: 'Works, with limits', c: 'var(--warn)', icon: <CircleDashed size={16} color="var(--warn)" />, items: [
    '13 of the 17 catalog models are checked from their file header, not run yet', 'GPU time estimates are 30–40% too optimistic on RTX 30-series laptops',
    'Linux: a command-line test program, processor only', 'Installer is not code-signed: Windows shows a warning first', 'Models download from the installer or the command line, not from the app'] },
  { h: 'On the roadmap', c: 'var(--violet-2)', icon: <Clock size={16} color="var(--violet-2)" />, items: [
    'Video models (Wan 2.x)', 'Text models (LLMs)', 'macOS (Apple silicon)', 'Code-signed installer', 'Download button in the app', 'ControlNet'] },
]

function Status() {
  return (
    <section id="status">
      <div className="wrap">
        <div className="kicker reveal">Status</div>
        <h2 className="reveal">What works today. <span className="grad-text">Honestly.</span></h2>
        <p className="sub reveal">Version {VERSION}. Every "built" line below has a test that ran. Everything else is said as it is.</p>
        <div className="matrix">
          {MATRIX.map((m) => (
            <div className="card reveal" key={m.h}>
              <h3><span className="dot" style={{ background: m.c }} />{m.h}</h3>
              <ul>{m.items.map((t) => <li key={t}>{m.icon}<span>{t}</span></li>)}</ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ architecture */
function Arch() {
  const ref = useRef<SVGSVGElement>(null)
  useEffect(() => {
    const svg = ref.current
    if (!svg || reducedMotion()) return
    const lines = svg.querySelectorAll<SVGPathElement>('.flow')
    lines.forEach((l) => { const n = l.getTotalLength(); l.style.strokeDasharray = `${n}`; l.style.strokeDashoffset = `${n}` })
    const tl = gsap.timeline({ scrollTrigger: { trigger: svg, start: 'top 75%', once: true } })
    tl.from(svg.querySelectorAll('.box'), { opacity: 0, y: 16, stagger: 0.08, duration: 0.6, ease: 'power3.out' })
      .to(lines, { strokeDashoffset: 0, duration: 0.9, stagger: 0.1, ease: 'power2.inOut' }, '-=0.3')
    return () => { tl.scrollTrigger?.kill(); tl.kill() }
  }, [])
  const box = (x: number, y: number, w: number, t: string, s: string, accent = false) => (
    <g className="box" transform={`translate(${x} ${y})`}>
      <rect width={w} height="74" rx="14" fill={accent ? 'url(#ag)' : 'var(--surface-solid)'} stroke="var(--border-strong)" />
      <text x="18" y="31" fontSize="16" fontWeight="650" fill={accent ? '#fff' : undefined}>{t}</text>
      <text x="18" y="54" className="small" fill={accent ? 'rgba(255,255,255,0.8)' : undefined}>{s}</text>
    </g>
  )
  return (
    <section id="architecture">
      <div className="wrap">
        <div className="kicker reveal">Architecture</div>
        <h2 className="reveal">Written from scratch. <span className="grad-text">No Python, no PyTorch.</span></h2>
        <p className="sub reveal">One small program (about 2 MB) holds the app server, the planner and the engine. Each picture runs in its own process, so memory goes back to Windows when it ends.</p>
        <div className="card arch reveal">
          <svg ref={ref} viewBox="0 0 1100 360" role="img" aria-label="Architecture: the browser app talks to rope serve; the planner uses the probe and the catalog; the engine runs the kernels on the CPU or the NVIDIA GPU">
            <defs>
              <linearGradient id="ag" x1="0" x2="1"><stop offset="0" stopColor="#7c3aed" /><stop offset="1" stopColor="#db2777" /></linearGradient>
              <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#a78bfa" /></marker>
            </defs>
            <g fill="none" stroke="#a78bfa" strokeWidth="1.6" markerEnd="url(#ah)">
              <path className="flow" d="M230 77 H300" />
              <path className="flow" d="M520 77 H590" />
              <path className="flow" d="M810 77 H870" />
              <path className="flow" d="M700 114 V200" />
              <path className="flow" d="M410 270 H585" />
              <path className="flow" d="M980 114 V200" />
              <path className="flow" d="M700 274 V300 H960 V274" />
            </g>
            {box(20, 40, 210, 'App in the browser', 'React · gallery · previews')}
            {box(300, 40, 220, 'rope serve', 'local HTTP API · one job')}
            {box(590, 40, 220, 'Planner', 'fit · time · decoder · size', true)}
            {box(870, 40, 210, 'Engine (Rust)', 'SD · SDXL · FLUX · LoRA')}
            {box(170, 236, 240, 'Probe (C)', 'CPU · RAM · GPU · disk')}
            {box(590, 200, 220, 'Catalog', '17 models · sha256 · licences')}
            {box(870, 200, 210, 'Kernels (C / CUDA)', 'AVX2 · SSE4.1 · CUDA')}
          </svg>
        </div>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ install */
function Install() {
  const [copied, setCopied] = useState(false)
  return (
    <section id="install">
      <div className="glow" style={{ background: '#7c3aed', left: '30%', top: '20%' }} />
      <div className="wrap">
        <div className="kicker reveal">Install</div>
        <h2 className="reveal">Two minutes to your <span className="grad-text">first picture.</span></h2>
        <div className="install">
          <div className="steps">
            {[
              ['Download Rope-Setup.exe', `${SETUP_MB} MB. The setup holds the program and the app; the model comes later, chosen for your laptop.`],
              ['Open it. Windows may warn you first', 'Rope is not code-signed yet, so Windows shows "Windows protected your PC". Click "More info", then "Run anyway". It installs for your user only and needs no administrator rights.'],
              ['Let Rope choose the model', 'The setup reads the laptop, shows the model that fits and its download size, and asks before it downloads.'],
              ['Start "Rope" from the Start menu', 'It opens in your browser. Type what you want to see. To remove Rope: Settings → Apps → Rope → Uninstall.'],
            ].map(([t, d]) => (
              <div className="step reveal" key={t}><div><h3>{t}</h3><p>{d}</p></div></div>
            ))}
          </div>
          <div className="card req reveal">
            <h3>Requirements</h3>
            <dl>
              <dt>System</dt><dd>Windows 10 or 11, 64-bit</dd>
              <dt>Memory</dt><dd>4 GB RAM to start · 8 GB recommended</dd>
              <dt>Disk</dt><dd>3 to 15 GB free, by model</dd>
              <dt>Graphics</dt><dd>Not needed · NVIDIA GTX 10-series or newer is used if present</dd>
              <dt>Internet</dt><dd>Once, for the model download</dd>
            </dl>
            <div className="cta-row" style={{ marginTop: 26 }}>
              <a className="btn btn-primary" href={DOWNLOAD}><Download size={18} /> Download v{VERSION}</a>
            </div>
            {!SETUP_SHA256.startsWith('__') && (
              <button className="sha mono" style={{ background: 'none', border: 0, cursor: 'pointer', textAlign: 'left', padding: 0 }}
                onClick={() => { navigator.clipboard?.writeText(SETUP_SHA256).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) }, () => {}) }}
                aria-label="Copy the SHA-256 fingerprint of Rope-Setup.exe">
                <ShieldCheck size={15} style={{ flex: 'none' }} /> SHA-256 {SETUP_SHA256} {copied ? '· copied' : ''}
              </button>
            )}
            <div className="warn-box"><b>Mac and Linux:</b> not yet. A Linux test program exists in the source; a Mac version is on the roadmap.</div>
          </div>
        </div>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ faq */
const FAQ = [
  ['Is Rope free?', 'Yes. Rope is open source under the MIT licence. The AI models are made by others and have their own licences; the app shows the licence of each model, and some limit how you may use the pictures.'],
  ['Does it need the internet?', 'Only once, to download the model the planner chose. After that Rope works offline. It never sends your prompts or pictures anywhere.'],
  ['My laptop has no graphics card. Will it work?', 'Yes. Rope was built for that. A picture with SD-Turbo takes about 20–35 seconds on a normal 8 GB laptop with only the processor. Bigger models take minutes, and the plan tells you how many before you start.'],
  ['Why does Windows warn me?', 'Code signing costs money, and Rope is not signed yet. Windows warns about every unsigned program it has not seen often. You can check the file: its SHA-256 fingerprint is on this page and on the GitHub release. Company-managed laptops may block unsigned programs completely.'],
  ['Can I use a model I already have?', 'Yes, if it is a Stable Diffusion 1.x, 2.x, SDXL or FLUX.1-schnell file in .safetensors or .gguf. Paste its path in the app under "Your own model files". Rope says if it can run it and how. It does not open .ckpt files, because they can contain hidden code.'],
  ['How good are the time estimates?', 'On 9 runs on real laptops, the plan was within 25% on 7. Laptops with very little free RAM were slower than planned; the planner now adds time for that. GPU estimates for RTX 30-series laptops are still too optimistic.'],
]

function Faq() {
  const [open, setOpen] = useState<number | null>(0)
  return (
    <section id="faq">
      <div className="wrap">
        <div className="kicker reveal">Questions</div>
        <h2 className="reveal">Good to know.</h2>
        <div className="faq">
          {FAQ.map(([q, a], i) => (
            <div key={q} className="reveal">
              <button aria-expanded={open === i} onClick={() => setOpen(open === i ? null : i)}>
                {q}<motion.span animate={{ rotate: open === i ? 180 : 0 }}><ChevronDown size={20} /></motion.span>
              </button>
              <AnimatePresence initial={false}>
                {open === i && (
                  <motion.div className="a" initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}>
                    <p>{a}</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function Final() {
  return (
    <section className="final">
      <div className="glow" style={{ background: '#c026d3', left: '35%', top: '10%' }} />
      <div className="wrap">
        <Sparkles size={34} color="var(--violet-2)" style={{ margin: '0 auto 18px' }} />
        <h2 className="reveal">Your laptop is <span className="grad-text">enough.</span></h2>
        <p className="sub reveal">Download Rope, let it read your laptop, and make your first picture in minutes.</p>
        <div className="cta-row reveal">
          <a className="btn btn-primary" href={DOWNLOAD}><MonitorDown size={19} /> Download for Windows</a>
          <a className="btn" href="/docs.html">Read the documentation <ArrowDown size={16} style={{ rotate: '-90deg' }} /></a>
        </div>
      </div>
    </section>
  )
}

export default function Home() {
  useEffect(() => {
    if (reducedMotion()) return
    const ctx = gsap.context(() => {
      gsap.set('.reveal', { opacity: 0, y: 36 })
      ScrollTrigger.batch('.reveal', {
        start: 'top 88%', once: true,
        onEnter: (els) => gsap.to(els, { opacity: 1, y: 0, duration: 0.9, stagger: 0.08, ease: 'power3.out', overwrite: true }),
      })
    })
    // pictures and fonts change the layout: measure again once they are in
    const refresh = () => ScrollTrigger.refresh()
    window.addEventListener('load', refresh)
    return () => { ctx.revert(); window.removeEventListener('load', refresh) }
  }, [])
  return (
    <>
      <Nav />
      <main id="main">
        <Hero />
        <Showcase />
        <Demo />
        <Problem />
        <How />
        <Planner />
        <Features />
        <Status />
        <Arch />
        <Install />
        <Faq />
        <Final />
      </main>
      <Footer />
    </>
  )
}
