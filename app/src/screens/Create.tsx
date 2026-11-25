import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Download, Layers, Maximize2, Pause, Play, RotateCcw, Square, Wand2 } from 'lucide-react'
import { api, fmtSeconds, type Addon, type Device, type JobEvent, type Model, type Quality } from '../api'
import Field from '../components/Field'

const QUALITIES: [Quality, string][] = [['fast', 'Fast'], ['balanced', 'Balanced'], ['best', 'Best']]
const STAGE: Record<string, string> = { text: 'Reading the prompt', denoise: 'Shaping the image', decode: 'Developing the pixels', upscale: 'Upscaling ×4' }

type Run =
  | { state: 'idle' }
  | { state: 'running'; id: string; stage: string; i: number; n: number; elapsed: number; eta: number | null; estimate: number | null; preview: number; paused: boolean; upShare: number }
  | { state: 'done'; image: string; seconds: number; seed: number }
  | { state: 'error'; message: string }

export default function CreateScreen({ device }: { device: Device | null }) {
  const [quality, setQuality] = useState<Quality>('balanced')
  const [models, setModels] = useState<Model[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [model, setModel] = useState<string>('')
  const [size, setSize] = useState<number | null>(null)       // null: the model's own default size
  const [prompt, setPrompt] = useState('')
  const [run, setRun] = useState<Run>({ state: 'idle' })
  const [announce, setAnnounce] = useState('')
  const close = useRef<() => void>(() => {})
  const [steerText, setSteerText] = useState('')
  const [steerNote, setSteerNote] = useState('')
  const [cfg, setCfg] = useState(7.5)
  const [addons, setAddons] = useState<Addon[]>([])
  const [lora, setLora] = useState('')                          // file name in models/addons, '' = none
  const [loraScale, setLoraScale] = useState(1)
  const [upscale, setUpscale] = useState(false)
  const [mine, setMine] = useState<Model[]>([])                // model files the user added by path
  const [minePath, setMinePath] = useState('')
  const [mineBusy, setMineBusy] = useState(false)
  const [mineError, setMineError] = useState('')

  useEffect(() => {
    setModels(null)
    api.models(quality).then((m) => {
      const ready = m.models.filter((x) => x.kind === 'image')
      setModels(ready)
      setModel((cur) => (ready.some((x) => x.id === cur && x.runs && x.on_disk) ? cur : ready.find((x) => x.runs && x.on_disk)?.id ?? ''))
    }, (e) => setLoadError(String(e.message ?? e)))
  }, [quality])
  useEffect(() => { api.mine(quality).then((m) => setMine(m.models), () => {}) }, [quality])
  async function addMine() {
    const p = minePath.trim()
    if (!p) return
    setMineBusy(true); setMineError('')
    try {
      const m = await api.addMine(p)
      setMine((cur) => [...cur.filter((x) => x.id !== m.id), m])
      setMinePath('')
      if (m.runs) setModel(m.id)
    } catch (e) { setMineError(String((e as Error).message ?? e)) } finally { setMineBusy(false) }
  }
  async function removeMine(id: string) {
    try { await api.removeMine(id); setMine((cur) => cur.filter((x) => x.id !== id)); if (model === id) setModel('') } catch (e) { setMineError(String((e as Error).message ?? e)) }
  }
  useEffect(() => () => close.current(), [])
  useEffect(() => { api.addons().then((a) => setAddons(a.addons), () => {}) }, [])
  useEffect(() => { setSize(null) }, [model, quality])
  useEffect(() => {
    if (!model || model.startsWith('mine-')) return          // a user's file: its plan comes from /api/mine
    let live = true
    api.models(quality, size ?? undefined, model, upscale).then((m) => {
      if (!live) return
      const fresh = m.models.find((x) => x.id === model)
      // with the upscaler the sizes, ceiling and times change too (its memory and time are in the plan): take all of it
      if (fresh) setModels((cur) => cur && cur.map((x) => (x.id === model ? (upscale ? fresh : { ...fresh, sizes: x.sizes, max_size: x.max_size, max_estimate_s: x.max_estimate_s, limit: x.limit, max_native: x.max_native }) : x)))
    }, () => {})
    return () => { live = false }
  }, [size, model, quality, upscale])

  const chosen = useMemo(() => models?.find((m) => m.id === model) ?? mine.find((m) => m.id === model), [models, mine, model])
  const loras = addons.filter((a) => a.kind === 'lora')
  const upscaler = addons.find((a) => a.kind === 'esrgan')
  // a LoRA fits one model family: its file says so ("for SDXL UNet" / "for SD 1.x / 2.x UNet"); FLUX has no LoRA support yet
  const loraWhy = (a: Addon): string | null => {
    const sdxl = model.includes('sdxl'), flux = model.includes('flux')
    if (flux) return 'not supported for FLUX yet'
    if (a.title.includes('SDXL') && !sdxl) return 'made for SDXL models'
    if (a.title.includes('SD 1.x') && sdxl) return 'made for SD 1.x / 2.x models'
    return null
  }
  const loraFile = loras.find((a) => a.name === lora)
  useEffect(() => { if (loraFile && loraWhy(loraFile)) setLora('') }, [model])   // eslint-disable-line react-hooks/exhaustive-deps
  const canRun = !!chosen && chosen.runs && chosen.on_disk && prompt.trim().length > 0 && run.state !== 'running'

  async function generate() {
    if (!canRun || !chosen) return
    try {
      const { id, seed } = await api.start({ model: chosen.id, prompt: prompt.trim(), quality, ...(size ? { size } : {}),
        ...(loraFile && !loraWhy(loraFile) ? { lora: loraFile.name, lora_scale: loraScale } : {}), ...(upscale && upscaler ? { upscale: upscaler.name } : {}) })
      setRun({ state: 'running', id, stage: 'text', i: 0, n: 1, elapsed: 0, eta: null, estimate: chosen.estimate_s ?? null, preview: 0, paused: false,
                       upShare: upscale && chosen.upscale_s && chosen.estimate_s ? Math.min(0.9, chosen.upscale_s / chosen.estimate_s) : 0 })
      setAnnounce('Started')
      close.current = api.events(id, (e: JobEvent) => {
        if (e.type === 'paused') {
          setRun((r) => (r.state === 'running' ? { ...r, paused: true, i: e.step, stage: 'denoise', elapsed: r.elapsed + (Date.now() - at.current) / 1000 } : r))   // freeze the clock where it stands
          setAnnounce(`Paused before step ${e.step + 1}`)
        }
        if (e.type === 'resumed') { setRun((r) => (r.state === 'running' ? { ...r, paused: false } : r)); setAnnounce('Resumed') }
        if (e.type === 'prompt') { setSteerNote(`Prompt changed before step ${e.step + 1}`); setAnnounce(`Prompt changed before step ${e.step + 1}`) }
        if (e.type === 'preview') setRun((r) => (r.state === 'running' ? { ...r, preview: e.step } : r))
        if (e.type === 'estimate') setRun((r) => (r.state === 'running' ? { ...r, estimate: e.total_s, eta: e.total_s } : r))
        if (e.type === 'progress') {
          setRun((r) => {
            if (r.state !== 'running') return r
            if (r.stage !== e.stage) setAnnounce(STAGE[e.stage] ?? e.stage)
            return { ...r, stage: e.stage, i: e.i, n: e.n, elapsed: e.elapsed_s, eta: e.eta_s }
          })
        }
        if (e.type === 'done') { setRun({ state: 'done', image: e.image, seconds: e.seconds, seed }); setAnnounce(`Image ready in ${fmtSeconds(e.seconds)}`) }
        if (e.type === 'error') { setRun({ state: 'error', message: e.message }); setAnnounce('Generation failed') }
      })
    } catch (err) {
      setRun({ state: 'error', message: String((err as Error).message ?? err) })
    }
  }

  // between server events, time moves on: elapsed and ETA advance locally from the last event (1 s ticks)
  const [now, setNow] = useState(() => Date.now())
  const at = useRef(Date.now())
  useEffect(() => { at.current = Date.now(); setNow(at.current) }, [run])
  useEffect(() => {
    if (run.state !== 'running') return
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [run.state])
  const frozen = run.state === 'running' && run.paused              // a pause is not work: the clock stands still
  const since = frozen ? 0 : (now - at.current) / 1000
  const elapsed = run.state === 'running' ? run.elapsed + since : 0
  const eta = run.state === 'running' && run.eta !== null ? Math.max(0, run.eta - since) : null

  async function steer(body: Parameters<typeof api.control>[1]) {
    if (run.state !== 'running') return
    try { await api.control(run.id, body); if (body.op === 'prompt') { setPrompt(body.text); setSteerText(''); setSteerNote('Sent: it takes effect at the next step') } }
    catch (err) { setSteerNote(`Not sent: ${String((err as Error).message ?? err)}`) }
  }

  // progress over the whole job: text 5%, denoising 80%, decoding 15%. Events set the floor; the time estimate
  // (elapsed / (elapsed + ETA)) moves it smoothly inside a long step, never past the next milestone.
  // with the upscaler the first three stages share (1 - its share of the plan's time) and the upscale stage gets the rest
  const pre = run.state === 'running' ? 1 - run.upShare : 1
  const step = run.state !== 'running' ? (run.state === 'done' ? 1 : 0)
    : run.stage === 'text' ? pre * 0.05 * (run.i / run.n)
    : run.stage === 'denoise' ? pre * (0.05 + 0.8 * (run.i / run.n))
    : run.stage === 'upscale' ? pre + (1 - pre) * (run.i / run.n)
    : pre * (0.85 + 0.15 * (run.i / run.n))
  const next = run.state !== 'running' ? step
    : run.stage === 'text' ? pre * 0.05 : run.stage === 'denoise' ? pre * (0.05 + 0.8 * Math.min(1, (run.i + 1) / run.n))
    : run.stage === 'upscale' ? pre + (1 - pre) * Math.min(1, (run.i + 1) / run.n) : pre
  const timeFrac = eta !== null && elapsed + eta > 0 ? elapsed / (elapsed + eta) : 0
  const frac = run.state === 'running' ? Math.max(step, Math.min(timeFrac, next - 0.01)) : step

  if (run.state === 'running' || run.state === 'done') {
    return (
      <div className="stack">
        <Field progress={chosen?.on_gpu ? frac : step} running={run.state === 'running'} image={run.state === 'done' ? run.image : null} alt={prompt}
               onGpu={!!chosen?.on_gpu} gpuDevice={!!device?.gpu}
               previewUrl={run.state === 'running' && run.preview > 0 ? `/api/jobs/${run.id}/preview?step=${run.preview}` : null} />
        <div className="progress-wrap">
          {run.state === 'running' ? (
            <>
              <div className="progress-row">
                <strong>{run.paused ? `Paused before step ${Math.min(run.i + 1, run.n)} of ${run.n}` : `${STAGE[run.stage] ?? run.stage}${run.stage === 'denoise' ? ` · step ${Math.min(run.i + 1, run.n)} of ${run.n}` : ''}`}</strong>
                <span className="mono muted">{fmtSeconds(elapsed)} {eta !== null ? `· about ${fmtSeconds(eta)} left` : ''}</span>
              </div>
              <div className="bar" role="progressbar" aria-label="Generation progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(frac * 100)}>
                <span style={{ transform: `scaleX(${frac})` }} />
              </div>
              <div className="row">
                {(run.stage === 'text' || run.stage === 'denoise') && (
                  <button className="btn" onClick={() => steer({ op: run.paused ? 'resume' : 'pause' })} aria-pressed={run.paused}>
                    {run.paused ? <Play size={16} aria-hidden /> : <Pause size={16} aria-hidden />}{run.paused ? 'Resume' : 'Pause'}
                  </button>
                )}
                <button className="btn" onClick={() => { api.cancel(run.id); close.current(); setRun({ state: 'idle' }) }}><Square size={16} aria-hidden />Cancel</button>
              </div>
              {(run.stage === 'text' || run.stage === 'denoise') && (
                <div className="field steer">
                  <label htmlFor="steer" className="label">Change the prompt</label>
                  <div className="row">
                    <input id="steer" className="input" value={steerText} maxLength={2000} placeholder={prompt} onChange={(e) => setSteerText(e.target.value)}
                           onKeyDown={(e) => { if (e.key === 'Enter' && steerText.trim()) steer({ op: 'prompt', text: steerText.trim() }) }} aria-describedby="steer-hint" />
                    <button className="btn" disabled={!steerText.trim()} onClick={() => steer({ op: 'prompt', text: steerText.trim() })}>Apply</button>
                  </div>
                  {(chosen?.guidance ?? 1) > 1 && (
                    <label className="row hint">Guidance {cfg.toFixed(1)}
                      <input type="range" min={1} max={15} step={0.5} value={cfg} onChange={(e) => setCfg(Number(e.target.value))} onMouseUp={() => steer({ op: 'cfg', value: cfg })}
                             onKeyUp={() => steer({ op: 'cfg', value: cfg })} aria-label="Guidance" />
                    </label>
                  )}
                  <span id="steer-hint" className="hint">It takes effect at the next step. The earlier you change it, the more it changes the image. {steerNote}</span>
                </div>
              )}
            </>
          ) : (
            <motion.div className="row" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
              <span className="badge ok mono">{fmtSeconds(run.seconds)}</span>
              <span className="muted mono">seed {run.seed}</span>
              <span className="spacer" />
              <a className="btn" href={run.image} download><Download size={16} aria-hidden />Save</a>
              <button className="btn primary" onClick={() => setRun({ state: 'idle' })}><RotateCcw size={16} aria-hidden />New image</button>
            </motion.div>
          )}
        </div>
        <p className="sr-only" aria-live="polite">{announce}</p>
      </div>
    )
  }

  return (
    <div className="stack">
      <div className="hero">
        <span className="label">Create</span>
        <h1>Describe an image</h1>
        <p className="muted">Rope plans the run for this laptop before it starts, and tells you how long it will take.</p>
      </div>
      <div className="create">
        <div className="stack">
          <div className="field">
            <label htmlFor="prompt" className="label">Prompt</label>
            <textarea id="prompt" className="prompt" value={prompt} maxLength={2000} placeholder="A red bicycle leaning on a brick wall, photo"
                      onChange={(e) => setPrompt(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate() }} aria-describedby="prompt-hint" />
            <span id="prompt-hint" className="hint">Ctrl + Enter to generate · {prompt.length}/2000</span>
          </div>
          <div className="field">
            <span className="label" id="quality-label">Quality</span>
            <div className="segmented" role="radiogroup" aria-labelledby="quality-label">
              {QUALITIES.map(([q, label]) => (
                <button key={q} role="radio" aria-checked={quality === q} onClick={() => setQuality(q)}>{label}</button>
              ))}
            </div>
          </div>
          <div className="field">
            <span className="label" id="model-label">Model</span>
            {loadError && <div className="card error" role="alert">{loadError}</div>}
            {!models && !loadError && <p className="muted" role="status">Planning models for this laptop…</p>}
            <div className="models" role="radiogroup" aria-labelledby="model-label">
              {models?.map((m) => {
                const ok = m.runs && m.on_disk
                return (
                  <button key={m.id} className="model" role="radio" aria-checked={model === m.id} disabled={!ok} onClick={() => setModel(m.id)}
                          aria-describedby={`why-${m.id}`}>
                    <span className="radio" aria-hidden />
                    <span>
                      <strong>{m.title}</strong>
                      <span id={`why-${m.id}`} className="hint" style={{ display: 'block' }}>
                        {m.runs ? `${m.size}×${m.size} · ${m.steps} step${m.steps === 1 ? '' : 's'}${(m.guidance ?? 1) > 1 ? ' · guidance' : ''}${m.on_disk ? '' : ` · not downloaded (${(m.download_gib ?? 0).toFixed(1)} GB)`}` : m.why}
                        {m.tested === false && ' · not tested on Rope yet'}
                        {m.license && ` · licence ${m.license}`}
                      </span>
                    </span>
                    {m.runs ? <span className={`badge mono ${(m.estimate_s ?? 0) < 60 ? 'ok' : 'warn'}`}>~{fmtSeconds(m.estimate_s ?? 0)}</span>
                            : <span className="badge no">does not fit</span>}
                  </button>
                )
              })}
            </div>
          </div>
          <div className="field">
            <span className="label" id="mine-label">Your own model files</span>
            <p className="hint">Have a model file already (a big one too)? Paste its full path. Rope reads the file where it is, checks if this laptop can run it, and finds a way: streaming from disk, a smaller copy made once, or a smaller picture size.</p>
            <div className="row">
              <input className="input mono" aria-labelledby="mine-label" placeholder="D:\models\my-model.safetensors" value={minePath}
                     onChange={(e) => setMinePath(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') addMine() }} />
              <button className="btn" onClick={addMine} disabled={mineBusy || !minePath.trim()}>{mineBusy ? 'Checking…' : 'Add'}</button>
            </div>
            {mineError && <div className="card error" role="alert">{mineError}</div>}
            {mine.length > 0 && (
              <div className="models" role="radiogroup" aria-labelledby="mine-label">
                {mine.map((m) => (
                  <div key={m.id} className="mine-row">
                    <button className="model" role="radio" aria-checked={model === m.id} disabled={!m.runs} onClick={() => setModel(m.id)} aria-describedby={`why-${m.id}`}>
                      <span className="radio" aria-hidden />
                      <span>
                        <strong>{m.file}</strong>
                        <span id={`why-${m.id}`} className="hint" style={{ display: 'block' }}>
                          {m.runs ? `${m.title} · ${m.size}×${m.size} · ${m.steps} step${m.steps === 1 ? '' : 's'}${m.source === 'quantize' ? ' · a smaller copy is made at the first picture' : m.source === 'stream' ? ' · read from disk while it runs' : ''}${(m.download_gib ?? 0) > 0.01 ? ` · needs ${(m.download_gib ?? 0).toFixed(1)} GB of helper files` : ''}` : m.why}
                        </span>
                      </span>
                      {m.runs ? <span className={`badge mono ${(m.estimate_s ?? 0) < 60 ? 'ok' : 'warn'}`}>~{fmtSeconds(m.estimate_s ?? 0)}</span>
                              : <span className="badge no">cannot run</span>}
                    </button>
                    <button className="btn ghost" onClick={() => removeMine(m.id)} aria-label={`Remove ${m.file} from the list (the file is not deleted)`}>Remove</button>
                  </div>
                ))}
              </div>
            )}
          </div>
          {chosen?.runs && chosen.sizes && chosen.sizes.length > 1 && (
            <div className="field">
              <span className="label" id="size-label">Size</span>
              <div className="segmented" role="radiogroup" aria-labelledby="size-label">
                {chosen.sizes.map((o) => (
                  <button key={o.size} role="radio" aria-checked={(size ?? chosen.size) === o.size} onClick={() => setSize(o.size)} title={`about ${fmtSeconds(o.estimate_s)} on the processor`}>{o.size}</button>
                ))}
              </div>
            </div>
            )}
          {(loras.length > 0 || upscaler) && (
            <div className="field addons">
              <span className="label" id="addons-label"><Layers size={14} aria-hidden /> Add-ons</span>
              {loras.length > 0 && (
                <div className="stack tight">
                  <label className="row" htmlFor="lora">LoRA style
                    <select id="lora" className="input" value={lora} onChange={(e) => setLora(e.target.value)}>
                      <option value="">None</option>
                      {loras.map((a) => { const why = loraWhy(a); return <option key={a.name} value={a.name} disabled={!!why}>{a.name.replace(/\.safetensors$/, '')}{why ? ` (${why})` : ''}</option> })}
                    </select>
                  </label>
                  {loraFile && !loraWhy(loraFile) && (
                    <label className="row hint">Strength {loraScale.toFixed(2)}
                      <input type="range" min={0} max={1.5} step={0.05} value={loraScale} onChange={(e) => setLoraScale(Number(e.target.value))} aria-label="LoRA strength" />
                    </label>
                  )}
                </div>
              )}
              {upscaler && (
                <label className="row check">
                  <input type="checkbox" checked={upscale} onChange={(e) => setUpscale(e.target.checked)} />
                  <Maximize2 size={14} aria-hidden /> Upscale ×4{chosen?.runs && chosen.size ? ` (to ${chosen.size * 4}×${chosen.size * 4})` : ''}
                </label>
              )}
            </div>
          )}
        </div>
        <aside className="card plan" aria-label="Plan">
          <span className="label">Plan on this laptop</span>
          <AnimatePresence mode="wait">
            <motion.div key={chosen?.id ?? 'none'} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }} className="stack">
              {chosen && chosen.runs ? (
                <>
                  <div><div className="plan-time">~{fmtSeconds(chosen.estimate_s ?? 0)}</div><p className="muted">estimated, {chosen.on_gpu && device?.gpu ? `with the ${device.gpu.name}` : 'on the processor'}</p></div>
                  <dl className="kv">
                    <dt>Size</dt><dd>{chosen.size}×{chosen.size}</dd>
                    {chosen.peak_gib !== undefined && (<><dt>Memory (peak)</dt><dd>{chosen.peak_gib.toFixed(2)} GiB</dd></>)}
                    {upscale && chosen.size && (<><dt>Output</dt><dd>{chosen.size * 4}×{chosen.size * 4} (upscaled)</dd></>)}
                    {upscale && chosen.upscale_s ? (<><dt>Upscale</dt><dd>+{fmtSeconds(chosen.upscale_s)}</dd></>) : null}
                    <dt>Steps</dt><dd>{chosen.steps}</dd>
                    <dt>Decoder</dt><dd>{chosen.decoder === 'taesd' ? 'tiny (fast)' : 'full VAE'}</dd>
                  </dl>
                  {chosen.max_size && chosen.max_size > (chosen.size ?? 0) && (
                    <p className="hint">This laptop can go up to {chosen.max_size}×{chosen.max_size} (about {fmtSeconds(chosen.max_estimate_s ?? 0)} on the processor).</p>
                  )}
                  {chosen.limit && <p className="hint">{chosen.limit}</p>}
                  {chosen.notes && chosen.notes.length > 0 && <ul className="notes">{chosen.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
                </>
              ) : <p className="muted">Pick a model that fits this laptop.</p>}
            </motion.div>
          </AnimatePresence>
          <button className="btn primary" onClick={generate} disabled={!canRun}><Wand2 size={18} aria-hidden />Generate</button>
          {run.state === 'error' && <div className="card error" role="alert">{run.message}</div>}
        </aside>
      </div>
    </div>
  )
}
