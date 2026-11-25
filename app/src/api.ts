// Client for `rope serve` (engine/rust/serve.rs). Same origin: in dev Vite proxies /api, in production rope serves the app.

export type Device = {
  os: string
  cpu: string
  cores: number
  threads: number
  cpu_path: 'avx2' | 'sse41'
  ram_usable_gib: number
  ram_free_gib: number
  gpu?: { name: string; vram_total_gib: number; vram_free_gib: number }
  disk?: { free_gib: number; ssd: boolean; bus: string; seq_read_gbs: number }
}

export type Model = {
  id: string
  kind: string
  title: string
  on_disk: boolean
  tested?: boolean   // false: same architecture as a tested model, checked from its header only
  license?: string
  runs: boolean
  size?: number
  steps?: number
  guidance?: number
  decoder?: string
  commit_gib?: number
  download_gib?: number
  estimate_s?: number
  upscale_s?: number
  peak_gib?: number
  vae_tile?: number
  max_size?: number
  max_estimate_s?: number
  max_native?: number
  sizes?: { size: number; estimate_s: number; peak_gib: number; vae_tile: number }[]
  limit?: string | null
  on_gpu?: boolean
  cpu_estimate_s?: number
  notes?: string[]
  why?: string
  file?: string      // the model file's name
  mine?: boolean      // a model file the user added by its path ("your own models")
  path?: string
  bytes?: number
  source?: 'as-is' | 'stream' | 'quantize'
}

export type Addon = { name: string; kind: 'lora' | 'esrgan'; title: string; bytes: number }
export type Models = { quality: string; device: { gflops: number; ram_gib: number; gpu: string | null }; models: Model[] }
export type Quality = 'fast' | 'balanced' | 'best'

export type JobEvent =
  | { type: 'start'; id: string; model: string; prompt: string; seed: number; image: string }
  | { type: 'estimate'; total_s: number }
  | { type: 'preview'; step: number; n: number }
  | { type: 'paused'; step: number }
  | { type: 'resumed'; step: number }
  | { type: 'prompt'; step: number }
  | { type: 'progress'; stage: 'text' | 'denoise' | 'decode' | string; i: number; n: number; elapsed_s: number; eta_s: number }
  | { type: 'log'; line: string }
  | { type: 'done'; id: string; model: string; prompt: string; seed: number; image: string; seconds: number }
  | { type: 'error'; message: string }

export type GalleryItem = { id: string; model: string; prompt: string; quality: string; seed: number; image: string; seconds: number; created_ms: number }

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error ?? `${r.status}`)
  return r.json() as Promise<T>
}

export const api = {
  device: () => get<Device>('/api/device'),
  models: (q: Quality, size?: number, only?: string, upscale?: boolean) => get<Models>(`/api/models?quality=${q}${size ? `&size=${size}` : ''}${only ? `&only=${only}` : ''}${upscale ? '&upscale=1' : ''}`),
  addons: () => get<{ addons: Addon[] }>('/api/addons'),
  mine: (q: Quality) => get<{ models: Model[] }>(`/api/mine?quality=${q}`),
  /** Add a model file by its full path on this laptop; returns rope's check of the file. */
  async addMine(path: string): Promise<Model> {
    const r = await fetch('/api/mine', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) })
    const j = await r.json()
    if (!r.ok) throw new Error(j.error ?? `${r.status}`)
    return j
  },
  async removeMine(id: string): Promise<void> {
    const r = await fetch('/api/mine/remove', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }) })
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error ?? `${r.status}`)
  },
  gallery: () => get<GalleryItem[]>('/api/gallery'),
  async start(body: { model: string; prompt: string; quality: Quality; seed?: number; size?: number; lora?: string; lora_scale?: number; upscale?: string }): Promise<{ id: string; seed: number }> {
    const r = await fetch('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    const j = await r.json()
    if (!r.ok) throw new Error(j.error ?? `${r.status}`)
    return j
  },
  cancel: (id: string) => fetch(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  /** Steer a running job; it takes the command at its next step boundary. */
  async control(id: string, body: { op: 'pause' | 'resume' } | { op: 'prompt'; text: string } | { op: 'cfg'; value: number }): Promise<void> {
    const r = await fetch(`/api/jobs/${id}/control`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error ?? `${r.status}`)
  },
  /** Every event of a job, from the first; the stream ends with `done` or `error`. Returns a close function. */
  events(id: string, on: (e: JobEvent) => void): () => void {
    const es = new EventSource(`/api/jobs/${id}/events`)
    es.onmessage = (m) => {
      const e = JSON.parse(m.data) as JobEvent
      on(e)
      if (e.type === 'done' || e.type === 'error') es.close()
    }
    es.onerror = () => es.close()
    return () => es.close()
  },
}

export const fmtSeconds = (s: number) => (s < 60 ? `${Math.max(0, Math.round(s))} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`)
