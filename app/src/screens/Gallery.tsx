import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { X } from 'lucide-react'
import { api, fmtSeconds, type GalleryItem } from '../api'

export default function GalleryScreen() {
  const [items, setItems] = useState<GalleryItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<GalleryItem | null>(null)
  const dlg = useRef<HTMLDialogElement>(null)

  useEffect(() => { api.gallery().then(setItems, (e) => setError(String(e.message ?? e))) }, [])
  useEffect(() => { if (open) dlg.current?.showModal(); else dlg.current?.close() }, [open])

  if (error) return <div className="card error" role="alert">{error}</div>
  if (!items) return <p className="muted" role="status">Loading the gallery…</p>
  return (
    <div className="stack">
      <div className="hero">
        <span className="label">Gallery</span>
        <h1>Made on this laptop</h1>
        <p className="muted">{items.length === 0 ? 'Nothing yet. Your images stay on this machine.' : `${items.length} image${items.length === 1 ? '' : 's'}, stored on this machine only.`}</p>
      </div>
      <div className="gallery">
        {items.map((g, k) => (
          <motion.button key={g.id} className="tile" onClick={() => setOpen(g)} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                         transition={{ delay: Math.min(k, 12) * 0.04, duration: 0.3 }} aria-label={`Open image: ${g.prompt}`}>
            <img src={g.image} alt="" loading="lazy" width={512} height={512} />
            <span className="meta"><span className="clamp">{g.prompt}</span><span className="hint mono">{g.model} · {fmtSeconds(g.seconds)}</span></span>
          </motion.button>
        ))}
      </div>
      <dialog ref={dlg} className="lightbox" onClose={() => setOpen(null)} aria-label="Image">
        {open && (
          <div className="lightbox-body">
            <div className="row"><strong className="clamp" style={{ flex: 1 }}>{open.prompt}</strong>
              <button className="icon-btn" onClick={() => setOpen(null)} aria-label="Close" autoFocus><X size={18} aria-hidden /></button></div>
            <img src={open.image} alt={open.prompt} />
            <p className="hint mono">{open.model} · seed {open.seed} · {open.quality} · {fmtSeconds(open.seconds)}</p>
          </div>
        )}
      </dialog>
    </div>
  )
}
