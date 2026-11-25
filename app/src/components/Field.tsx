import { lazy, memo, Suspense, useCallback, useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import type { FieldMode } from './DiffusionField'

// three.js loads only when the stage is shown (the first screen stays small)
const DiffusionField = lazy(() => import('./DiffusionField'))

export type FieldProps = { progress: number; running: boolean; image: string | null; alt: string; onGpu: boolean; gpuDevice: boolean; previewUrl: string | null }

const webgl = (() => {
  try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')) } catch { return false }
})()
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches

/** The stage: the WebGL diffusion field while an image is made, then the image. A still, light fallback without WebGL
 *  or with reduced motion. While the engine runs on the processor the field draws a few frames per second only. */
function Field({ progress, running, image, alt, onGpu, gpuDevice, previewUrl }: FieldProps) {
  const [developed, setDeveloped] = useState(false)
  useEffect(() => { if (!image) setDeveloped(false) }, [image])
  const onDeveloped = useCallback(() => setDeveloped(true), [])
  const live = webgl && !reducedMotion()
  const mode: FieldMode = running ? (onGpu ? 'gpu' : 'cpu') : 'idle'
  // live preview: the engine's denoised estimate after a step. Once one exists it replaces the particle field; the final
  // image fades in over the last preview (kept until the next job starts)
  const [lastPreview, setLastPreview] = useState<string | null>(null)
  useEffect(() => { if (previewUrl) setLastPreview(previewUrl); else if (!running && !image) setLastPreview(null) }, [previewUrl, running, image])
  const showImage = !!image && (developed || !live || !!lastPreview)
  return (
    <div className="stage">
      {!live && !image && !lastPreview && (
        <div style={{ position: 'absolute', inset: 0, opacity: 0.35 + 0.65 * progress,
                      background: 'radial-gradient(60% 60% at 50% 50%, var(--glow), transparent 70%)' }} />
      )}
      {lastPreview && <img key="preview" className="preview" src={lastPreview} alt="" aria-hidden />}
      {live && !developed && !lastPreview && (
        <Suspense fallback={null}>
          <DiffusionField progress={progress} mode={mode} imageUrl={image} onDeveloped={onDeveloped} density={onGpu ? 128 : gpuDevice ? 96 : 64} />
        </Suspense>
      )}
      <AnimatePresence>
        {showImage && <motion.img key={image} src={image!} alt={alt} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }} />}
      </AnimatePresence>
      {running && <span className="sr-only">Generating</span>}
    </div>
  )
}

// memo: the page's 1 s clock re-renders the progress text, not the field (a still frame on CPU jobs stays still)
export default memo(Field)
