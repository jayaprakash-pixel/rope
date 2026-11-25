import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

// The diffusion field (P5.1.5): one particle per image cell. They start as noise (random place, random hue) and settle
// onto the grid as denoising progresses; when the image is ready each particle takes its pixel's color, so the field
// develops into the picture before the PNG fades in. Rendering is on demand at a rate chosen for the device.

export type FieldMode = 'idle' | 'gpu' | 'cpu'          // cpu: the engine uses the processor -> almost no rendering
// cpu: a still frame, redrawn only when progress moves by 10% (~10 frames per image). Measured on the CPU path
// (SD-Turbo, 6 interleaved rounds): a live canvas cost +2.8% image time even at 2 fps (the browser's GPU process and
// compositor stay busy), on top of +2.2% for the app page itself.
const FPS: Record<FieldMode, number> = { idle: 60, gpu: 30, cpu: 0 }

const vert = /* glsl */ `
  attribute vec3 aStart; attribute vec3 aNoiseColor; attribute vec3 aColor; attribute float aSeed;
  uniform float uProgress; uniform float uDevelop; uniform float uTime; uniform float uSize;
  varying vec3 vColor; varying float vAlpha;
  void main() {
    float p = smoothstep(0.0, 1.0, uProgress);
    float wob = (1.0 - p) * 0.035;
    vec3 jitter = vec3(sin(uTime * 1.3 + aSeed * 40.0), cos(uTime * 1.1 + aSeed * 31.0), 0.0) * wob;
    vec3 pos = mix(aStart, position, p) + jitter;
    vColor = mix(aNoiseColor, aColor, uDevelop);
    vAlpha = mix(0.55, 1.0, max(p, uDevelop));
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * mix(1.6, 1.0, p);                // orthographic: no depth scaling
  }`
const frag = /* glsl */ `
  varying vec3 vColor; varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.5, 0.2, d));
  }`

function Particles({ n, progress, imageUrl, onDeveloped, still }: { n: number; progress: number; imageUrl: string | null; onDeveloped: () => void; still: boolean }) {
  const mat = useRef<THREE.ShaderMaterial>(null)
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry()
    const count = n * n
    const target = new Float32Array(count * 3), start = new Float32Array(count * 3), noise = new Float32Array(count * 3)
    const color = new Float32Array(count * 3), seed = new Float32Array(count)
    const c = new THREE.Color()
    for (let y = 0, k = 0; y < n; y++) for (let x = 0; x < n; x++, k++) {
      target.set([(x + 0.5) / n * 2 - 1, 1 - (y + 0.5) / n * 2, 0], k * 3)
      // noise: a gaussian cloud around the centre (Box-Muller), hues from violet to pink
      const r = Math.sqrt(-2 * Math.log(Math.random() + 1e-9)) * 0.55, a = Math.random() * Math.PI * 2
      start.set([Math.cos(a) * r, Math.sin(a) * r, (Math.random() - 0.5) * 0.6], k * 3)
      c.setHSL(0.72 + Math.random() * 0.2, 0.75, 0.45 + Math.random() * 0.25)
      noise.set([c.r, c.g, c.b], k * 3)
      color.set([c.r, c.g, c.b], k * 3)
      seed[k] = Math.random()
    }
    g.setAttribute('position', new THREE.BufferAttribute(target, 3))
    g.setAttribute('aStart', new THREE.BufferAttribute(start, 3))
    g.setAttribute('aNoiseColor', new THREE.BufferAttribute(noise, 3))
    g.setAttribute('aColor', new THREE.BufferAttribute(color, 3))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    return g
  }, [n])
  useEffect(() => () => geo.dispose(), [geo])
  const uniforms = useMemo(() => ({ uProgress: { value: 0 }, uDevelop: { value: 0 }, uTime: { value: 0 }, uSize: { value: 1 } }), [])
  const { size, viewport, invalidate } = useThree()
  uniforms.uSize.value = (size.width / n) * viewport.dpr * 1.15     // one cell, in framebuffer pixels

  // the finished image: sample one color per particle, then develop
  const develop = useRef(0)
  useEffect(() => {
    if (!imageUrl) { develop.current = 0; return }
    const img = new Image()
    img.onload = () => {
      const cv = document.createElement('canvas'); cv.width = n; cv.height = n
      const ctx = cv.getContext('2d', { willReadFrequently: true })
      if (!ctx) { onDeveloped(); return }
      ctx.drawImage(img, 0, 0, n, n)
      const px = ctx.getImageData(0, 0, n, n).data
      const col = geo.getAttribute('aColor') as THREE.BufferAttribute
      const lin = new THREE.Color()
      for (let k = 0; k < n * n; k++) { lin.setRGB(px[k * 4] / 255, px[k * 4 + 1] / 255, px[k * 4 + 2] / 255, THREE.SRGBColorSpace); col.setXYZ(k, lin.r, lin.g, lin.b) }
      col.needsUpdate = true
      develop.current = 0.0001
      invalidate()
    }
    img.onerror = onDeveloped
    img.src = imageUrl
  }, [imageUrl, n, geo, invalidate, onDeveloped])

  const shown = useRef(0)
  useFrame((st, dt) => {
    // write through the material's own uniforms: three.js ShaderMaterial may hold a copy of the object it was given
    const u = (mat.current?.uniforms ?? uniforms) as typeof uniforms
    u.uSize.value = uniforms.uSize.value
    if (still) shown.current = progress                                  // still frames: no easing, no wobble
    else {
      u.uTime.value = st.clock.elapsedTime
      shown.current += (progress - shown.current) * Math.min(1, dt * 3)    // ease toward the real progress
    }
    u.uProgress.value = imageUrl ? 1 : shown.current
    if (develop.current > 0 && develop.current < 1) {
      develop.current = Math.min(1, develop.current + dt / 0.9)
      if (develop.current >= 1) onDeveloped()       // frames come from the Clock (60 fps when idle), never uncapped
    }
    u.uDevelop.value = develop.current
    ;(window as unknown as { __ropeFrames?: number }).__ropeFrames = ((window as unknown as { __ropeFrames?: number }).__ropeFrames ?? 0) + 1
  })
  return (
    <points geometry={geo} scale={[size.width / 2, size.height / 2, 1]}>   {/* grid -1..1 -> the canvas in pixels */}
      <shaderMaterial ref={mat} vertexShader={vert} fragmentShader={frag} uniforms={uniforms} transparent depthWrite={false} />
    </points>
  )
}

/** Renders at most `FPS[mode]` frames per second (demand mode), none while the page is hidden; with 0 (cpu), one
 *  frame per 10% of progress. */
function Clock({ mode, progress }: { mode: FieldMode; progress: number }) {
  const { invalidate } = useThree()
  const tenth = Math.floor(progress * 10)
  useEffect(() => { if (FPS[mode] === 0) invalidate() }, [mode, tenth, invalidate])
  useEffect(() => {
    if (FPS[mode] === 0) return
    const id = window.setInterval(() => { if (!document.hidden) invalidate() }, 1000 / FPS[mode])
    return () => window.clearInterval(id)
  }, [mode, invalidate])
  return null
}

export default function DiffusionField({ progress, mode, imageUrl, onDeveloped, density }:
  { progress: number; mode: FieldMode; imageUrl: string | null; onDeveloped: () => void; density: number }) {
  return (
    <Canvas frameloop="demand" orthographic camera={{ position: [0, 0, 5], zoom: 1 }}
            dpr={[1, 1.5]} gl={{ antialias: false, powerPreference: mode === 'cpu' ? 'low-power' : 'default', alpha: true }}
            aria-hidden>
      <Clock mode={mode} progress={progress} />
      <Particles n={density} progress={progress} imageUrl={imageUrl} onDeveloped={onDeveloped} still={mode === 'cpu'} />
    </Canvas>
  )
}
