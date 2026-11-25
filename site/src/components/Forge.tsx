import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { reducedMotion } from '../site'

// The hero picture: noise that becomes a real Rope picture, the way diffusion works. Each pixel of the picture is one
// particle; it starts as violet noise somewhere in the card and settles into its place and colour, step by step. Then
// the picture scatters and the next one forms. `onPhase` tells the page what to show in the plan readout.
export type Phase = { index: number; step: number; steps: number; state: 'forming' | 'done' | 'scatter' }

const vert = /* glsl */ `
  attribute vec2 aGrid;
  attribute vec3 aNoise;
  attribute vec3 aColor;
  attribute float aSeed;
  uniform float uT;          // 0 = noise, 1 = the picture
  uniform float uTime;
  uniform float uCell;       // one picture pixel in device pixels
  uniform vec2 uMouse;
  varying vec3 vColor;
  varying float vK;
  float ease(float x) { return x < 0.5 ? 4.0 * x * x * x : 1.0 - pow(-2.0 * x + 2.0, 3.0) / 2.0; }
  void main() {
    float d = aSeed * 0.45;                              // particles arrive at different times
    float k = ease(clamp((uT - d) / 0.55, 0.0, 1.0));
    vec3 n = aNoise;
    n.xy += 0.04 * vec2(sin(uTime * 0.9 + aSeed * 37.0), cos(uTime * 0.7 + aSeed * 53.0));
    vec3 p = mix(n, vec3(aGrid, 0.0), k);
    // the mouse pushes settled pixels a little
    vec2 dm = p.xy - uMouse;
    float push = exp(-dot(dm, dm) * 18.0) * 0.06 * k;
    p.xy += normalize(dm + 1e-5) * push;
    vec3 noiseCol = mix(vec3(0.49, 0.23, 0.93), vec3(0.93, 0.28, 0.6), aSeed);
    vColor = mix(noiseCol * 0.9, aColor, k);
    vK = k;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
    gl_PointSize = mix(uCell * 0.55, uCell * 1.06, k);
  }
`
const frag = /* glsl */ `
  varying vec3 vColor;
  varying float vK;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float round = 1.0 - smoothstep(0.32, 0.5, length(c));            // soft dot while it travels
    float square = step(max(abs(c.x), abs(c.y)), 0.5);                // a crisp pixel when it has arrived
    float a = mix(round, square, smoothstep(0.75, 1.0, vK));
    if (a < 0.02) discard;
    gl_FragColor = vec4(vColor, a * mix(0.75, 1.0, vK));
  }
`

function pixels(src: string, n: number): Promise<Float32Array> {
  return new Promise((ok, fail) => {
    const img = new Image()
    img.onload = () => {
      const c = document.createElement('canvas'); c.width = c.height = n
      const g = c.getContext('2d')!; g.drawImage(img, 0, 0, n, n)
      const d = g.getImageData(0, 0, n, n).data
      const out = new Float32Array(n * n * 3)
      for (let i = 0; i < n * n; i++) {
        // the colour space of the canvas is sRGB; the material writes it straight out
        out[i * 3] = d[i * 4] / 255; out[i * 3 + 1] = d[i * 4 + 1] / 255; out[i * 3 + 2] = d[i * 4 + 2] / 255
      }
      ok(out)
    }
    img.onerror = fail
    img.src = src
  })
}

export default function Forge({ images, steps, onPhase }: { images: string[]; steps: number[]; onPhase: (p: Phase) => void }) {
  const host = useRef<HTMLDivElement>(null)
  const phaseCb = useRef(onPhase)
  phaseCb.current = onPhase
  useEffect(() => {
    const el = host.current
    if (!el || !images.length) return
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true }) } catch { el.classList.add('no-webgl'); return }
    const dpr = Math.min(window.devicePixelRatio, 2)
    renderer.setPixelRatio(dpr)
    el.appendChild(renderer.domElement)
    const N = window.innerWidth < 700 ? 96 : 150                       // picture pixels per side
    const scene = new THREE.Scene()
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10)

    const count = N * N
    const grid = new Float32Array(count * 2), noise = new Float32Array(count * 3), color = new Float32Array(count * 3), seed = new Float32Array(count)
    const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x
      grid[i * 2] = ((x + 0.5) / N) * 2 - 1
      grid[i * 2 + 1] = 1 - ((y + 0.5) / N) * 2
      noise[i * 3] = gauss() * 0.55; noise[i * 3 + 1] = gauss() * 0.55; noise[i * 3 + 2] = 0
      seed[i] = Math.random()
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3))
    geo.setAttribute('aGrid', new THREE.BufferAttribute(grid, 2))
    geo.setAttribute('aNoise', new THREE.BufferAttribute(noise, 3))
    const colAttr = new THREE.BufferAttribute(color, 3)
    geo.setAttribute('aColor', colAttr)
    geo.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    const mat = new THREE.ShaderMaterial({ vertexShader: vert, fragmentShader: frag, transparent: true, depthWrite: false,
      uniforms: { uT: { value: 0 }, uTime: { value: 0 }, uCell: { value: 4 }, uMouse: { value: new THREE.Vector2(9, 9) } } })
    scene.add(new THREE.Points(geo, mat))

    const resize = () => {
      const w = el.clientWidth
      renderer.setSize(w, w, false)
      mat.uniforms.uCell.value = (w * dpr) / N
    }
    resize()
    const ro = new ResizeObserver(resize); ro.observe(el)
    const onMove = (e: PointerEvent) => {
      const r = el.getBoundingClientRect()
      mat.uniforms.uMouse.value.set(((e.clientX - r.left) / r.width) * 2 - 1, 1 - ((e.clientY - r.top) / r.height) * 2)
    }
    const onLeave = () => mat.uniforms.uMouse.value.set(9, 9)
    el.addEventListener('pointermove', onMove); el.addEventListener('pointerleave', onLeave)

    let alive = true, raf = 0, visible = true
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting }); io.observe(el)
    const cache: Record<number, Float32Array> = {}
    const load = (i: number) => (cache[i] ? Promise.resolve(cache[i]) : pixels(images[i], N).then((p) => (cache[i] = p)))
    const still = reducedMotion()

    // timeline of one picture: form (steps), hold, scatter
    const FORM = 3.2, HOLD = 2.6, SCATTER = 1.1
    let index = 0, t0 = performance.now() / 1000, last = ''
    const report = (p: Phase) => { const k = `${p.index}-${p.step}-${p.state}`; if (k !== last) { last = k; phaseCb.current(p) } }
    const begin = (i: number) => load(i).then((px) => { if (!alive) return; color.set(px); colAttr.needsUpdate = true; index = i; t0 = performance.now() / 1000; load((i + 1) % images.length).catch(() => {}) })

    const frame = () => {
      raf = requestAnimationFrame(frame)
      if (!visible) return
      const now = performance.now() / 1000
      mat.uniforms.uTime.value = now
      const s = steps[index] || 4
      const e = now - t0
      if (!isFinite(t0)) { /* the next picture is loading: stay as noise */ }
      else if (still) { mat.uniforms.uT.value = 1; report({ index, step: s, steps: s, state: 'done' }) }
      else if (e < FORM) {
        // the picture sharpens in `s` visible steps, like the live preview
        const k = e / FORM, stepK = Math.min(s, Math.floor(k * s) + 1)
        // settle gradually over the whole forming time (a 1-step model still takes the full 3.2 s to appear)
        mat.uniforms.uT.value = THREE.MathUtils.lerp(mat.uniforms.uT.value, Math.min(k * 1.08, stepK / s), 0.12)
        report({ index, step: stepK, steps: s, state: 'forming' })
      } else if (e < FORM + HOLD) {
        mat.uniforms.uT.value = THREE.MathUtils.lerp(mat.uniforms.uT.value, 1, 0.12)
        report({ index, step: s, steps: s, state: 'done' })
      } else if (e < FORM + HOLD + SCATTER) {
        mat.uniforms.uT.value = 1 - (e - FORM - HOLD) / SCATTER
        report({ index, step: s, steps: s, state: 'scatter' })
      } else {
        mat.uniforms.uT.value = 0
        t0 = Infinity                                                // wait until the next picture is ready
        begin((index + 1) % images.length)
      }
      renderer.render(scene, camera)
    }
    begin(0).then(() => { if (alive) frame() })

    return () => {
      alive = false
      cancelAnimationFrame(raf)
      ro.disconnect(); io.disconnect()
      el.removeEventListener('pointermove', onMove); el.removeEventListener('pointerleave', onLeave)
      geo.dispose(); mat.dispose(); renderer.dispose(); renderer.domElement.remove()
    }
  }, [images, steps])
  return <div ref={host} className="forge" aria-hidden />
}
