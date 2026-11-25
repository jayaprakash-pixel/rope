import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { reducedMotion } from '../site'

// The hero, as a short film driven by scroll:
//   intro (by itself): a closed laptop floats in the dark, opens, its screen wakes, a scan beam reads it (the probe);
//   scroll 0 -> 0.45: the camera flies to the screen; the Rope app on it plans and forms a picture from noise;
//   scroll 0.45 -> 0.8: a glowing braided rope unfurls out of the screen and the pictures fly out along it;
//   scroll 0.8 -> 1: the camera pulls back: the laptop in an orbit of the pictures it made.
// `progress` is a ref the page writes (0..1); the HUD labels are DOM nodes this scene moves every frame.

export type Pic = { src: string; model: string; secs: number; prompt: string; steps: number }
type Props = { progress: React.MutableRefObject<number>; pics: Pic[]; hud: React.MutableRefObject<(HTMLElement | null)[]>; onIntro?: () => void }

const strandVert = /* glsl */ `
  varying float vU; varying vec3 vN; varying vec3 vV;
  void main() { vU = uv.x; vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`
const strandFrag = /* glsl */ `
  uniform float uTime; uniform float uPhase; uniform float uReveal; uniform vec3 uA; uniform vec3 uB;
  varying float vU; varying vec3 vN; varying vec3 vV;
  void main() {
    if (vU > uReveal) discard;
    float fres = pow(1.0 - max(dot(vN, vV), 0.0), 2.0);
    float p = fract(vU * 4.0 - uTime * 0.35 + uPhase);
    float pulse = smoothstep(0.0, 0.04, p) * (1.0 - smoothstep(0.04, 0.2, p));
    float tip = smoothstep(uReveal - 0.03, uReveal, vU);              // a bright growing tip
    vec3 c = mix(uA, uB, vU);
    gl_FragColor = vec4(c * (0.55 + 0.9 * fres) + c * pulse * 2.2 + vec3(1.0) * tip * 1.5, 1.0);
  }`

const clamp01 = (x: number) => Math.min(1, Math.max(0, x))
const smooth = (a: number, b: number, x: number) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t) }

// The Rope app as it looks on the laptop's screen, drawn into a canvas: prompt, plan, and the picture forming from noise.
class Screen {
  canvas = document.createElement('canvas')
  g: CanvasRenderingContext2D
  tex: THREE.CanvasTexture
  small = document.createElement('canvas')
  noise: Uint8ClampedArray
  imgs: (ImageData | null)[] = []
  constructor(pics: Pic[]) {
    this.canvas.width = 1280; this.canvas.height = 800
    this.g = this.canvas.getContext('2d')!
    this.tex = new THREE.CanvasTexture(this.canvas)
    this.tex.colorSpace = THREE.SRGBColorSpace
    this.tex.anisotropy = 8
    this.small.width = this.small.height = 160
    this.noise = new Uint8ClampedArray(160 * 160 * 4)
    for (let i = 0; i < 160 * 160; i++) {
      const v = Math.random()
      this.noise[i * 4] = 60 + v * 140; this.noise[i * 4 + 1] = 20 + v * 50; this.noise[i * 4 + 2] = 110 + v * 140; this.noise[i * 4 + 3] = 255
    }
    pics.forEach((p, i) => {
      this.imgs[i] = null
      const im = new Image()
      im.onload = () => { const c = document.createElement('canvas'); c.width = c.height = 160; const g = c.getContext('2d')!; g.drawImage(im, 0, 0, 160, 160); this.imgs[i] = g.getImageData(0, 0, 160, 160) }
      im.src = p.src
    })
  }
  draw(o: { power: number; typed: string; plan: number; form: number; pic: Pic; idx: number; step: number }) {
    const g = this.g, W = 1280, H = 800
    g.globalAlpha = 1
    g.fillStyle = '#050409'; g.fillRect(0, 0, W, H)
    if (o.power <= 0) { this.tex.needsUpdate = true; return }
    g.globalAlpha = o.power
    const bg = g.createLinearGradient(0, 0, W, H); bg.addColorStop(0, '#120e22'); bg.addColorStop(1, '#0a0814')
    g.fillStyle = bg; g.fillRect(0, 0, W, H)
    // top bar
    g.fillStyle = 'rgba(255,255,255,0.05)'; g.fillRect(0, 0, W, 64)
    g.fillStyle = '#a78bfa'; g.beginPath(); g.arc(44, 32, 11, 0, Math.PI * 2); g.fill()
    g.fillStyle = '#f4f2fb'; g.font = '600 26px Inter Variable, system-ui, sans-serif'; g.fillText('rope', 66, 41)
    g.fillStyle = '#8b86a8'; g.font = '500 20px Inter Variable, system-ui, sans-serif'; g.fillText('Create      Gallery      This laptop', 170, 40)
    g.fillStyle = '#6ee7b7'; g.font = '500 18px JetBrains Mono, monospace'; g.fillText('8 GB RAM · GTX 1650 4 GB', 960, 40)
    // left: prompt and plan
    g.fillStyle = '#f4f2fb'; g.font = '700 44px Inter Variable, system-ui, sans-serif'; g.fillText('Describe an image', 56, 140)
    g.strokeStyle = 'rgba(167,139,250,0.55)'; g.lineWidth = 2; this.round(56, 170, 560, 120, 16); g.stroke()
    g.fillStyle = '#d9d5ee'; g.font = '400 23px Inter Variable, system-ui, sans-serif'
    this.wrap(o.typed + (o.typed.length < o.pic.prompt.length && Math.floor(performance.now() / 400) % 2 ? '|' : ''), 78, 210, 520, 32)
    if (o.plan > 0) {
      g.globalAlpha = o.power * o.plan
      g.fillStyle = 'rgba(139,92,246,0.10)'; this.round(56, 320, 560, 250, 18); g.fill()
      g.strokeStyle = 'rgba(167,139,250,0.35)'; this.round(56, 320, 560, 250, 18); g.stroke()
      g.fillStyle = '#a39fbd'; g.font = '600 18px Inter Variable, system-ui, sans-serif'; g.fillText('PLAN ON THIS LAPTOP', 84, 362)
      g.fillStyle = '#ffffff'; g.font = '700 76px Inter Variable, system-ui, sans-serif'; g.fillText(`~${o.pic.secs} s`, 80, 450)
      g.fillStyle = '#a39fbd'; g.font = '500 20px JetBrains Mono, monospace'
      g.fillText(`${o.pic.model.split(' · ')[0]} · 512×512 · ${o.pic.steps} step${o.pic.steps === 1 ? '' : 's'}`, 84, 500)
      g.fillText('memory 1.4 GB · fits', 84, 535)
      g.globalAlpha = o.power
    }
    // right: the picture forming
    const X = 680, Y = 100, S = 560
    g.fillStyle = '#07060d'; this.round(X, Y, S, S, 22); g.fill()
    const im = this.imgs[o.idx]
    if (im && o.form > 0) {
      const sg = this.small.getContext('2d')!
      const out = sg.createImageData(160, 160), d = out.data, n = this.noise, s = im.data
      const k = o.form
      for (let i = 0; i < d.length; i += 4) {
        const r = Math.min(1, Math.max(0, k * 1.25 - (n[i] / 255) * 0.25))
        d[i] = n[i] + (s[i] - n[i]) * r; d[i + 1] = n[i + 1] + (s[i + 1] - n[i + 1]) * r; d[i + 2] = n[i + 2] + (s[i + 2] - n[i + 2]) * r; d[i + 3] = 255
      }
      sg.putImageData(out, 0, 0)
      g.save(); this.round(X, Y, S, S, 22); g.clip()
      g.imageSmoothingEnabled = k > 0.92
      g.drawImage(this.small, X, Y, S, S)
      g.restore()
    }
    // progress
    g.fillStyle = 'rgba(255,255,255,0.08)'; this.round(X, Y + S + 26, S, 8, 4); g.fill()
    const bar = g.createLinearGradient(X, 0, X + S, 0); bar.addColorStop(0, '#a78bfa'); bar.addColorStop(0.6, '#f472b6'); bar.addColorStop(1, '#22d3ee')
    g.fillStyle = bar; this.round(X, Y + S + 26, Math.max(8, S * o.form), 8, 4); g.fill()
    g.fillStyle = o.form >= 1 ? '#6ee7b7' : '#a39fbd'; g.font = '500 19px JetBrains Mono, monospace'
    g.fillText(o.form >= 1 ? `✓ done in ${o.pic.secs} s · saved` : o.form > 0 ? `shaping the image · step ${o.step} of ${o.pic.steps}` : 'ready', X, Y + S + 66)
    this.tex.needsUpdate = true
  }
  round(x: number, y: number, w: number, h: number, r: number) { const g = this.g; g.beginPath(); g.roundRect(x, y, w, h, r) }
  wrap(t: string, x: number, y: number, w: number, lh: number) {
    const g = this.g; let line = ''
    for (const word of t.split(' ')) {
      const test = line ? line + ' ' + word : word
      if (g.measureText(test).width > w && line) { g.fillText(line, x, y); y += lh; line = word } else line = test
    }
    g.fillText(line, x, y)
  }
}

export default function Journey({ progress, pics, hud, onIntro }: Props) {
  const host = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = host.current
    if (!el) return
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' }) } catch { el.classList.add('no-webgl'); onIntro?.(); return }
    const small = window.innerWidth < 760
    const still = reducedMotion()
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, small ? 1.5 : 2))
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.05
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#05040a')
    scene.fog = new THREE.FogExp2('#05040a', 0.045)
    const pmrem = new THREE.PMREMGenerator(renderer)
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200)

    // light: a violet rim from behind, a pink key from the front left, a cool fill
    scene.add(new THREE.HemisphereLight('#6d5bd0', '#05040a', 0.35))
    const rim = new THREE.DirectionalLight('#a78bfa', 3.2); rim.position.set(-3, 5, -6); scene.add(rim)
    const key = new THREE.PointLight('#f472b6', 30, 20, 2); key.position.set(-4, 3, 4); scene.add(key)
    const fill = new THREE.PointLight('#22d3ee', 12, 16, 2); fill.position.set(5, 1.5, 3); scene.add(fill)

    // ---------------------------------------------------------------- the laptop
    const laptop = new THREE.Group()
    const shell = new THREE.MeshPhysicalMaterial({ color: '#1b1828', metalness: 0.85, roughness: 0.32, clearcoat: 0.6, clearcoatRoughness: 0.25 })
    const base = new THREE.Mesh(new RoundedBoxGeometry(3.4, 0.13, 2.3, 4, 0.06), shell)
    laptop.add(base)
    const keyMat = new THREE.MeshStandardMaterial({ color: '#0e0c18', metalness: 0.3, roughness: 0.6, emissive: '#7c3aed', emissiveIntensity: 0.0 })
    const cols = 15, rows = 5, kw = 0.17
    const keys = new THREE.InstancedMesh(new RoundedBoxGeometry(kw, 0.03, kw, 2, 0.015), keyMat, cols * rows)
    const m4 = new THREE.Matrix4()
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      m4.makeTranslation(-1.33 + c * 0.19, 0.075, -0.82 + r * 0.19); keys.setMatrixAt(r * cols + c, m4)
    }
    laptop.add(keys)
    const glowStrip = new THREE.Mesh(new THREE.PlaneGeometry(2.9, 1.0), new THREE.MeshBasicMaterial({ color: '#8b5cf6', transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }))
    glowStrip.rotation.x = -Math.PI / 2; glowStrip.position.set(0, 0.067, -0.44); laptop.add(glowStrip)
    const pad = new THREE.Mesh(new RoundedBoxGeometry(1.1, 0.01, 0.62, 2, 0.02), new THREE.MeshPhysicalMaterial({ color: '#221f33', metalness: 0.6, roughness: 0.25 }))
    pad.position.set(0, 0.07, 0.68); laptop.add(pad)
    // the lid hinges at the back edge; rotation.x = PI/2 is closed, -0.28 is open
    const hinge = new THREE.Group(); hinge.position.set(0, 0.065, -1.15); laptop.add(hinge)
    const lid = new THREE.Mesh(new RoundedBoxGeometry(3.4, 2.2, 0.08, 4, 0.05), shell); lid.position.set(0, 1.1, -0.0); hinge.add(lid)
    const pics_ = pics
    const screen = new Screen(pics_)
    const screenMat = new THREE.MeshBasicMaterial({ map: screen.tex, toneMapped: false, color: new THREE.Color(1, 1, 1) })
    const display = new THREE.Mesh(new THREE.PlaneGeometry(3.16, 1.975), screenMat)
    display.position.set(0, 1.1, 0.042); hinge.add(display)
    const logo = new THREE.Mesh(new THREE.CircleGeometry(0.12, 40), new THREE.MeshBasicMaterial({ color: '#c4b5fd', toneMapped: false }))
    logo.position.set(0, 1.1, -0.042); logo.rotation.y = Math.PI; hinge.add(logo)
    hinge.rotation.x = still ? -0.28 : Math.PI / 2 - 0.02
    const LX = small ? 0 : 2.5
    laptop.position.set(LX, -0.55, 0)
    laptop.rotation.y = small ? -0.25 : -0.42
    scene.add(laptop)

    // the probe: a scan beam sweeping over the laptop
    const beam = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 0.05), new THREE.MeshBasicMaterial({ color: '#c4b5fd', transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }))
    const beamGlow = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 0.9), new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uO: { value: 0 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: 'uniform float uO; varying vec2 vUv; void main(){ float a = pow(1.0 - abs(vUv.y - 0.5) * 2.0, 3.0); gl_FragColor = vec4(0.6,0.45,1.0, a * uO * 0.6); }' }))
    const scan = new THREE.Group(); scan.add(beam); scan.add(beamGlow); scan.rotation.x = -Math.PI / 2
    laptop.add(scan)
    // HUD anchors (laptop space): CPU, RAM, GPU, disk
    const anchors = [new THREE.Vector3(-1.1, 0.1, -0.5), new THREE.Vector3(1.05, 0.1, -0.6), new THREE.Vector3(-0.9, 1.9, -1.05), new THREE.Vector3(1.2, 0.1, 0.75)]

    // ---------------------------------------------------------------- the rope out of the screen
    const ropeCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 1.1, 0.05), new THREE.Vector3(0.5, 1.5, 1.5), new THREE.Vector3(2.0, 2.3, 2.2),
      new THREE.Vector3(3.8, 1.8, 0.8), new THREE.Vector3(4.6, 0.6, -2.0), new THREE.Vector3(3.2, -0.3, -4.8), new THREE.Vector3(0.2, 0.4, -6.0),
    ], false, 'catmullrom', 0.4)
    const frames = ropeCurve.computeFrenetFrames(800, false)
    class Strand extends THREE.Curve<THREE.Vector3> {
      phase: number
      constructor(phase: number) { super(); this.phase = phase }
      getPoint(t: number, out = new THREE.Vector3()) {
        const i = Math.min(800, Math.round(t * 800))
        const c = ropeCurve.getPointAt(t), a = t * Math.PI * 22 + this.phase, r = 0.09 * (0.4 + 0.6 * Math.min(1, t * 6))
        return out.copy(c).addScaledVector(frames.normals[i], Math.cos(a) * r).addScaledVector(frames.binormals[i], Math.sin(a) * r)
      }
    }
    const ropeGroup = new THREE.Group(); hinge.add(ropeGroup)
    const strandMats: THREE.ShaderMaterial[] = []
    ;[['#a78bfa', '#f472b6'], ['#7c3aed', '#e879f9'], ['#c084fc', '#22d3ee']].forEach(([a, b], i) => {
      const m = new THREE.ShaderMaterial({ vertexShader: strandVert, fragmentShader: strandFrag, toneMapped: false,
        uniforms: { uTime: { value: 0 }, uPhase: { value: i / 3 }, uReveal: { value: 0 }, uA: { value: new THREE.Color(a) }, uB: { value: new THREE.Color(b) } } })
      strandMats.push(m)
      ropeGroup.add(new THREE.Mesh(new THREE.TubeGeometry(new Strand((i * Math.PI * 2) / 3), small ? 500 : 900, 0.032, 8, false), m))
    })

    // ---------------------------------------------------------------- the pictures: they fly out along the rope into an orbit
    const loader = new THREE.TextureLoader()
    const cards = pics_.map((p, i) => {
      const tex = loader.load(p.src); tex.colorSpace = THREE.SRGBColorSpace
      const g = new THREE.Group()
      const img = new THREE.Mesh(new THREE.PlaneGeometry(1.25, 1.25), new THREE.MeshBasicMaterial({ map: tex, toneMapped: false, transparent: true, opacity: 0 }))
      const halo = new THREE.Mesh(new THREE.PlaneGeometry(1.45, 1.45), new THREE.MeshBasicMaterial({ color: i % 2 ? '#f472b6' : '#a78bfa', transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }))
      halo.position.z = -0.01
      g.add(halo); g.add(img)
      g.visible = false
      scene.add(g)
      const ang = (i / pics_.length) * Math.PI * 2 + 0.6
      return { g, img, halo, ang, y: [0.9, -0.6, 1.6, -0.2, 1.2, -0.9][i % 6] }
    })

    // ---------------------------------------------------------------- dust
    const N = small ? 900 : 2200
    const dp = new Float32Array(N * 3)
    for (let i = 0; i < N; i++) { dp[i * 3] = (Math.random() - 0.5) * 40; dp[i * 3 + 1] = (Math.random() - 0.5) * 22; dp[i * 3 + 2] = (Math.random() - 0.5) * 40 }
    const dg = new THREE.BufferGeometry(); dg.setAttribute('position', new THREE.BufferAttribute(dp, 3))
    const dot = document.createElement('canvas'); dot.width = dot.height = 32
    const dg2 = dot.getContext('2d')!, rg = dg2.createRadialGradient(16, 16, 0, 16, 16, 16)
    rg.addColorStop(0, 'rgba(255,255,255,1)'); rg.addColorStop(0.4, 'rgba(255,255,255,0.5)'); rg.addColorStop(1, 'rgba(255,255,255,0)')
    dg2.fillStyle = rg; dg2.fillRect(0, 0, 32, 32)
    const dust = new THREE.Points(dg, new THREE.PointsMaterial({ color: '#b9a6ff', size: 0.05, map: new THREE.CanvasTexture(dot), transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending }))
    scene.add(dust)

    // ---------------------------------------------------------------- post: bloom (desktop)
    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), small ? 0.55 : 0.85, 0.55, 0.62)
    composer.addPass(bloom)
    composer.addPass(new OutputPass())

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h, false); composer.setSize(w, h); bloom.setSize(w, h)
      camera.aspect = w / h; camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize); ro.observe(el)
    const mouse = new THREE.Vector2()
    const onMove = (e: PointerEvent) => mouse.set(e.clientX / window.innerWidth - 0.5, e.clientY / window.innerHeight - 0.5)
    window.addEventListener('pointermove', onMove)

    // camera keyframes over the scroll: [position, look-at]
    const L = new THREE.Vector3(LX, 0, 0)
    const kf: [number, THREE.Vector3, THREE.Vector3][] = small
      // phone: the text sits at the top, so the camera looks above the laptop and it appears in the lower half
      ? [[0, new THREE.Vector3(0, 2.6, 17), new THREE.Vector3(0, 3.3, 0)], [0.4, new THREE.Vector3(-0.4, 2.4, 9.5), new THREE.Vector3(0, 2.1, 0)],
         [0.72, new THREE.Vector3(-1.4, 3.6, 15), new THREE.Vector3(1.0, 3.0, -1)], [1, new THREE.Vector3(0, 5.8, 20), new THREE.Vector3(0.4, 3.6, -1)]]
      : [[0, new THREE.Vector3(0.0, 1.9, 9.6), new THREE.Vector3(0.0, -0.1, 0)],
         [0.4, new THREE.Vector3(L.x - 0.8, 1.7, 6.0), new THREE.Vector3(L.x - 1.5, 0.45, -0.6)],
         [0.72, new THREE.Vector3(-1.6, 2.8, 12.6), new THREE.Vector3(0.9, 0.9, -1.0)],
         [1, new THREE.Vector3(-2.8, 4.6, 17.5), new THREE.Vector3(-0.5, 0.3, -1.0)]]
    const at = (p: number) => {
      let i = 0; while (i < kf.length - 2 && p > kf[i + 1][0]) i++
      const [p0, a0, b0] = kf[i], [p1, a1, b1] = kf[i + 1]
      const t = smooth(p0, p1, p)
      return [a0.clone().lerp(a1, t), b0.clone().lerp(b1, t)] as const
    }

    // intro timeline (seconds since start), then a loop of pictures on the screen
    const T = { open: [0.5, 2.2], power: [1.9, 2.6], scan: [2.4, 4.2], hud: [3.0, 4.6], type: [4.4, 6.6], plan: [6.6, 7.2] }
    const start = performance.now() / 1000
    let introSent = false, raf = 0, visible = true, cam = at(0)
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting }); io.observe(el)
    const pos = new THREE.Vector3(), look = new THREE.Vector3()
    const wp = new THREE.Vector3()
    let lastDraw = 0

    const frame = () => {
      raf = requestAnimationFrame(frame)
      if (!visible) return
      const now = performance.now() / 1000
      const e = still ? 60 : now - start
      const p = clamp01(progress.current)

      // lid, screen power, keyboard glow
      const open = smooth(T.open[0], T.open[1], e)
      hinge.rotation.x = THREE.MathUtils.lerp(Math.PI / 2 - 0.02, -0.28, open)
      const power = smooth(T.power[0], T.power[1], e)
      keyMat.emissiveIntensity = 0.55 * power
      ;(glowStrip.material as THREE.MeshBasicMaterial).opacity = 0.12 * power
      // scan beam: sweeps front to back over the keyboard, then up the screen
      const sk = smooth(T.scan[0], T.scan[1], e), son = sk > 0 && sk < 1 ? 1 : 0
      scan.position.set(0, 0.12, THREE.MathUtils.lerp(1.2, -1.15, sk))
      ;(beam.material as THREE.MeshBasicMaterial).opacity = son * 0.9
      ;(beamGlow.material as THREE.ShaderMaterial).uniforms.uO.value = son
      if (!introSent && e > T.plan[1]) { introSent = true; onIntro?.() }

      // the picture loop on the screen (after the intro): type, plan, form, hold
      const cyc = 9.5, loopE = Math.max(0, e - T.type[0]), idx = Math.floor(loopE / cyc) % pics_.length, le = loopE % cyc
      const pic = pics_[idx]
      const typed = pic.prompt.slice(0, Math.floor(clamp01(le / 2.0) * pic.prompt.length))
      const planK = smooth(2.1, 2.7, le), form = clamp01((le - 2.9) / 3.4)
      const step = Math.min(pic.steps, Math.floor(form * pic.steps) + 1)
      if (now - lastDraw > 1 / 30) { screen.draw({ power, typed: loopE > 0 ? typed : '', plan: loopE > 0 ? planK : 0, form, pic, idx, step }); lastDraw = now }

      // HUD labels follow their anchors while the probe result is on screen
      const hudK = smooth(T.hud[0], T.hud[1], e) * (1 - smooth(0.18, 0.32, p))
      anchors.forEach((a, i) => {
        const node = hud.current[i]; if (!node) return
        wp.copy(a); laptop.localToWorld(wp); wp.project(camera)
        const x = (wp.x * 0.5 + 0.5) * el.clientWidth, y = (-wp.y * 0.5 + 0.5) * el.clientHeight
        const k = clamp01(hudK * 1.6 - i * 0.18)
        node.style.transform = `translate(${x}px, ${y}px)`
        node.style.opacity = String(k)
      })

      // the rope and the pictures, by scroll
      const reveal = smooth(0.38, 0.72, p)
      strandMats.forEach((m) => { m.uniforms.uTime.value = now; m.uniforms.uReveal.value = reveal })
      hinge.updateMatrixWorld()
      cards.forEach((c, i) => {
        const k = smooth(0.46 + i * 0.045, 0.74 + i * 0.03, p)
        c.g.visible = k > 0.001
        if (!c.g.visible) return
        const orbit = new THREE.Vector3(L.x + (small ? 0 : 0.4) + Math.cos(c.ang + now * 0.06) * (small ? 3.0 : 3.3), c.y, Math.sin(c.ang + now * 0.06) * (small ? 3.0 : 3.3) - 1.0)
        // along the rope first, then to its place in the orbit
        const onRope = ropeCurve.getPointAt(Math.min(1, k * 1.4) * 0.55).clone(); hinge.localToWorld(onRope)
        c.g.position.lerpVectors(onRope, orbit, smooth(0.45, 1, k))
        c.g.scale.setScalar(0.25 + 0.75 * k)
        c.g.lookAt(camera.position)
        ;(c.img.material as THREE.MeshBasicMaterial).opacity = smooth(0, 0.3, k)
        ;(c.halo.material as THREE.MeshBasicMaterial).opacity = 0.35 * smooth(0, 0.3, k)
      })

      // camera: keyframes by scroll, a little float and mouse parallax
      cam = at(p)
      pos.copy(cam[0]).add(new THREE.Vector3(mouse.x * 0.7, -mouse.y * 0.45 + Math.sin(now * 0.4) * 0.06, 0))
      camera.position.lerp(pos, still ? 1 : 0.08)
      look.lerp(cam[1], still ? 1 : 0.08)
      camera.lookAt(look)
      laptop.position.y = -0.55 + Math.sin(now * 0.7) * 0.05 * (still ? 0 : 1)
      dust.rotation.y = now * 0.01
      composer.render()
    }
    look.copy(cam[1]); camera.position.copy(cam[0])
    frame()

    return () => {
      cancelAnimationFrame(raf); ro.disconnect(); io.disconnect()
      window.removeEventListener('pointermove', onMove)
      scene.traverse((o) => { const m = o as THREE.Mesh; m.geometry?.dispose(); const mt = m.material as THREE.Material | undefined; mt?.dispose?.() })
      screen.tex.dispose(); pmrem.dispose(); composer.dispose(); renderer.dispose(); renderer.domElement.remove()
    }
  }, [progress, pics, hud, onIntro])
  return <div ref={host} className="journey-canvas" aria-hidden />
}
