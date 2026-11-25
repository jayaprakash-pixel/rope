import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { reducedMotion } from '../site'

gsap.registerPlugin(ScrollTrigger)

// The hero: three strands braided into one glowing rope, light pulses running along it (data flowing through),
// and a field of particles. Mouse moves the camera a little; scrolling turns the rope away.
const strandVert = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vN;
  varying vec3 vView;
  void main() {
    vUv = uv;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal);
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`
const strandFrag = /* glsl */ `
  uniform float uTime;
  uniform float uPhase;
  uniform vec3 uA;
  uniform vec3 uB;
  uniform vec3 uC;
  varying vec2 vUv;
  varying vec3 vN;
  varying vec3 vView;
  void main() {
    float t = vUv.x;
    vec3 base = mix(uA, uB, smoothstep(0.0, 0.55, t));
    base = mix(base, uC, smoothstep(0.55, 1.0, t));
    float fres = pow(1.0 - max(dot(vN, vView), 0.0), 2.2);
    // pulses travelling along the strand
    float p = fract(t * 3.0 - uTime * 0.18 + uPhase);
    float pulse = smoothstep(0.0, 0.05, p) * (1.0 - smoothstep(0.05, 0.22, p));
    vec3 col = base * (0.35 + 0.65 * fres) + base * pulse * 1.6 + vec3(1.0) * pulse * 0.25;
    float edge = smoothstep(0.0, 0.06, t) * (1.0 - smoothstep(0.94, 1.0, t));
    gl_FragColor = vec4(col, (0.55 + 0.45 * fres + pulse) * edge);
  }
`
const pointVert = /* glsl */ `
  uniform float uTime;
  uniform float uSize;
  attribute float aSeed;
  varying float vA;
  void main() {
    vec3 p = position;
    p.y += sin(uTime * 0.25 + aSeed * 6.2831) * 0.25;
    p.x += cos(uTime * 0.18 + aSeed * 12.0) * 0.18;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_PointSize = uSize * (0.6 + aSeed) * (8.0 / -mv.z);
    vA = 0.25 + 0.75 * fract(aSeed * 7.31 + uTime * 0.05);
    gl_Position = projectionMatrix * mv;
  }
`
const pointFrag = /* glsl */ `
  uniform vec3 uColor;
  varying float vA;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.0, d);
    gl_FragColor = vec4(uColor, a * vA * 0.7);
  }
`

class Braid extends THREE.Curve<THREE.Vector3> {
  phase: number
  radius: number
  constructor(phase: number, radius: number) { super(); this.phase = phase; this.radius = radius }
  getPoint(t: number, target = new THREE.Vector3()) {
    // the rope's axis: a soft S across the view; the strand winds around it
    const x = (t - 0.5) * 16
    const y = Math.sin(t * Math.PI * 1.4) * 1.3 - (t - 0.5) * 1.6
    const z = Math.cos(t * Math.PI) * 1.2
    const a = t * Math.PI * 9 + this.phase
    return target.set(x, y + Math.cos(a) * this.radius, z + Math.sin(a) * this.radius)
  }
}

export default function RopeScene() {
  const host = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = host.current
    if (!el) return
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' }) }
    catch { el.classList.add('no-webgl'); return }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100)
    camera.position.set(0, 0, 11)
    const rope = new THREE.Group()
    rope.rotation.set(0.15, -0.35, -0.12)
    scene.add(rope)

    const palette: [string, string, string][] = [['#7c3aed', '#c084fc', '#ec4899'], ['#6d28d9', '#a78bfa', '#f472b6'], ['#8b5cf6', '#e879f9', '#22d3ee']]
    const strands: THREE.ShaderMaterial[] = []
    palette.forEach((c, i) => {
      const mat = new THREE.ShaderMaterial({
        vertexShader: strandVert, fragmentShader: strandFrag, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { uTime: { value: 0 }, uPhase: { value: i / 3 }, uA: { value: new THREE.Color(c[0]) }, uB: { value: new THREE.Color(c[1]) }, uC: { value: new THREE.Color(c[2]) } },
      })
      strands.push(mat)
      rope.add(new THREE.Mesh(new THREE.TubeGeometry(new Braid((i * Math.PI * 2) / 3, 0.42), 600, 0.16, 20, false), mat))
      // a thin bright core inside each strand
      const core = new THREE.Mesh(new THREE.TubeGeometry(new Braid((i * Math.PI * 2) / 3, 0.42), 600, 0.035, 8, false),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(c[1]), transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }))
      rope.add(core)
    })

    const N = window.innerWidth < 700 ? 700 : 1600
    const pos = new Float32Array(N * 3)
    const seed = new Float32Array(N)
    for (let i = 0; i < N; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 26
      pos[i * 3 + 1] = (Math.random() - 0.5) * 14
      pos[i * 3 + 2] = (Math.random() - 0.5) * 10 - 2
      seed[i] = Math.random()
    }
    const pg = new THREE.BufferGeometry()
    pg.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    pg.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    const pm = new THREE.ShaderMaterial({ vertexShader: pointVert, fragmentShader: pointFrag, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uTime: { value: 0 }, uSize: { value: 2.6 * Math.min(window.devicePixelRatio, 2) }, uColor: { value: new THREE.Color('#c4b5fd') } } })
    scene.add(new THREE.Points(pg, pm))

    let wide = true
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      wide = w > 900
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      // keep the whole rope in view on narrow screens
      camera.position.z = w < h ? 17 : 11
      camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(el)

    const mouse = { x: 0, y: 0 }
    const onMove = (e: PointerEvent) => { mouse.x = e.clientX / window.innerWidth - 0.5; mouse.y = e.clientY / window.innerHeight - 0.5 }
    window.addEventListener('pointermove', onMove)

    const still = reducedMotion()
    const scroll = { turn: 0 }
    const st = still ? null : gsap.to(scroll, { turn: 1, ease: 'none', scrollTrigger: { trigger: el, start: 'top top', end: 'bottom top', scrub: 0.6 } })

    let visible = true
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting })
    io.observe(el)
    const clock = new THREE.Clock()
    let raf = 0
    const frame = () => {
      raf = requestAnimationFrame(frame)
      if (!visible) return
      const t = still ? 6 : clock.getElapsedTime()
      strands.forEach((m) => (m.uniforms.uTime.value = t))
      pm.uniforms.uTime.value = t
      rope.rotation.x = 0.15 + Math.sin(t * 0.15) * 0.05 + scroll.turn * 0.6
      rope.rotation.y = -0.35 + t * 0.03 + scroll.turn * 0.9
      rope.position.y = scroll.turn * 1.5 - (wide ? 0.6 : 0)
      rope.position.x = wide ? 2.6 : 0
      camera.position.x += (mouse.x * 1.2 - camera.position.x) * 0.04
      camera.position.y += (-mouse.y * 0.8 - camera.position.y) * 0.04
      camera.lookAt(0, 0, 0)
      renderer.render(scene, camera)
    }
    if (still) { renderer.render(scene, camera); frame() } else frame()

    return () => {
      cancelAnimationFrame(raf)
      st?.scrollTrigger?.kill(); st?.kill()
      ro.disconnect(); io.disconnect()
      window.removeEventListener('pointermove', onMove)
      scene.traverse((o) => {
        const m = o as THREE.Mesh
        m.geometry?.dispose()
        const mat = m.material as THREE.Material | undefined
        mat?.dispose()
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])
  return <div ref={host} className="rope-scene" aria-hidden />
}
