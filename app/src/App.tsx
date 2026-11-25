import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Cpu, Images, Moon, Sparkles, Sun } from 'lucide-react'
import { api, type Device } from './api'
import DeviceScreen from './screens/Device'
import CreateScreen from './screens/Create'
import GalleryScreen from './screens/Gallery'

type Route = 'device' | 'create' | 'gallery'
const routeOf = (h: string): Route => (h === '#/device' ? 'device' : h === '#/gallery' ? 'gallery' : 'create')

// enter slower than exit; transform + opacity only (no layout properties animated)
export const page = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.32, ease: [0.22, 1, 0.36, 1] as const } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.18 } },
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeOf(location.hash))
  const [device, setDevice] = useState<Device | null>(null)
  const [deviceError, setDeviceError] = useState<string | null>(null)
  const [theme, setTheme] = useState<'dark' | 'light' | null>(() => {
    try { return (localStorage.getItem('rope.theme') as 'dark' | 'light' | null) ?? null } catch { return null }
  })

  useEffect(() => {
    const on = () => setRoute(routeOf(location.hash))
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  useEffect(() => { api.device().then(setDevice, (e) => setDeviceError(String(e.message ?? e))) }, [])
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
    else delete document.documentElement.dataset.theme
    try { if (theme) localStorage.setItem('rope.theme', theme); else localStorage.removeItem('rope.theme') } catch { /* private window */ }
  }, [theme])

  const dark = theme ? theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches
  const nav: [Route, string, typeof Cpu][] = [['create', 'Create', Sparkles], ['gallery', 'Gallery', Images], ['device', 'This laptop', Cpu]]

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/create" aria-label="Rope home"><img src="/rope.svg" alt="" />rope</a>
        <nav className="nav" aria-label="Main">
          {nav.map(([r, label, Icon]) => (
            <a key={r} href={`#/${r}`} aria-current={route === r ? 'page' : undefined}><Icon size={18} aria-hidden />{label}</a>
          ))}
        </nav>
        <span className="spacer" />
        {device && (
          <span className="chip mono" title={device.cpu}>
            {device.ram_usable_gib.toFixed(1)} GiB RAM · {device.gpu ? `${device.gpu.vram_total_gib.toFixed(0)} GB GPU` : 'no GPU'}
          </span>
        )}
        <button className="icon-btn" onClick={() => setTheme(dark ? 'light' : 'dark')} aria-label={dark ? 'Use light theme' : 'Use dark theme'}>
          {dark ? <Sun size={18} aria-hidden /> : <Moon size={18} aria-hidden />}
        </button>
      </header>
      <main id="main">
        <AnimatePresence mode="wait">
          <motion.div key={route} {...page}>
            {route === 'device' && <DeviceScreen device={device} error={deviceError} />}
            {route === 'create' && <CreateScreen device={device} />}
            {route === 'gallery' && <GalleryScreen />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
