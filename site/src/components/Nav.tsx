import { useEffect, useState } from 'react'
import { Download, Moon, Sun } from 'lucide-react'
import { Github } from './GithubIcon'
import { DOWNLOAD, REPO } from '../site'

export function useTheme() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try { return (localStorage.getItem('rope-theme') as 'dark' | 'light') || 'dark' } catch { return 'dark' }
  })
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('rope-theme', theme) } catch { /* private mode */ }
  }, [theme])
  return [theme, setTheme] as const
}

export default function Nav({ home = true }: { home?: boolean }) {
  const [theme, setTheme] = useTheme()
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 20)
    on(); window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  const h = (id: string) => (home ? `#${id}` : `/#${id}`)
  return (
    <header className={`nav ${scrolled ? 'scrolled' : ''}`}>
      <a className="skip" href="#main">Skip to content</a>
      <div className="wrap">
        <a className="logo" href="/"><img src="/rope.svg" alt="" />rope</a>
        <nav className="nav-links" aria-label="Main">
          <a href={h('how')}>How it works</a>
          <a href={h('planner')}>Will it run?</a>
          <a href={h('features')}>Features</a>
          <a href={h('install')}>Install</a>
          <a href="/docs.html">Docs</a>
          <a href={REPO} target="_blank" rel="noreferrer"><Github size={15} style={{ verticalAlign: -2 }} /> GitHub</a>
        </nav>
        <button className="icon-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
          {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
        </button>
        <a className="btn btn-primary btn-sm" href={DOWNLOAD}><Download size={15} /> Download</a>
      </div>
    </header>
  )
}

export function Footer() {
  return (
    <footer>
      <div className="wrap">
        <span>Rope · AI images on the laptop you already have · MIT licence</span>
        <span style={{ display: 'flex', gap: 22, flexWrap: 'wrap' }}>
          <a href="/docs.html">Documentation</a>
          <a href={REPO} target="_blank" rel="noreferrer">Source code</a>
          <a href={`${REPO}/releases`} target="_blank" rel="noreferrer">Releases</a>
          <a href={`${REPO}/issues`} target="_blank" rel="noreferrer">Report a problem</a>
        </span>
      </div>
    </footer>
  )
}
