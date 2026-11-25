// Facts the page states. One place, so the page and the README never disagree.
export const VERSION = '0.1.0'
export const REPO = 'https://github.com/jayaprakash-pixel/rope'
export const DOWNLOAD = `${REPO}/releases/latest/download/Rope-Setup.exe`
export const RELEASES = `${REPO}/releases`
export const SETUP_MB = '2.5'
// sha256 of the released Rope-Setup.exe (filled at release time by the build)
export const SETUP_SHA256 = '03d06ce8d36b8625b23545f6e5c29194e78063829e58fde51a5ac54ca3aee1f8'

export const reducedMotion = () => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
