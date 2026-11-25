import { Cpu, HardDrive, MemoryStick, MonitorSmartphone } from 'lucide-react'
import type { Device } from '../api'

const Stat = ({ icon: Icon, label, value, sub }: { icon: typeof Cpu; label: string; value: string; sub: string }) => (
  <section className="card stack" aria-label={label}>
    <div className="row"><Icon size={18} aria-hidden /><span className="label">{label}</span></div>
    <div className="stat-value">{value}</div>
    <p className="stat-sub">{sub}</p>
  </section>
)

export default function DeviceScreen({ device, error }: { device: Device | null; error: string | null }) {
  if (error) return <div className="card error" role="alert">Rope could not read this laptop: {error}</div>
  if (!device) return <p className="muted" role="status">Checking this laptop…</p>
  const g = device.gpu
  return (
    <div className="stack">
      <div className="hero">
        <span className="label">This laptop</span>
        <h1>Rope checked your machine</h1>
        <p className="muted">These numbers decide how each model runs here: what stays in memory, what goes to the GPU and what is read from disk.</p>
      </div>
      <div className="grid grid-4">
        <Stat icon={Cpu} label="Processor" value={`${device.cores} cores`} sub={`${device.cpu} · ${device.cpu_path === 'avx2' ? 'AVX2 kernels' : 'SSE4.1 kernels (no AVX2)'}`} />
        <Stat icon={MemoryStick} label="Memory" value={`${device.ram_usable_gib.toFixed(1)} GiB`} sub={`${device.ram_free_gib.toFixed(1)} GiB free now`} />
        <Stat icon={MonitorSmartphone} label="Graphics" value={g ? `${g.vram_total_gib.toFixed(0)} GB` : 'No GPU'}
              sub={g ? `${g.name} · ${g.vram_free_gib.toFixed(1)} GB free` : 'Everything runs on the processor'} />
        <Stat icon={HardDrive} label="Disk" value={device.disk ? `${device.disk.free_gib.toFixed(0)} GiB free` : '—'}
              sub={device.disk ? `${device.disk.ssd ? device.disk.bus || 'SSD' : 'HDD'} · ${device.disk.seq_read_gbs.toFixed(1)} GB/s read` : ''} />
      </div>
      <div className="row"><a className="btn primary" href="#/create">Start creating</a></div>
    </div>
  )
}
