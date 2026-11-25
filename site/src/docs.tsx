import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import Nav, { Footer } from './components/Nav'
import { DOWNLOAD, REPO, VERSION } from './site'

const TOC = [
  ['install', 'Install'], ['first', 'Your first picture'], ['quality', 'Quality, size and time'], ['models', 'Models'], ['mine', 'Your own model files'],
  ['addons', 'LoRA and upscaler'], ['steer', 'Previews and steering'], ['cli', 'Command line'], ['api', 'HTTP API and library'],
  ['trouble', 'Troubleshooting'], ['privacy', 'Privacy and safety'], ['uninstall', 'Uninstall'],
]

function Docs() {
  const [on, setOn] = useState('install')
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setOn(e.target.id)), { rootMargin: '-30% 0px -60% 0px' })
    TOC.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [])
  return (
    <>
      <Nav home={false} />
      <main id="main" className="wrap docs">
        <nav className="toc" aria-label="On this page">
          {TOC.map(([id, t]) => <a key={id} href={`#${id}`} className={on === id ? 'on' : ''}>{t}</a>)}
        </nav>
        <article className="doc">
          <div className="kicker">Documentation · v{VERSION}</div>
          <h1>Using <span className="grad-text">Rope</span></h1>
          <p>Rope makes AI pictures on your own laptop. It measures the laptop first, then plans every picture so it fits: the model, the picture size, the number of steps, the decoder — and the time it will take. This page explains how to install and use it.</p>

          <h2 id="install">Install</h2>
          <ol>
            <li><a href={DOWNLOAD}>Download Rope-Setup.exe</a> (Windows 10 or 11, 64-bit).</li>
            <li>Open it. Rope is not code-signed yet, so Windows can show <b>"Windows protected your PC"</b>. Click <b>More info</b>, then <b>Run anyway</b>. Company-managed laptops can block unsigned programs; then Rope cannot run there.</li>
            <li>The setup copies Rope to <code>%LOCALAPPDATA%\Programs\Rope</code>. It needs no administrator rights and changes nothing outside your user folder.</li>
            <li>It reads the laptop and shows the model that fits, with its download size. Answer <b>Y</b> to download it now, or <b>N</b> to download it later.</li>
            <li>Rope starts and opens in your browser. Later, start it from the Start menu: <b>Rope</b>.</li>
          </ol>
          <p>Models and pictures go to <code>%LOCALAPPDATA%\Rope</code>. When the setup finds an NVIDIA driver, it installs the version that uses the graphics card.</p>

          <h2 id="first">Your first picture</h2>
          <ol>
            <li>Open the <b>Create</b> screen. Type what you want to see, for example <code>a lighthouse on a cliff at dusk, cinematic photo</code>.</li>
            <li>Look at <b>Plan on this laptop</b>: it shows the time, the size and the memory before you start.</li>
            <li>Click <b>Generate</b> (or press Ctrl + Enter). The picture appears step by step and goes to the <b>Gallery</b>.</li>
          </ol>
          <p>A Rope window (a black console) stays open while Rope runs. Close it to stop Rope.</p>

          <h2 id="quality">Quality, size and time</h2>
          <table>
            <thead><tr><th>Quality</th><th>What Rope does</th></tr></thead>
            <tbody>
              <tr><td>Fast</td><td>Fewest steps and the tiny decoder (TAESD) where the model has one.</td></tr>
              <tr><td>Balanced</td><td>The model's normal steps. The full decoder if it takes under 60 seconds on this laptop.</td></tr>
              <tr><td>Best</td><td>More steps and the full decoder.</td></tr>
            </tbody>
          </table>
          <p>The size buttons show only the sizes that fit in this laptop's memory, with the time for each. When the laptop has less than 2 GB of RAM free, the plan adds 20% to the time and tells you to close other programs. On battery, the speed can differ a lot from the plan.</p>

          <h2 id="models">Models</h2>
          <p>Rope runs four model families: <b>Stable Diffusion 1.x</b>, <b>Stable Diffusion 2.x</b> (SD-Turbo), <b>SDXL</b> (SDXL-Turbo, SDXL-Lightning and fine-tunes) and <b>FLUX.1-schnell</b>. The catalog lists 17 models. Each one shows its download size, its licence, and "not tested on Rope yet" when it was checked only from its file header.</p>
          <p>To download a catalog model from the command line, open a terminal in the Rope program folder:</p>
          <pre>rope.exe get --models-dir "%LOCALAPPDATA%\Rope\models" --catalog catalog\models.json</pre>
          <p>The models are made by others. Read each model's licence before you use its pictures; some licences limit commercial use or certain content.</p>

          <h2 id="mine">Your own model files</h2>
          <p>Have a model file already? On the Create screen, under <b>Your own model files</b>, paste its full path (for example <code>D:\models\my-model.safetensors</code>) and click <b>Add</b>. Rope reads only the start of the file, so this takes seconds even for a big file. Then it shows one of these:</p>
          <ul>
            <li><b>It runs</b> — with the time, and how Rope reads the file: as it is, streamed from disk, or as a smaller copy (Q8_0 or Q4_K) made once at the first picture and kept.</li>
            <li><b>It cannot run</b> — with the reason: a LoRA or upscaler (put it in <code>models\addons</code>), only a part of a model (a VAE or text encoder), an architecture Rope does not run yet (FLUX-dev, SD3, video, text), or a <code>.ckpt</code>/<code>.pt</code> file (a Python pickle that can run hidden code: convert it to <code>.safetensors</code> first).</li>
          </ul>
          <p>The file stays where it is. <b>Remove</b> only takes it off the list.</p>

          <h2 id="addons">LoRA and upscaler</h2>
          <p>Put add-on files in <code>%LOCALAPPDATA%\Rope\models\addons</code>. A LoRA style file (kohya format, for SD 1.x/2.x or SDXL) appears under <b>Add-ons → LoRA style</b> with a strength slider; the app turns off a LoRA that does not fit the chosen model and says why. A Real-ESRGAN ×4 file enables <b>Upscale ×4</b>: a 512 picture becomes 2048, in tiles, inside the same memory limit.</p>

          <h2 id="steer">Previews and steering</h2>
          <p>While a picture is made, the app shows a small preview after each step. You can <b>pause</b> and <b>resume</b>, change the <b>prompt</b> (it is taken at the next step), and change the guidance on models that use it. <b>Cancel</b> stops the run within one step.</p>

          <h2 id="cli">Command line</h2>
          <pre>{`rope probe                 what this laptop has (CPU, RAM, GPU, disk, power)
rope suggest               every catalog model: runs here? time, size, memory
rope plan                  the plan for the best model on this laptop
rope get                   download and verify the planned model's files
rope image --prompt "..."  make a picture (add --model <file> for your own)
rope check-model <file>    can this laptop run this file, and how?
rope app                   start the app and open it in the browser
rope selftest              one test picture + a report (time and memory vs plan)`}</pre>

          <h2 id="api">HTTP API and library</h2>
          <p>The app talks to a local HTTP API (version 1, stable) on <code>127.0.0.1</code>. Other programs on the laptop can use it too: list models, start a job, follow its events, read the picture. Only requests from the laptop itself are answered. The reference is in <a href={`${REPO}/blob/main/docs/HTTP_API.md`} target="_blank" rel="noreferrer">docs/HTTP_API.md</a>, with an example client in Python.</p>
          <p>For programs that embed Rope, <b>librope</b> is a C library with a progress callback and cancel; see <code>engine/include/rope.h</code> and the C and Python examples in the repository.</p>

          <h2 id="trouble">Troubleshooting</h2>
          <table>
            <thead><tr><th>What you see</th><th>What to do</th></tr></thead>
            <tbody>
              <tr><td>"Windows protected your PC"</td><td>Click More info → Run anyway. Rope is not code-signed yet.</td></tr>
              <tr><td>"blocked by your organization" / Device Guard</td><td>The laptop is managed by a company and allows only approved programs. Rope cannot run there.</td></tr>
              <tr><td>The picture takes much longer than the plan</td><td>Close big programs (browser tabs, games) and connect the charger. Low free RAM and battery mode slow it down.</td></tr>
              <tr><td>A model shows "does not fit"</td><td>The plan says why (memory or disk). Choose a smaller model or size, or free disk space.</td></tr>
              <tr><td>The browser does not open</td><td>Open <code>http://127.0.0.1:4003</code> yourself (the Rope window shows the address).</td></tr>
            </tbody>
          </table>

          <h2 id="privacy">Privacy and safety</h2>
          <ul>
            <li>Rope works offline after the model download. It sends no prompts, pictures or usage data anywhere, and needs no account.</li>
            <li>Every model file is checked before first use: its SHA-256 against the catalog, and a safety scan of its contents.</li>
            <li>Rope opens only <code>.safetensors</code> and <code>.gguf</code> files, never Python pickles.</li>
            <li>The app server answers only the laptop itself, and only the app's own page can start jobs.</li>
          </ul>

          <h2 id="uninstall">Uninstall</h2>
          <p>Settings → Apps → <b>Rope</b> → Uninstall. Rope removes its program folder, the Start menu entry and the Apps entry, then asks whether to delete your pictures and the downloaded models (several GB). It never deletes your own model files.</p>
        </article>
      </main>
      <Footer />
    </>
  )
}

createRoot(document.getElementById('root')!).render(<StrictMode><Docs /></StrictMode>)
