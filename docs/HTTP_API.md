# Rope HTTP API, version 1

`rope serve` gives a local HTTP API. The app uses it. Any program can use it.

- Start: `rope serve [--port 4003] [--models-dir models] [--catalog catalog/models.json] [--gallery gallery] [--static app/dist] [--ram-gib X] [--no-gpu]`.
- Address: `http://127.0.0.1:<port>`. The server listens on the loopback address only.
- The `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`. Other values give `403`.
- A `POST` that carries an `Origin` header must carry the server's own origin (or the one given with `--dev-origin`). Other origins give `403`. A program without an `Origin` header is allowed.
- Bodies are JSON (`Content-Type: application/json`), at most 64 KiB. Every error is `{"error": "<text>"}` with a 4xx or 5xx status.
- One job runs at a time. A second `POST /api/jobs` gives `409` while a job runs.
- **Version rule.** `GET /api/health` returns `"api": 1`. Inside version 1, no route, field or meaning is removed or changed. New fields can appear: a client must ignore fields it does not know.

## Routes

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Is the server up. |
| `GET /api/device` | What this laptop is (CPU, RAM, GPU, disk). |
| `GET /api/models` | Every catalog model, whether it runs here, its plan. |
| `GET /api/addons` | LoRA and upscaler files in `models/addons`. |
| `POST /api/jobs` | Start one image. |
| `GET /api/jobs/{id}/events` | Server-sent events of the job. |
| `GET /api/jobs/{id}/image` | The finished PNG. |
| `GET /api/jobs/{id}/preview` | The latest live preview PNG. |
| `POST /api/jobs/{id}/control` | Pause, resume, change prompt or guidance. |
| `POST /api/jobs/{id}/cancel` | Stop the job. |
| `GET /api/gallery` | Finished images, newest first. |
| `GET /api/gallery/{name}.png` | One finished image. |

### GET /api/health
Response `200`: `{"ok": true, "api": 1, "version": "0.1.0"}`.

### GET /api/device
Response `200`: an object with `os`, `cpu`, `cores`, `threads`, `cpu_path` (`"avx2"` or `"sse41"`), `ram_usable_gib`, `ram_free_gib`, and when present `gpu` (`name`, `vram_total_gib`, `vram_free_gib`) and `disk` (`free_gib`, `ssd`, `bus`, `seq_read_gbs`). The answer is measured once per server run.

### GET /api/models
Query (all optional): `quality` = `fast` | `balanced` (default) | `best`; `size` = a multiple of 64 from 64 to 2048; `only` = one catalog model id; `upscale` = `1` (plan with the 4x upscaler stage).
Response `200`: `{"quality": "...", "device": {"gflops", "ram_gib", "gpu"}, "models": [Model, ...]}`.
Every `Model` has `id`, `kind` (`"image"`), `title`, `file`, `on_disk`, `runs`.
If `runs` is `false`: `supported` and `why` (the reason in plain words).
If `runs` is `true`: `size`, `steps`, `guidance`, `decoder` (`"taesd"` or `"vae"`), `estimate_s`, `upscale_s`, `peak_gib` (predicted peak private memory), `vae_tile`, `commit_gib`, `download_gib`, `on_gpu`, `cpu_estimate_s`, `max_size` (largest size this laptop can run), `max_estimate_s`, `max_native` (largest size of the model), `sizes` (`[{size, estimate_s, peak_gib, vae_tile}]`), `limit` (why the next size is refused, or `null`), `notes` (list of text).
Errors: `400` for a wrong `quality` or `size`.

### GET /api/addons
Response `200`: `{"addons": [{"name": "file.safetensors", "kind": "lora" | "esrgan", "title": "...", "bytes": 123}]}`. Rope detects the kind from the tensor names. Other files are not listed.

### POST /api/jobs
Request:
```json
{"model": "sdxl-turbo", "prompt": "a red bicycle", "quality": "fast",
 "seed": 1, "size": 512, "steps": 1,
 "lora": "pixel-art-xl.safetensors", "lora_scale": 1.0, "upscale": "realesrgan_x4plus.safetensors"}
```
Required: `model` (a catalog id whose file is on disk) and `prompt` (1 to 2000 characters). Optional: `quality` (default `balanced`), `seed` (default: from the clock), `size`, `steps`, `lora` and `upscale` (file names from `GET /api/addons`; the kind must match), `lora_scale` (0 to 2, default 1). A LoRA works with SD 1.x, SD 2.x and SDXL models. It does not work with FLUX yet: the job ends with an `error` event. With `upscale` the image is 4 times larger on each side.
Response `202`: `{"id": "<job id>", "seed": <seed used>}`.
Errors: `400` (bad body, prompt, quality, model, or add-on), `409` (model file not downloaded, or a job is running).

### GET /api/jobs/{id}/events
Server-sent events (`Content-Type: text/event-stream`). The stream starts at the first event of the job, so a late client misses nothing. It ends after `done` or `error`. Each message is `data: <JSON>`. The `type` field names the event:

| `type` | Fields | Meaning |
|---|---|---|
| `start` | `id`, `model`, `prompt`, `quality`, `seed`, `image`, and `lora`, `lora_scale`, `upscale` when used | The job began. `image` is the gallery path of the result. |
| `log` | `line` | A line of the engine's report (plan, notes, timings). |
| `estimate` | `total_s` | Planned time of the whole job. |
| `progress` | `stage` (`text`, `denoise`, `decode`, `upscale`), `i`, `n`, `elapsed_s`, `eta_s` | Step `i` of `n` in a stage. Pause time is not counted. |
| `preview` | `step`, `n` | A new preview exists at `/preview`. |
| `paused`, `resumed` | `step` | The job holds, or runs again. |
| `prompt` | `step` | A prompt change took effect before this step. |
| `done` | `id`, `model`, `prompt`, `seed`, `image`, `seconds` | The image is ready. |
| `error` | `message` | The job failed. The message is the engine's last line. |

### GET /api/jobs/{id}/image
Response `200`: `image/png`. Error `404` before the image exists or for another job id.

### GET /api/jobs/{id}/preview
Response `200`: `image/png`, a small estimate of the final image, updated at step boundaries (SD 1.x and 2.x models). Error `404` when there is no preview.

### POST /api/jobs/{id}/control
Request: `{"op": "pause"}`, `{"op": "resume"}`, `{"op": "prompt", "text": "..."}` (1 to 2000 characters) or `{"op": "cfg", "value": 3.5}` (0 to 30, only for models that use guidance). The job reads the command at its next step boundary. A change early in the run changes the image more.
Response `200`: `{"ok": true}`. Errors: `400` (bad op or value), `404` (no such job), `409` (job ended).

### POST /api/jobs/{id}/cancel
Response `200`: `{"ok": true}`. The job process stops within one step.

### GET /api/gallery
Response `200`: a list of records, newest first. Each: `id`, `model`, `prompt`, `quality`, `seed`, `image`, `seconds`, `created_ms`, and `lora`, `lora_scale`, `upscale` when used.

### GET /api/gallery/{name}.png
Response `200`: `image/png`. Errors: `400` (bad name), `404`.

## Example client
`examples/client.py` generates an image with this API only (Python standard library). See the check `bench/tools/check_api.py`: it starts a server, runs the client, and tests every route and error in this document.
