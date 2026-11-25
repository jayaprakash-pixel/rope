// Pictures made by Rope on the laptop it is built on (HP Victus, Ryzen 5 5600H, 8 GB RAM, GTX 1650 4 GB).
// Seconds: the whole run, model already on disk. Filled from the showcase run log.
export type Shot = { file: string; prompt: string; model: string; secs: number }
export const SHOWCASE: Shot[] = [
  {
    "file": "xl-lighthouse.webp",
    "prompt": "a lighthouse on a cliff at dusk, violet sky, crashing waves, cinematic photo, soft light",
    "model": "SDXL-Turbo \u00b7 512",
    "secs": 16
  },
  {
    "file": "xl-fox.webp",
    "prompt": "a red fox sleeping in fresh snow, golden hour, shallow depth of field, wildlife photo",
    "model": "SDXL-Turbo \u00b7 512",
    "secs": 15
  },
  {
    "file": "xl-tea.webp",
    "prompt": "a cup of masala chai on a wooden table by a rainy window, warm cozy light, photo",
    "model": "SDXL-Turbo \u00b7 512",
    "secs": 15
  },
  {
    "file": "xl-city.webp",
    "prompt": "a neon city street at night in the rain, reflections, purple and pink lights, cinematic",
    "model": "SDXL-Turbo \u00b7 512",
    "secs": 15
  },
  {
    "file": "flux-robot.webp",
    "prompt": "a small friendly robot watering plants in a sunlit greenhouse, detailed, warm colors",
    "model": "FLUX.1-schnell Q4_K \u00b7 512",
    "secs": 115
  },
  {
    "file": "flux-mountains.webp",
    "prompt": "snowy mountains above a sea of clouds at sunrise, epic landscape photograph",
    "model": "FLUX.1-schnell Q4_K \u00b7 512",
    "secs": 114
  }
]
