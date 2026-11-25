# Rope design system (master)

Source: ui-ux-pro-max (`--design-system "AI image generation creative tool dark premium"`, motion 6, density 5),
adapted: the suggested landing-page pattern does not fit a local app and was dropped; style, palette and type kept.
Tokens live in `src/styles.css` (`:root`); components use tokens only, never raw hex.

## Direction
- Style: AI-native, minimal chrome. The one large visual is the stage (the diffusion field, then the image).
- Honesty is the product: every time shown is rope's own estimate or measurement; say where the work runs (GPU / processor).
- Light on weak laptops: no motion while the CPU generates on a CPU-only machine (P5.1.5); no CDN, no web fonts from the network.

## Color (dark first; light by OS or toggle)
| token | dark | light | use |
|---|---|---|---|
| bg / surface / surface-2 | #0b0a12 / #14121f / #1c1930 | #faf7ff / #ffffff / #f3eefc | page, cards, raised |
| text / text-muted | #f2f0fa / #aaa4c4 | #0f172a / #4b5563 | body / secondary |
| primary / primary-strong | #8b5cf6 / #7c3aed | #7c3aed / #6d28d9 | focus, selection / primary button |
| accent | #ec4899 | #db2777 | generation only (progress gradient) |
| success / warning / danger | #34d399 / #fbbf24 / #f87171 | #047857 / #92400e / #b91c1c | time badges, errors |
Checked by `bench/tools/check_app.py`: every text pair >= 4.5:1 (worst: dark 5.70, light 4.82).

## Type
Inter Variable (UI) + JetBrains Mono 400/500 (every number: GiB, seconds, steps, seeds), both bundled with @fontsource.
Base 16 px / 1.5; labels 12 px uppercase +0.08em; h1 clamp(28-40 px) 650; plan time 44 px mono.

## Motion
150 / 240 / 420 ms, ease-out cubic-bezier(0.22, 1, 0.36, 1); page enter 320 ms, exit 180 ms (exit faster); transform and
opacity only. `MotionConfig reducedMotion="user"` + a CSS rule that sets all transitions to 0 ms under prefers-reduced-motion.

## Layout
Top bar (brand, Create / Gallery / This laptop, device chip, theme toggle). Create: prompt + quality + model list | sticky
plan card; running: stage + progress (stage name, step i of n, elapsed, ETA, Cancel); done: image + Save / New image.
Gallery: square tiles, lightbox `<dialog>` (Esc closes, focus to Close). Breakpoint 900 px: one column.

## Accessibility
Visible focus ring (2 px, ring token, 3 px offset); radiogroups for quality and model; progressbar with value; one polite
live region announces stage changes, not every tick; every icon has aria-hidden and the control has a text label.
