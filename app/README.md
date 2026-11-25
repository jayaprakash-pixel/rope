# Rope app

The local UI of Rope: prompt → honest plan → image with live progress → gallery. React + Vite + TypeScript, Framer Motion,
three.js (react-three-fiber). It talks only to `rope serve` on 127.0.0.1 (see `engine/rust/serve.rs`); nothing leaves the laptop
(the server sends a Content-Security-Policy that allows this origin only; fonts are bundled).

```
npm install
npm run build                                   # -> dist/
../build/rust-cuda/release/rope.exe serve --static app/dist     # from the repo root: http://127.0.0.1:4003
```

Development: `rope serve --dev-origin http://127.0.0.1:3005` and `npm run dev` (Vite on 127.0.0.1:3005, `/api` proxied to 4003).
Ports are registered in the Port Registry. Design rules: `design-system/MASTER.md`. Browser check: `bench/tools/check_app.py`.
