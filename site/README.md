# Rope website

The public site: Vite + React + three.js (the hero rope), GSAP ScrollTrigger and Motion. Everything is bundled; no CDN at run time.

```
npm install
npm run dev        # http://localhost:5173
npm run build      # dist/ (two pages: index.html, docs.html)
python check_site.py http://127.0.0.1:4180   # after: npx vite preview --port 4180 --host 127.0.0.1
```

The planner numbers in `src/data/plans.json` are real `rope suggest --json` output for four example laptops.
