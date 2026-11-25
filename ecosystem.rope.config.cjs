// PM2 process for `rope serve` (the app's local API + the built app), 127.0.0.1:4003 (Port Registry).
// Runs a deployed copy (bin/rope.exe, made by pm2-rope.ps1) so cargo can rebuild while the service runs.
// PM2 runs as SYSTEM: every path is absolute and nothing depends on per-user environment variables.
const root = __dirname.split(require("path").sep).join("/");
module.exports = {
  apps: [{
    name: "rope-serve",
    script: `${root}/bin/rope.exe`,
    args: ["serve", "--port", "4003", "--static", "app/dist", "--gallery", "gallery", "--models-dir", "models", "--catalog", "catalog/models.json"],
    interpreter: "none",
    cwd: root,
    autorestart: true,
    max_restarts: 10,
    out_file: `${root}/bin/rope-serve.out.log`,
    error_file: `${root}/bin/rope-serve.err.log`,
    env: { PORT: "4003" },
  }],
};
