/* eslint-disable */
const { spawnSync } = require("child_process");

if (process.platform === "win32" && process.arch === "ia32") {
  console.warn("Warning: Biome does not support win32 ia32 platform. Skipping check.");
  process.exit(0);
}

const isWindows = process.platform === "win32";
const command = isWindows ? "npx.cmd" : "npx";
const args = ["biome", ...process.argv.slice(2)];

const result = spawnSync(command, args, { stdio: "inherit", shell: true });
process.exit(result.status ?? 0);
