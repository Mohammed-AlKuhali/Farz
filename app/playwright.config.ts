import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  retries: 0,
  workers: 1,
  reporter: [["list"], ["json", { outputFile: "reports/e2e-results.json" }]],
  use: {
    baseURL: "http://localhost:4180",
    ...devices["Pixel 7"],
    browserName: "chromium",
    serviceWorkers: "allow",
  },
  webServer: {
    command: "npm run build && npx vite preview --port 4180 --strictPort",
    url: "http://localhost:4180",
    reuseExistingServer: true,
    timeout: 180_000,
  },
});
