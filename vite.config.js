import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { defineConfig } from "vite";

const rootDir = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
    build: {
        rollupOptions: {
            input: {
                posts: resolve(rootDir, "index.html"),
                finds: resolve(rootDir, "finds.html"),
                experience: resolve(rootDir, "experience.html"),
                llmPretrainingLessons: resolve(rootDir, "llm-pretraining-lessons.html")
            }
        }
    }
});
