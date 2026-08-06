import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { basename, dirname, extname, resolve } from "node:path";
import { defineConfig } from "vite";

const rootDir = dirname(fileURLToPath(import.meta.url));
const postsDir = resolve(rootDir, "posts");
const postInputs = Object.fromEntries(
    readdirSync(postsDir, { withFileTypes: true })
        .filter((entry) => entry.isFile() && extname(entry.name) === ".md" && !entry.name.startsWith("_"))
        .map((entry) => {
            const slug = basename(entry.name, ".md");
            return [slug, resolve(rootDir, `${slug}.html`)];
        })
);

export default defineConfig({
    build: {
        rollupOptions: {
            input: {
                posts: resolve(rootDir, "index.html"),
                finds: resolve(rootDir, "finds.html"),
                experience: resolve(rootDir, "experience.html"),
                ...postInputs
            }
        }
    }
});
