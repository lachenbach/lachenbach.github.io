import { readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, extname, basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDirectory, "..");
const postsDirectory = resolve(projectRoot, "posts");
const homepagePath = resolve(projectRoot, "index.html");

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function parseFrontMatter(source, filename) {
    const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);

    if (!match) {
        throw new Error(`${filename} needs a front matter block between --- lines.`);
    }

    const metadata = {};

    for (const line of match[1].split(/\r?\n/)) {
        if (!line.trim() || line.trimStart().startsWith("#")) {
            continue;
        }

        const separatorIndex = line.indexOf(":");
        if (separatorIndex === -1) {
            throw new Error(`Invalid front matter in ${filename}: ${line}`);
        }

        const key = line.slice(0, separatorIndex).trim();
        const rawValue = line.slice(separatorIndex + 1).trim();
        metadata[key] = rawValue.replace(/^(["'])(.*)\1$/, "$2");
    }

    for (const key of ["title", "description", "date", "kicker", "status"]) {
        if (!metadata[key]) {
            throw new Error(`${filename} is missing required front matter: ${key}`);
        }
    }

    return { metadata, body: match[2] };
}

function formatDate(isoDate) {
    const date = new Date(`${isoDate}T12:00:00Z`);

    if (Number.isNaN(date.valueOf())) {
        throw new Error(`Invalid date: ${isoDate}. Use YYYY-MM-DD.`);
    }

    return new Intl.DateTimeFormat("en-US", {
        month: "long",
        day: "numeric",
        year: "numeric",
        timeZone: "UTC"
    }).format(date);
}

function protectDelimitedMath(source, opening, closing, tokens) {
    let result = "";
    let cursor = 0;
    let start = source.indexOf(opening, cursor);

    while (start !== -1) {
        const end = source.indexOf(closing, start + opening.length);

        if (end === -1) {
            break;
        }

        const formula = source.slice(start, end + closing.length);
        const token = `MATH_TOKEN_${tokens.length}_END`;
        tokens.push({ token, formula });
        result += `${source.slice(cursor, start)}${token}`;
        cursor = end + closing.length;
        start = source.indexOf(opening, cursor);
    }

    return `${result}${source.slice(cursor)}`;
}

function renderMarkdown(source) {
    const tokens = [];
    const displayProtected = protectDelimitedMath(source, String.raw`\[`, String.raw`\]`, tokens);
    const protectedSource = protectDelimitedMath(displayProtected, String.raw`\(`, String.raw`\)`, tokens);
    let html = marked.parse(protectedSource);

    for (const { token, formula } of tokens) {
        html = html.replaceAll(token, formula);
    }

    return html;
}

const mathJaxScript = String.raw`    <script>
        window.MathJax = {
            tex: {
                inlineMath: [["\\(", "\\)"]],
                displayMath: [["\\[", "\\]"]]
            }
        };
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>`;

function renderPostPage(post) {
    const title = escapeHtml(post.title);
    const description = escapeHtml(post.description);
    const kicker = escapeHtml(post.kicker);
    const status = escapeHtml(post.status);

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="${description}">
    <title>Liam | ${title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="styles.css">
${mathJaxScript}
</head>
<body>
    <div class="page-shell">
        <header class="site-header">
            <div class="brand-group">
                <a class="brand" href="index.html">LiamLog</a>
                <span class="brand-meta">73|76</span>
            </div>

            <nav class="site-nav" aria-label="Primary navigation">
                <a class="is-active" href="index.html">Posts</a>
                <a href="finds.html">Finds</a>
                <a href="experience.html">Experience</a>
            </nav>
        </header>

        <main class="content">
            <article class="blog-post">
                <p class="section-kicker">${kicker}</p>
                <h1>${title}</h1>
                <p class="post-meta">${post.displayDate} | ${status}</p>

                <div class="blog-body">
${post.htmlBody.trim().split("\n").map((line) => `                    ${line}`).join("\n")}
                </div>
            </article>
        </main>
    </div>
</body>
</html>
`;
}

function renderPostCard(post) {
    return `                <a class="post-card post-link" href="${post.slug}.html">
                    <h2>${escapeHtml(post.title)}</h2>
                    <p>${escapeHtml(post.description)}</p>
                </a>`;
}

async function loadPosts() {
    const filenames = await readdir(postsDirectory);
    const postFilenames = filenames.filter((filename) =>
        extname(filename) === ".md" && !basename(filename).startsWith("_")
    );

    const posts = await Promise.all(postFilenames.map(async (filename) => {
        const source = await readFile(resolve(postsDirectory, filename), "utf8");
        const { metadata, body } = parseFrontMatter(source, filename);
        const slug = basename(filename, ".md");

        if (!/^[a-z0-9-]+$/.test(slug)) {
            throw new Error(`${filename} must use a lowercase, hyphenated filename.`);
        }

        return {
            ...metadata,
            slug,
            displayDate: formatDate(metadata.date),
            htmlBody: renderMarkdown(body)
        };
    }));

    return posts.sort((first, second) => second.date.localeCompare(first.date));
}

async function buildPosts() {
    const posts = await loadPosts();
    const homepage = await readFile(homepagePath, "utf8");
    const cards = posts.map(renderPostCard).join("\n");
    const updatedHomepage = homepage.replace(
        /<!-- POSTS:START -->[\s\S]*?<!-- POSTS:END -->/,
        `<!-- POSTS:START -->\n${cards}\n                <!-- POSTS:END -->`
    );

    if (updatedHomepage === homepage && !homepage.includes("<!-- POSTS:START -->")) {
        throw new Error("Could not find the Posts section markers in index.html.");
    }

    await Promise.all([
        writeFile(homepagePath, updatedHomepage),
        ...posts.map((post) => writeFile(
            resolve(projectRoot, `${post.slug}.html`),
            renderPostPage(post)
        ))
    ]);

    console.log(`Built ${posts.length} post${posts.length === 1 ? "" : "s"}.`);
}

buildPosts().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
});
