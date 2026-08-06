# Writing posts

Each post is one Markdown file in this folder. Its filename becomes its URL: `egocentric-pretraining.md` becomes `egocentric-pretraining.html`.

Start every post with this front matter:

```md
---
title: Pretraining on egocentric data
description: One or two sentences shown on the Posts page.
date: 2026-08-04
draft: true
---
```

Set `draft: true` while a post is in progress. Drafts are still generated so you can
preview them at their direct `.html` URL, but they are omitted from the Posts page.
Remove the line (or set it to `false`) when the post is ready to publish.

This is not access control: anyone who knows a draft's URL can still open it. A real
password-protected post requires authentication on the hosting platform; hiding it
with browser-side JavaScript would expose both the post and the password.

The title is rendered automatically, so begin the article itself with `##` for its first section heading.

Use normal Markdown for paragraphs, headings, lists, links, blockquotes, and fenced code blocks. Store image files in `assets/` and use a root-relative-to-the-page path, for example:

```md
![A robot wrist camera view](assets/egocentric-overview.svg)
*Figure 1. An egocentric observation.*
```

For equations, write MathJax notation:

```md
Inline: \(p(a_t \mid o_{\leq t})\)

Display:
\[
\mathcal{L} = -\sum_t \log p_\theta(a_t \mid o_{\leq t})
\]
```

After adding or editing a post, run:

```sh
npm run posts:build
```

This regenerates the post HTML pages and the list on the Posts homepage. Run `npm run build` before publishing to also verify the production site build.
