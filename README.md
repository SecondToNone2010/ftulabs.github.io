# FTU Labs Website

Static website for [FTU Labs](https://ftulabs.github.io), hosted on GitHub Pages.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Contributing Guide](#contributing-guide)
  - [1. Clone the Repo](#1-clone-the-repo)
  - [2. Create a Branch](#2-create-a-branch)
  - [3. Write Clean Code](#3-write-clean-code)
  - [4. Push and Open a Pull Request](#4-push-and-open-a-pull-request)
  - [5. After Your PR Is Merged](#5-after-your-pr-is-merged)
- [Writing Blog Posts](#writing-blog-posts)
  - [Option A: Write in Markdown (Recommended)](#option-a-write-in-markdown-recommended)
  - [Option B: Write HTML Manually](#option-b-write-html-manually)
- [Adding Other Content](#adding-other-content)
  - [Team Members](#team-members)
  - [Projects](#projects)
  - [Research Papers](#research-papers)
- [Media](#media)
  - [Images](#images)
  - [Videos](#videos)
  - [Audio](#audio)
  - [Other Embeds](#other-embeds)
  - [Media Guidelines](#media-guidelines)
- [Code Blocks & Syntax Highlighting](#code-blocks--syntax-highlighting)
- [Localization (l10n)](#localization-l10n)
  - [How It Works](#how-it-works)
  - [Translate an Existing Page](#translate-an-existing-page)
  - [Add a New Language](#add-a-new-language)
- [Tooling & Automation](#tooling--automation)
  - [Markdown-to-Post Converter](#markdown-to-post-converter)
  - [Cache Busting](#cache-busting)
  - [Vendor Library Updates](#vendor-library-updates)
- [Theme & Customization](#theme--customization)
- [Development](#development)
- [Deployment](#deployment)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ftulabs/ftulabs.github.io.git
cd ftulabs.github.io

# 2. Preview locally
python3 -m http.server 8000        # then open http://localhost:8000

# 3. Write a blog post in Markdown (see blog/post-template.md for an example)
python3 scripts/md2post.py blog/my-post.md

# 4. Commit and push
git add -A
git commit -m "blog: Add my new post"
git push origin main
```

That's it. The GitHub Actions workflow handles cache-busting automatically on push — no extra steps needed.

---

## Project Structure

```
ftulabs.github.io/
├── index.html                  # Home page
├── blog.html                   # Blog listing page
├── projects.html               # Projects page
├── team.html                   # Team members page
├── research.html               # Research publications page
│
├── blog/                       # Blog posts
│   ├── 1.ftu-ai.html
│   ├── 2.lms.html
│   ├── 3.swarm.html
│   ├── 4.introduction.html
│   ├── post-template.html      # HTML reference template
│   └── post-template.md        # Markdown reference template
│
├── css/
│   └── style.css               # Global stylesheet
├── js/
│   ├── main.js                 # Nav toggle, scroll-reveal animations
│   ├── hljs.js                 # Syntax highlighting + copy button injection
│   └── l10n.js                 # Language switcher & auto-detection
├── img/                        # All images (root-relative paths)
├── vendor/
│   └── hljs/                   # Highlight.js core + language modules
│
├── vi/                         # Vietnamese translations (mirrors root)
│   ├── index.html
│   ├── blog.html
│   ├── blog/
│   └── …
├── l10n/
│   └── manifest.json           # Maps pages → available translations
│
├── scripts/
│   ├── cachebust.sh            # Asset version stamper (content-hash)
│   ├── md2post.py              # Markdown → blog post converter
│   └── update_vendor.py        # Vendor library updater
│
└── .github/workflows/
    ├── cachebust.yml           # Auto-stamps CSS/JS versions on push
    └── update-vendor-libs.yml  # Weekly vendor library update
```

---

## Contributing Guide

This workflow mirrors how professional teams ship code — learning it now will serve you in internships, open-source projects, and your career.

### 1. Clone the Repo

```bash
git clone https://github.com/ftulabs/ftulabs.github.io.git
cd ftulabs.github.io
```

Already cloned? Make sure you're up to date:

```bash
git checkout main
git pull origin main
```

### 2. Create a Branch

**Never work directly on `main`.** Create a descriptive branch for your change:

```bash
git checkout -b <type>/<short-description>
```

| Type        | When to use                        | Example                           |
| ----------- | ---------------------------------- | --------------------------------- |
| `feature/`  | Adding something new               | `feature/add-member-nguyen-van-a` |
| `fix/`      | Fixing a bug or typo               | `fix/broken-nav-link`             |
| `blog/`     | Adding or editing a blog post      | `blog/intro-to-ml-post`          |
| `docs/`     | Updating documentation             | `docs/update-readme`              |
| `refactor/` | Restructuring without new features | `refactor/css-variables`          |
| `l10n/`     | Translating a page to a new locale | `l10n/vi/blog/4.introduction`     |

> **Why branches?** They isolate your work so a half-finished change never breaks the live site. They also make code review possible — your teammates can see exactly what changed.

### 3. Write Clean Code

#### Use Meaningful Names

```html
<!-- ❌ Bad -->
<div class="x">
  <div class="a">stuff</div>
</div>

<!-- ✅ Good -->
<div class="member">
  <div class="member-name">Minh Tran</div>
</div>
```

#### Comment the "Why", Not the "What"

```html
<!-- ❌ Bad: just restates the code -->
<!-- This is a paragraph -->
<p>Welcome to FTU Labs.</p>

<!-- ✅ Good: explains intent -->
<!-- Shown only on the home page; blog.html has its own intro -->
<p>Welcome to FTU Labs.</p>
```

For complex or collapsible sections, use HTML comments as markers:

```html
<!-- ============================================ -->
<!-- OLYMPIC TIN HOC -->
<!-- ============================================ -->
<h2>Olympic Tin học 2025</h2>

<details>
  <summary>Show detailed results</summary>
  <p>…</p>
</details>
```

#### Keep It Consistent

- Indent with **4 spaces** (not tabs).
- Use **double quotes** for HTML attributes.
- Use **root-relative paths** for assets: `/img/logo.png`, `/css/style.css`.
- Match the style of surrounding code — if a file uses a certain pattern, follow it.

#### Keep Commits Small and Focused

Each commit should do **one thing** and have a clear message.

```bash
# ❌ Bad
git commit -m "changes"
git commit -m "update stuff"

# ✅ Good — follows the project's convention:
git commit -m "blog: Add Docker Swarm cluster post"
git commit -m "css: Fix code block scrollbar spacing"
git commit -m "l10n(vi): Translate team page"
```

Commit message format: `category: Short description`

| Category    | Use for                              |
| ----------- | ------------------------------------ |
| `blog`      | Blog post content                    |
| `site`      | Cross-cutting site changes           |
| `css`       | Stylesheet changes                   |
| `js`        | JavaScript changes                   |
| `feature`   | New features                         |
| `fix`       | Bug fixes                            |
| `l10n`      | Localization / translations          |
| `l10n(vi)`  | Vietnamese-specific translation      |
| `docs`      | Documentation (README, comments)     |
| `chore`     | Maintenance / tooling                |

### 4. Push and Open a Pull Request

```bash
git push origin feature/add-member-nguyen-van-a
```

Then open a Pull Request on GitHub. In the PR description, explain **what** you changed and **why**.

#### PR Checklist

- [ ] Branch is up to date with `main`
- [ ] Changes preview correctly in a local browser
- [ ] No broken links or missing images
- [ ] Commit messages follow the convention
- [ ] Code is clean (no debug logs, commented-out junk, etc.)
- [ ] If adding a blog post, it appears in `blog.html`
- [ ] If translating, `l10n/manifest.json` is updated

### 5. After Your PR Is Merged

```bash
git checkout main
git pull origin main
git branch -d feature/add-member-nguyen-van-a    # delete local branch
```

#### Quick Reference

```
main (protected — always deployable)
 └── feature/add-member-nguyen-van-a   ← your work happens here
      ├── commit: "Add member photo"
      ├── commit: "Add member bio and links"
      └── → Open PR → Code review → Merge → Done
```

---

## Writing Blog Posts

### Option A: Write in Markdown (Recommended)

The easiest way. Write a `.md` file, run the converter, and it generates the full HTML page + updates the blog listing automatically.

#### 1. Create a Markdown file

Create a file anywhere (e.g. `blog/my-post.md`) with front matter at the top:

```markdown
<!-- #!ftulabs-scripts
title: My Post Title
description: A short summary shown in the blog listing.
date: 2026-04-01
authors: Alice, Bob
readtime: 8 min
lang: en
-->

Your content here in standard Markdown…
```

| Field         | Required | Description                                          |
| ------------- | -------- | ---------------------------------------------------- |
| `title`       | ✅       | Post title (used in `<title>`, heading, listing)     |
| `description` | ✅       | One-line summary (used in `<meta>` and blog listing) |
| `date`        | ✅       | Publication date in `YYYY-MM-DD` format              |
| `authors`     | ✅       | Comma-separated author names                         |
| `readtime`    | ✅       | Estimated reading time (e.g. `8 min`)                |
| `lang`        | ✅       | `en` for English, `vi` for Vietnamese                |

#### 2. Write the body in Markdown

All standard Markdown features are supported:

| Syntax                             | Renders as                        |
| ---------------------------------- | --------------------------------- |
| `## Heading`                       | `<h2>` through `<h6>`            |
| `**bold**`                         | **bold**                          |
| `*italic*`                         | *italic*                          |
| `` `code` ``                       | `code`                            |
| `[text](url)`                      | link                              |
| `![alt](src)` (first in post)      | `<figure>` with caption           |
| `![alt](src)` (subsequent)         | `<img>`                           |
| ` ```lang … ``` `                  | Syntax-highlighted code block     |
| `> quote`                          | Blockquote                        |
| `* item` / `- item`               | Unordered list                    |
| `1. item`                          | Ordered list                      |
| `---`                              | Horizontal rule                   |
| `~~strike~~`                       | ~~strikethrough~~                 |
| `| table |`                        | Table (with alignment)            |
| Raw HTML                           | Passed through unchanged          |

See [`blog/post-template.md`](blog/post-template.md) for a full working example.

#### 3. Run the converter

```bash
# Try it with the example file first:
python3 scripts/md2post.py blog/post-template.md --dry-run

# Then convert your own post:
python3 scripts/md2post.py blog/my-post.md
```

This will:
- Generate `blog/5.my-post.html` (auto-numbered, slug from filename)
- Insert a date-ordered entry into `blog.html`
- Detect code blocks and include syntax highlighting assets automatically

**Options:**

```bash
# Override the post number
python3 scripts/md2post.py blog/my-post.md --number 5

# Override the slug (filename portion)
python3 scripts/md2post.py blog/my-post.md --slug custom-slug

# Preview without writing anything
python3 scripts/md2post.py blog/my-post.md --dry-run

# Vietnamese post (detected from lang: vi in front matter)
python3 scripts/md2post.py vi/blog/my-post.md
```

> **Note:** The slug is derived from the markdown **filename**, not the title. Name your `.md` file with the slug you want (e.g. `cuda-kernels.md` → `5.cuda-kernels.html`). Use `--slug` only if you need to override this.

#### 4. Commit and push

```bash
git add blog/5.my-post.html blog.html
git commit -m "blog: Add my new post"
git push
```

### Option B: Write HTML Manually

If you prefer working directly in HTML or need more control over the markup:

1. **Copy the template:** Copy `blog/post-template.html` to a new file (e.g. `blog/5.my-post.html`).
2. **Edit the metadata:**
   - `<title>` and `<meta name="description">` in `<head>`
   - `<h1 class="post-title">` — the post title
   - `<div class="post-meta">` — date, authors, read time
3. **Write content** inside `<article class="post-content">` using standard HTML tags (`<p>`, `<h2>`, `<pre><code>`, `<blockquote>`, `<ul>`, `<ol>`, etc.).
4. **Add to the listing** — open `blog.html` and insert a new entry inside `<div class="blog-list">`:

   ```html
   <div class="blog-item reveal">
     <div class="blog-date">2026-04-01</div>
     <h2 class="blog-title"><a href="blog/5.my-post.html">Post Title</a></h2>
     <p class="blog-excerpt">A short description of the post.</p>
   </div>
   ```

   Entries are ordered newest-first by date.

5. **If using code blocks**, add the syntax highlighting assets. In `<head>` (before `style.css`):

   ```html
   <link rel="stylesheet" href="/vendor/hljs/atom-one-dark.min.css">
   ```

   Before `</body>` (after `main.js`):

   ```html
   <script src="/js/hljs.js" defer></script>
   ```

---

## Adding Other Content

### Team Members

Open `team.html` and add a `<div class="member reveal">` block inside `<div class="team-grid">`:

```html
<div class="member reveal">
  <div class="member-avatar">XX</div>       <!-- Initials (2 chars) -->
  <div class="member-name">Full Name</div>
  <div class="member-role">Role / Title</div>
  <p class="member-bio">Short bio describing expertise and background.</p>
  <div class="member-links">
    <a href="https://github.com/username">GitHub</a>
    <a href="https://twitter.com/username">Twitter</a>
    <a href="https://scholar.google.com/...">Scholar</a>
  </div>
</div>
```

### Projects

Open `projects.html` and add a `<div class="card reveal">` block inside the appropriate `<div class="card-grid cols-2">` section (active or completed):

```html
<div class="card reveal">
  <div class="status mb-1"><span class="status-dot"></span> active</div>
  <h3 class="card-title"><a href="#">Project Name</a></h3>
  <p class="card-desc">Short description of the project.</p>
  <div class="card-meta">
    <span class="tag active">Category</span>
    <span class="tag">Tech</span>
    <span class="tag">Stack</span>
  </div>
</div>
```

### Research Papers

Open `research.html` and add a `<div class="paper reveal">` block inside the appropriate year's `<div class="research-list">`:

```html
<div class="paper reveal">
  <h3 class="paper-title"><a href="#">Paper Title</a></h3>
  <p class="paper-authors">Author 1, Author 2, Author 3</p>
  <p class="paper-venue">Conference / Journal Name, Year</p>
  <div class="paper-links">
    <a href="#">[paper]</a>
    <a href="#">[code]</a>
    <a href="#">[arxiv]</a>
  </div>
</div>
```

To add a new year, create a new `<div class="research-year">` above a new `<div class="research-list">`.

---

## Media

### Images

Place files in `/img/`. Images inside `<article class="post-content">` are automatically styled (full-width, bordered). Always use root-relative paths:

```html
<!-- Basic image -->
<img src="/img/my-diagram.png" alt="Description of the image">

<!-- Image with caption -->
<figure>
  <img src="/img/results-chart.png" alt="Benchmark results">
  <figcaption>Figure 1: Inference latency comparison.</figcaption>
</figure>
```

In Markdown posts, use `![alt text](/img/my-image.jpg)`. The first image becomes a `<figure>` with a caption automatically.

### Videos

```html
<!-- Self-hosted -->
<video controls width="100%">
  <source src="../assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<!-- YouTube -->
<iframe width="100%" height="400" src="https://www.youtube.com/embed/VIDEO_ID"
  frameborder="0" allowfullscreen></iframe>

<!-- Vimeo -->
<iframe width="100%" height="400" src="https://player.vimeo.com/video/VIDEO_ID"
  frameborder="0" allowfullscreen></iframe>
```

Prefer YouTube/Vimeo embeds over committing large video files to the repo.

### Audio

```html
<audio controls>
  <source src="../assets/podcast-episode.mp3" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>
```

Supported formats: `.mp3`, `.ogg`, `.wav`.

### Other Embeds

```html
<!-- PDF -->
<embed src="../assets/paper-draft.pdf" type="application/pdf" width="100%" height="600px">

<!-- CodePen / Observable -->
<iframe src="https://codepen.io/user/embed/pen-id" width="100%" height="400"
  frameborder="0" allowfullscreen></iframe>

<!-- Google Slides -->
<iframe src="https://docs.google.com/presentation/d/SLIDE_ID/embed"
  width="100%" height="400" frameborder="0" allowfullscreen></iframe>
```

### Media Guidelines

| Guideline         | Details                                                                              |
| ----------------- | ------------------------------------------------------------------------------------ |
| **File location** | Store all media in `/img/` at the root                                               |
| **File size**     | Keep under 10 MB. Use external hosting for large files                               |
| **Alt text**      | Always include descriptive `alt` attributes for accessibility                        |
| **Responsive**    | Images and videos use `max-width: 100%` automatically                                |
| **Paths**         | Use root-relative paths starting with `/` (e.g. `/img/logo.png`)                    |
| **Formats**       | Images: `.png`, `.jpg`, `.webp`, `.svg`, `.gif` — keep optimized for web (< 500 KB) |

---

## Code Blocks & Syntax Highlighting

Blog posts use [Highlight.js](https://highlightjs.org/) for automatic syntax highlighting. The script `js/hljs.js` handles loading, highlighting, and injecting a **copy-to-clipboard button** into every code block.

> **If you use the Markdown converter (`md2post.py`), all of this is handled automatically.** The sections below are for manual HTML authors.

### Setup

In `<head>`, **before** `style.css`:

```html
<link rel="stylesheet" href="/vendor/hljs/atom-one-dark.min.css">
<link rel="stylesheet" href="/css/style.css">
```

Before `</body>`, **after** `main.js`:

```html
<script src="/js/main.js" defer></script>
<script src="/js/hljs.js" defer></script>
```

### Writing Code Blocks

Use `<pre><code class="language-xxx">` where `xxx` is the language — equivalent to ` ```xxx ` in Markdown:

| Markdown            | HTML                                      |
| ------------------- | ----------------------------------------- |
| ` ```bash `         | `<pre><code class="language-bash">`       |
| ` ```python `       | `<pre><code class="language-python">`     |
| ` ```javascript `   | `<pre><code class="language-javascript">` |
| ` ```yaml `         | `<pre><code class="language-yaml">`       |
| ` ```dockerfile `   | `<pre><code class="language-dockerfile">` |
| ` ```latex `        | `<pre><code class="language-latex">`      |
| ` ```ini `          | `<pre><code class="language-ini">`        |
| ` ```plaintext `    | `<pre><code class="language-plaintext">`  |

Example:

```html
<pre><code class="language-python"># This will be syntax-highlighted as Python
def hello(name: str) -> str:
    return f"Hello, {name}!"</code></pre>
```

> **Tip:** Use `language-plaintext` for code blocks that shouldn't be highlighted (ASCII diagrams, command output). Omitting the class entirely causes Highlight.js to auto-detect, which may produce unexpected results.

### Copy Button

Every `<pre>` block automatically gets a copy-to-clipboard button (top-right corner). No extra markup needed. This is handled entirely by `js/hljs.js`, with styling in `css/style.css` under `.copy-btn`.

### Adding Languages

The default Highlight.js build covers ~40 common languages. Extra languages (LaTeX, Dockerfile, INI) are bundled in `js/hljs.js`. To add more, edit the `LANGUAGES` array in that file. See the [full language list](https://highlightjs.org/download).

---

## Localization (l10n)

The site supports multiple languages. English is the default; translations live under a language prefix (e.g. `vi/` for Vietnamese). A language switcher appears in the navigation bar.

### How It Works

| Concept                  | Details                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Default language**     | English — pages at the root (`blog/4.introduction.html`)                                               |
| **Translated pages**     | Mirrored under a prefix (`vi/blog/4.introduction.html`)                                                |
| **Manifest**             | `l10n/manifest.json` — maps each base path to available language codes                                 |
| **Switcher JS**          | `js/l10n.js` — reads the manifest, builds a dropdown with flags                                       |
| **Auto-detection**       | First visit checks `navigator.language`. If a translation exists, the user is redirected. Stored in `localStorage` (`ftu-lang`). |
| **Greyed-out languages** | If the current page has no translation for a language, that flag is disabled                            |

### Translate an Existing Page

1. **Create a branch:**

   ```bash
   git checkout -b l10n/vi/blog/4.introduction
   ```

2. **Copy the English page:**

   ```bash
   mkdir -p vi/blog
   cp blog/4.introduction.html vi/blog/4.introduction.html
   ```

3. **Edit the translated file:**
   - Change `<html lang="en">` to `<html lang="vi">`
   - **Keep root-relative paths** for assets — they don't change:
     ```html
     <link rel="stylesheet" href="/css/style.css">
     <script src="/js/main.js"></script>
     <img src="/img/logo.png">
     ```
   - **Keep relative paths** for page navigation so the user stays in the same language:
     ```html
     <!-- ✅ Stays in Vietnamese -->
     <a href="../index.html">Trang chủ</a>

     <!-- ❌ Jumps to English — avoid! -->
     <a href="/index.html">Trang chủ</a>
     ```
   - Translate all user-facing text (title, headings, paragraphs, etc.)
   - **Do NOT translate** code blocks, terminal commands, URLs, or common English technical terms (Git, IDE, CUDA, etc.)
   - Keep the same HTML structure (classes, IDs, nesting)

4. **Register the translation** in `l10n/manifest.json`:

   ```json
   {
     "blog/4.introduction.html": ["en", "vi"],
     "blog/post-template.html": ["en", "vi"]
   }
   ```

5. **Commit and open a PR:**

   ```bash
   git add vi/blog/4.introduction.html l10n/manifest.json
   git commit -m "l10n(vi): Translate blog/4.introduction"
   git push origin l10n/vi/blog/4.introduction
   ```

### Add a New Language

1. Register the language in `js/l10n.js` — add an entry to the `LANGUAGES` array with `code`, `label`, and `flag` SVG.
2. Create the directory structure (`ja/`, `ja/blog/`, etc.).
3. Translate pages and update `l10n/manifest.json`.

---

## Tooling & Automation

### Markdown-to-Post Converter

**`scripts/md2post.py`** — Converts a Markdown file into a fully-formed blog post HTML page and updates the blog listing.

```bash
python3 scripts/md2post.py blog/my-post.md
```

What it does:

1. Parses the front matter for metadata (title, date, authors, etc.)
2. Converts Markdown body → HTML matching the site's template
3. Writes `blog/{N}.{slug}.html` (auto-numbered, slug from filename)
4. Inserts a date-ordered entry into `blog.html` (or `vi/blog.html` for `lang: vi`)
5. Includes hljs assets only when code blocks are present

Options:

| Flag              | Description                              |
| ----------------- | ---------------------------------------- |
| `--number N`, `-n`| Override the auto-assigned post number   |
| `--slug NAME`, `-s`| Override the slug (default: filename)   |
| `--dry-run`       | Preview what would happen without writing files |

**Requirements:** Python 3.6+ (standard library only — no pip install needed).

### Cache Busting

When CSS or JS files are updated, returning visitors may see stale cached versions. The site uses **per-file content hashes** appended as query strings (e.g. `style.css?v=a141b4e6`) to bust caches without affecting load time.

**This is fully automated.** On every push to `main`, the GitHub Actions workflow (`.github/workflows/cachebust.yml`) runs `scripts/cachebust.sh`, which:

1. Finds all local CSS/JS references across every HTML file
2. Computes a content hash for each asset using `git hash-object`
3. Stamps (or re-stamps) the `?v=` query string
4. Commits the changes back if anything changed

**You don't need to do anything.** Just write plain references like `<link rel="stylesheet" href="/css/style.css">` or `<script src="/js/main.js">` and the CI handles the rest.

To run it manually (optional):

```bash
./scripts/cachebust.sh
```

The script is idempotent — running it twice without changing assets produces zero diff. It uses `git hash-object` instead of `md5sum` for cross-platform compatibility (Linux, macOS, Windows Git Bash).

### Vendor Library Updates

**`.github/workflows/update-vendor-libs.yml`** — Runs weekly (Wednesday 9:00 AM GMT+7) and opens a PR if any vendor libraries (Highlight.js, etc.) have newer versions. Can also be triggered manually via `workflow_dispatch`.

---

## Theme & Customization

The site uses a dark terminal-style design with **red** as the accent color. All theme colors are CSS custom properties in `css/style.css` under `:root`:

```css
:root {
  --accent: …;
  --accent-hover: …;
  --accent-bg: …;
  --accent-border: …;
}
```

To change the accent color, update these four variables.

---

## Development

No build tools required. Open any `.html` file in a browser, or use a local server for root-relative paths to work:

```bash
# Python
python3 -m http.server 8000

# Node.js
npx serve .
```

---

## Deployment

Push to `main`. GitHub Pages deploys automatically. The cache-busting workflow runs after each push to ensure fresh assets for all visitors.
