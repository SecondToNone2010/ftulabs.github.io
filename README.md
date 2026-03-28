# FTU Labs Website

Static website for FTU Labs, hosted on GitHub Pages at [ftulabs.github.io](https://ftulabs.github.io).

## Contributing Guide (For Students)

Follow these steps every time you make a change. This workflow mirrors how professional teams ship code — learning it now will serve you in internships, open-source projects, and your career.

### 1. Clone the Repo (First Time Only)

```bash
git clone https://github.com/ftulabs/ftulabs.github.io.git
cd ftulabs.github.io
```

If you already cloned it before, make sure you're up to date:

```bash
git checkout main
git pull origin main
```

### 2. Create a Branch

**Never work directly on `main`.** Create a descriptive branch for your change:

```bash
git checkout -b <type>/<short-description>
```

Branch naming conventions:

| Type        | When to use                        | Example                            |
| ----------- | ---------------------------------- | ---------------------------------- |
| `feature/`  | Adding something new               | `feature/add-member-nguyen-van-a`  |
| `fix/`      | Fixing a bug or typo               | `fix/broken-nav-link`              |
| `blog/`     | Adding or editing a blog post      | `blog/intro-to-ml-post`            |
| `docs/`     | Updating documentation             | `docs/update-readme`               |
| `refactor/` | Restructuring without new features | `refactor/css-variables`           |

> **Why branches?** They isolate your work so a half-finished change never breaks the live site. They also make code review possible — your teammates can see exactly what changed.

### 3. Write Clean Code

Before you commit anything, make sure your code is clean and readable. Here are the rules:

#### Use Meaningful Names

```html
<!-- Bad -->
<div class="x1">
  <div class="a"></div>
</div>

<!-- Good -->
<div class="member-card">
  <div class="member-avatar"></div>
</div>
```

#### Comment the "Why", Not the "What"

Comments should explain **why** something exists or **why** a non-obvious decision was made — not repeat what the code already says.

```css
/* Bad — restates what the code does */
/* Set color to red */
color: #8B1A1A;

/* Good — explains the reason behind the choice */
/* FTU official crimson — matches university branding guidelines */
color: #8B1A1A;
```

```html
<!-- Bad — obvious from the tag itself -->
<!-- This is a paragraph -->
<p>Welcome to FTU Labs.</p>

<!-- Good — explains a decision that isn't obvious -->
<!-- Using <details> instead of JS toggle for accessibility and zero-JS support -->
<details>
  <summary>Show more</summary>
  <p>Additional content here.</p>
</details>
```

```js
// Bad — describes what the code does
// Loop through items
items.forEach(item => { ... });

// Good — explains intent
// Stagger reveal animations so cards don't all appear at once
items.forEach((item, i) => {
  item.style.transitionDelay = `${i * 100}ms`;
});
```

#### Keep It Consistent

- Use **2-space indentation** for HTML, CSS, and JS (match the existing files).
- Use **lowercase with hyphens** for CSS classes: `blog-title`, not `blogTitle` or `Blog_Title`.
- Keep lines under **100 characters** when possible.
- Remove dead code — don't comment it out "just in case" (that's what git history is for).

#### Keep Commits Small and Focused

Each commit should do **one thing**. If you added a team member and also fixed a typo in the blog, make two commits:

```bash
# Stage only the team page change
git add team.html
git commit -m "Add Nguyen Van A to team page"

# Stage only the blog fix
git add blog.html
git commit -m "Fix typo in ML blog post title"
```

Write commit messages in the **imperative mood** (like giving an order):

```
# Good
Add blog post about neural networks
Fix navigation link on mobile
Remove unused CSS class

# Bad
Added blog post
fixed stuff
changes
```

### 4. Push Your Branch and Open a Pull Request

```bash
git push origin <your-branch-name>
```

Then open a Pull Request (PR) on GitHub:

1. Go to the repo on GitHub. You'll see a banner saying your branch was recently pushed — click **"Compare & pull request"**.
2. Fill in the PR template:
   - **Title:** Short and descriptive (e.g., `Add Nguyen Van A to team page`).
   - **Description:** Explain **what** you changed and **why**. Include a screenshot if it's a visual change.
3. **Request a review** from at least one teammate.
4. Wait for feedback. If changes are requested, push new commits to the same branch — the PR updates automatically.

You can also create a PR from the command line:

```bash
gh pr create --title "Add Nguyen Van A to team page" --body "Added new member profile with bio and links."
```

#### PR Checklist

Before marking your PR as ready, verify:

- [ ] I created a branch (not committing to `main` directly)
- [ ] I tested my changes locally (opened the HTML in a browser)
- [ ] My code follows the existing style (indentation, class naming)
- [ ] My commits are small, focused, and have clear messages
- [ ] I described what changed and why in the PR description

### 5. After Your PR Is Merged

Switch back to `main`, pull the latest changes, and **delete your branch both locally and on GitHub**:

```bash
git checkout main
git pull origin main

# Delete the local branch
git branch -d <your-branch-name>

# Delete the branch on GitHub (required — otherwise it stays visible on the remote)
git push origin --delete <your-branch-name>
```

> **Why do deleted branches still show up on GitHub?**
> `git branch -d` only removes the **local** branch. The branch on GitHub is a separate copy — you must explicitly delete it with `git push origin --delete <branch>`. If you skip this step, stale branches pile up on the remote repo.
>
> **If the branch was already deleted on GitHub** (e.g., via the "Delete branch" button after merging the PR), clean up your local stale references with:
>
> ```bash
> git fetch --prune
> ```
>
> **Tip for repo admins:** Enable **Settings → General → Pull Requests → "Automatically delete head branches"** on GitHub so merged branches are removed automatically.

### Quick Reference

```
main (protected — always deployable)
 └── feature/add-member-nguyen-van-a   ← your work happens here
      ├── commit: "Add member photo"
      ├── commit: "Add member bio and links"
      └── → Open PR → Code review → Merge → Done
```

## Project Structure

```
ftulabs.github.io/
├── index.html            # Home page
├── blog.html             # Blog listing page
├── projects.html         # Projects page
├── team.html             # Team members page
├── research.html         # Research publications page
├── css/
│   └── style.css         # Global stylesheet (theme colors, layout, components)
├── js/
│   └── main.js           # Navigation toggle & scroll reveal animations
├── assets/               # Images, videos, and other media files
└── blog/
    └── post-template.html  # Example blog post (use as template for new posts)
```

## Adding New Content

This is a static HTML site with no build step. To add content, edit the HTML files directly and commit.

### Add a Blog Post

1. **Create the post file:** Copy `blog/post-template.html` to a new file in the `blog/` folder (e.g. `blog/my-new-post.html`).
2. **Edit the post content:**
   - Update the `<title>` and `<meta name="description">` in `<head>`.
   - Update the `<h1 class="post-title">` with your title.
   - Update the `<div class="post-meta">` with the date, authors, and read time.
   - Write your content inside `<article class="post-content">`. Use standard HTML tags (`<p>`, `<h2>`, `<code>`, `<pre>`, `<blockquote>`, etc.).
3. **Add to the blog listing:** Open `blog.html` and add a new `<div class="blog-item reveal">` entry inside the `<div class="blog-list">` section:
   ```html
   <div class="blog-item reveal">
     <div class="blog-date">YYYY-MM-DD</div>
     <h2 class="blog-title"><a href="blog/my-new-post.html">Post Title</a></h2>
     <p class="blog-excerpt">A short description of the post.</p>
   </div>
   ```
4. **(Optional) Add to homepage:** If this is a featured post, add it to the latest section in `index.html`.

### Add a Team Member

Open `team.html` and add a new `<div class="member reveal">` block inside the `<div class="team-grid">` section:

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

### Add a Project

Open `projects.html` and add a new `<div class="card reveal">` block inside the appropriate `<div class="card-grid cols-2">` section (active or completed):

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

### Add a Research Paper

Open `research.html` and add a new `<div class="paper reveal">` block inside the appropriate year's `<div class="research-list">` section. To add a new year, create a new `<div class="research-year">` above a new `<div class="research-list">`:

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

### Add Images

Place image files in an `assets/` or `images/` folder (create it if needed). Images inside `<article class="post-content">` are automatically styled (full-width, bordered). Use standard HTML:

```html
<!-- Basic image -->
<img src="../assets/my-diagram.png" alt="Description of the image">

<!-- Image with caption -->
<figure>
  <img src="../assets/results-chart.png" alt="Benchmark results">
  <figcaption>Figure 1: Inference latency comparison across sequence lengths.</figcaption>
</figure>
```

Supported formats: `.png`, `.jpg`, `.webp`, `.svg`, `.gif`. Keep images optimized for web (aim for < 500 KB per image).

### Add Videos

Embed videos directly or host them externally:

```html
<!-- Self-hosted video -->
<video controls width="100%">
  <source src="../assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<!-- YouTube embed -->
<iframe width="100%" height="400" src="https://www.youtube.com/embed/VIDEO_ID"
  frameborder="0" allowfullscreen></iframe>

<!-- Vimeo embed -->
<iframe width="100%" height="400" src="https://player.vimeo.com/video/VIDEO_ID"
  frameborder="0" allowfullscreen></iframe>
```

For large video files, prefer hosting on YouTube/Vimeo and embedding via `<iframe>` rather than committing them to the repo.

### Add Audio

```html
<!-- Audio player -->
<audio controls>
  <source src="../assets/podcast-episode.mp3" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>
```

Supported formats: `.mp3`, `.ogg`, `.wav`. For podcast-style content, consider hosting on an external platform and linking to it.

### Add Other Embedded Content

```html
<!-- PDF embed -->
<embed src="../assets/paper-draft.pdf" type="application/pdf" width="100%" height="600px">

<!-- Interactive demo (CodePen, Observable, etc.) -->
<iframe src="https://codepen.io/user/embed/pen-id" width="100%" height="400"
  frameborder="0" allowfullscreen></iframe>

<!-- Slides (Google Slides, Speaker Deck, etc.) -->
<iframe src="https://docs.google.com/presentation/d/SLIDE_ID/embed"
  width="100%" height="400" frameborder="0" allowfullscreen></iframe>
```

### Media Guidelines

- **File organization:** Store media files in an `assets/` folder at the root. For blog-specific media, use `blog/assets/` or `assets/blog/`.
- **File size:** Keep the repo lean. Avoid committing files larger than 10 MB. Use external hosting (YouTube, Vimeo, cloud storage) for large media.
- **Alt text:** Always include descriptive `alt` attributes on `<img>` tags for accessibility.
- **Responsive:** Images and videos use `max-width: 100%` by default and scale to fit their container.

## Theme

The site uses a dark terminal-style design with **red** as the accent color. All theme colors are defined as CSS custom properties in `css/style.css` under the `:root` selector. To change the accent color, update the `--accent`, `--accent-hover`, `--accent-bg`, and `--accent-border` variables.

## Development

No build tools required. Open any `.html` file in a browser to preview, or use a local server:

```bash
# Python
python3 -m http.server 8000

# Node.js
npx serve .
```

## Deployment

Push to `main` branch. GitHub Pages will automatically deploy the site.
