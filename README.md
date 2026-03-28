# FTU Labs Website

Static website for FTU Labs, hosted on GitHub Pages at [ftulabs.github.io](https://ftulabs.github.io).

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
