# Project page

A single static page with no build step and no external dependencies — no
webfonts, no CDN scripts, no analytics. Open `index.html` and it works.

## Figures

The page loads figures from `assets/` *relative to `index.html`*, while the
repository keeps them in the top-level `assets/`. Link the two:

```bash
cd project_page
ln -s ../assets assets
```

The symlink is enough for local preview. For deployment, copy instead, because
GitHub Pages does not reliably follow symlinks:

```bash
cd project_page
rm -f assets && mkdir assets && cp ../assets/*.jpg assets/
```

Any figure that does not exist yet renders as a labelled dashed box naming the
missing file, so the page is safe to preview and to publish before all the
figures are ready. See [../assets/README.md](../assets/README.md) for the file
list.

## Preview

```bash
cd project_page && python -m http.server 8000
# then open http://localhost:8000
```

Opening `index.html` directly as a `file://` URL also works.

## Deploy to GitHub Pages

Copy the figures in as above, then either:

- **Serve from a branch subdirectory.** Settings &rarr; Pages &rarr; deploy from
  branch, folder `/` on `main`. The page is then at
  `https://<user>.github.io/<repo>/project_page/`.
- **Serve from a `gh-pages` branch root**, which gives the shorter
  `https://<user>.github.io/<repo>/`:

```bash
git subtree push --prefix project_page origin gh-pages
```

Add an empty `.nojekyll` file at the served root to stop Jekyll from
reinterpreting the directory.

## Before publishing

The paper, project-page, GitHub, Hugging Face, Open Graph, and BibTeX links are
already set to their expected release locations.

The Open Graph image needs an absolute URL to render in link previews on social
and chat platforms, so it cannot be left relative.
