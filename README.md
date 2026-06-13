# ASCII.STUDIO — Reflex

Image/video → ASCII art. Fully custom UI built with [Reflex](https://reflex.dev)
(Python → React), reusing the original NumPy conversion backend unchanged.

## Layout

```
ascii-studio-reflex/
├── rxconfig.py                 # Reflex config (app_name = "ascii_studio")
├── requirements.txt            # reflex + numpy + opencv + pillow + matplotlib
├── assets/styles.css           # the entire design system (JetBrains Mono, cards, sliders…)
├── ascii_studio/
│   └── ascii_studio.py         # State + UI  ← the app
└── ascii_engine.py renderer.py fx.py themes.py utils.py video_processor.py   # backend (untouched)
```

## Run locally  (needs Python 3.10+ and Node 18+ — Reflex installs its own frontend toolchain)

```bash
cd ascii-studio-reflex
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
reflex init        # first run only — accepts defaults, wires up .web/
reflex run         # opens http://localhost:3000
```

> `reflex init` will NOT overwrite `ascii_studio/ascii_studio.py` — it only
> scaffolds the `.web/` frontend folder. Say yes to any prompts.

## Deploy to Reflex Cloud (free tier)

1. `pip install reflex` locally and log in:
   ```bash
   reflex login
   ```
   A browser opens; sign in with GitHub/Google. This stores a token locally.

2. From the project root, deploy:
   ```bash
   reflex deploy
   ```
   On first deploy it asks for an **app name** (e.g. `ascii-studio`) and a
   **region**. It zips the repo, installs `requirements.txt` in the cloud,
   builds the frontend, and returns a public URL like
   `https://ascii-studio.reflex.run`.

3. To ship updates later, just re-run `reflex deploy` from the project root.

That's the whole flow — no Dockerfile, no separate frontend host. Reflex Cloud
runs both the Python backend (your State + the NumPy pipeline) and the compiled
React frontend.

### Alternative hosts
- **Render / Railway / Fly.io** — run `reflex export` to produce a frontend
  bundle + backend, or run `reflex run --env prod` in a container. Reflex's docs
  have per-platform guides; Reflex Cloud is by far the least work.

## Notes
- `matplotlib` is in requirements only to guarantee a bundled monospace font
  (DejaVuSansMono) for the renderer on cloud images that lack system fonts.
- Video output is returned as an inline `data:` URI, so very large clips are
  memory-heavy — keep clips short (the 30 MB cap from the original app stands).
