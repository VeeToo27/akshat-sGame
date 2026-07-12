# Deploying to Streamlit Community Cloud

## Repo layout required
Place these files in the root of your GitHub repo (same repo you already
have, `Autonomous-Lunar-Landing-using-Proximal-Policy-Optimization`):

```
app.py
requirements.txt
packages.txt
runtime.txt
models/ppo_lunarlander.zip   <- already in your repo
logs/training_monitor.csv    <- already in your repo (optional)
graphs/learning_curve.png    <- already in your repo (optional)
```

`app.py`, `requirements.txt`, `packages.txt`, and `runtime.txt` are new —
add them at the repo root (same level as `train.py`, `evaluate.py`, etc.).

## Steps
1. Commit and push `app.py`, `requirements.txt`, `packages.txt`, and
   `runtime.txt` to the `main` branch of your GitHub repo.
2. Go to https://share.streamlit.io (Streamlit Community Cloud) and sign in
   with GitHub.
3. Click **New app**, select your repo, branch `main`, and set
   **Main file path** to `app.py`.
4. In **Advanced settings** before deploying, explicitly set the Python
   version to **3.11** (see note below on why this matters).
5. Deploy. First build will take a few minutes since it compiles
   `pygame`/`box2d-py` from source (`packages.txt` installs `swig`,
   `build-essential`, and the SDL2 dev headers pygame needs) and installs
   PyTorch (CPU build, via the pinned `--extra-index-url`) and
   Stable-Baselines3.

## Why the build was failing (pygame / Python 3.14)
Your previous build error —
`RuntimeError: Unable to run "sdl-config"` while building `pygame` —
happened because Streamlit Cloud currently defaults new apps to
**Python 3.14**, and `pygame` (a dependency of `gymnasium[box2d]`, used for
rendering) has no pre-built wheel for 3.14 yet, so pip tries to compile it
from source and fails without the SDL2 development headers.

Two things fix this together:
- `packages.txt` now installs the SDL2/build headers pygame needs to
  compile from source (`libsdl2-dev`, `pkg-config`, etc.), so the build
  succeeds even on 3.14.
- `runtime.txt` (containing `3.11`) *and* setting Python 3.11 explicitly in
  **Advanced settings** at deploy time both request Python 3.11, where
  pre-built `pygame`/`torch` wheels exist and installs are much faster.
  Note: there's a currently-open Streamlit Cloud bug where `runtime.txt`
  and the Advanced settings Python picker are sometimes ignored and the
  build still runs on 3.14 — if that happens, the `packages.txt` fix above
  should still let the build succeed (just via a slower source compile).
  If you've already deployed once, you must **delete and redeploy** the
  app to change its Python version — editing `runtime.txt` alone won't
  change an existing deployment.

## Notes
- The app only *loads* the already-trained model
  (`models/ppo_lunarlander.zip`) for inference — it does not retrain on
  Streamlit Cloud, so no GPU is needed.
- If `models/ppo_lunarlander.zip` is missing from the repo, the app still
  loads and shows the Overview page, but Live Simulation / Evaluation are
  disabled.
- Video rendering uses `imageio` + `imageio-ffmpeg` (bundled ffmpeg binary,
  no system ffmpeg required).
- Free-tier Streamlit Cloud apps have limited CPU/RAM — episodes run fast
  (LunarLander episodes are short), so this should stay well within limits.
