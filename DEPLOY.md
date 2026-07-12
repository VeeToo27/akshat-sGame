# Deploying to Streamlit Community Cloud

## Repo layout required
Place these files in the root of your GitHub repo (same repo you already
have, `Autonomous-Lunar-Landing-using-Proximal-Policy-Optimization`):

```
app.py
requirements.txt
packages.txt
models/ppo_lunarlander.zip   <- already in your repo
logs/training_monitor.csv    <- already in your repo (optional)
graphs/learning_curve.png    <- already in your repo (optional)
```

`app.py`, `requirements.txt`, and `packages.txt` are new — add them at the
repo root (same level as `train.py`, `evaluate.py`, etc.).

## Steps
1. Commit and push `app.py`, `requirements.txt`, and `packages.txt` to the
   `main` branch of your GitHub repo.
2. Go to https://share.streamlit.io (Streamlit Community Cloud) and sign in
   with GitHub.
3. Click **New app**, select your repo, branch `main`, and set
   **Main file path** to `app.py`.
4. Deploy. First build will take a few minutes since it compiles Box2D
   (`packages.txt` installs `swig`/`build-essential` for this) and installs
   PyTorch (CPU build, via the pinned `--extra-index-url`) and
   Stable-Baselines3.

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
