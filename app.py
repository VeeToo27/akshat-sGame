"""
Autonomous Lunar Landing using PPO — Streamlit Demo
-----------------------------------------------------
A single-file Streamlit app that loads a trained Stable-Baselines3 PPO
policy for the LunarLander-v3 environment and lets you:
  * Watch the trained agent land (rendered as an in-browser video/GIF)
  * Run quantitative evaluation over N episodes
  * View the training learning curve

Expected repo layout (relative to this file):
    app.py
    requirements.txt
    packages.txt
    models/ppo_lunarlander.zip
    logs/training_monitor.csv      (optional, for the training curve)
    graphs/learning_curve.png      (optional, pre-rendered curve)
"""

import os
import csv
import tempfile

import numpy as np
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Lunar Lander PPO",
    page_icon="🚀",
    layout="wide",
)

MODEL_PATH = "models/ppo_lunarlander.zip"
LOG_PATH = "logs/training_monitor.csv"
CURVE_IMG_PATH = "graphs/learning_curve.png"


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained PPO policy...")
def load_model(model_path: str):
    from stable_baselines3 import PPO

    if not os.path.exists(model_path):
        return None
    return PPO.load(model_path)


@st.cache_data(show_spinner=False)
def load_monitor_log(log_path: str):
    """Parse an SB3 Monitor CSV file into a list of episode rewards."""
    rewards = []
    if not os.path.exists(log_path):
        return rewards
    with open(log_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not row[0].startswith("#") and row[0] != "r":
                try:
                    rewards.append(float(row[0]))
                except ValueError:
                    continue
    return rewards


def run_episode(model, deterministic: bool = True, seed: int | None = None):
    """Run a single episode, capturing rgb_array frames and the total reward."""
    import gymnasium as gym

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        total_reward += reward
        done = terminated or truncated
        steps += 1

    env.close()
    return frames, total_reward, steps


def frames_to_video_bytes(frames, fps: int = 30) -> bytes:
    """Encode a list of rgb_array frames to an mp4 file and return its bytes."""
    import imageio

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    imageio.mimsave(tmp_path, frames, fps=fps, macro_block_size=1)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("🚀 Lunar Lander PPO")
st.sidebar.markdown(
    "A PPO (Proximal Policy Optimization) agent trained with "
    "**Stable-Baselines3** on Gymnasium's `LunarLander-v3` environment."
)

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Live Simulation", "Evaluation", "Training Analytics"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model file expected at `models/ppo_lunarlander.zip`. "
    "If it's missing, the app will only show static training info."
)

model = load_model(MODEL_PATH)

# ------------------------------------------------------------------
# Overview page
# ------------------------------------------------------------------
if page == "Overview":
    st.title("Autonomous Lunar Landing using PPO")
    st.markdown(
        """
This project trains an agent to autonomously land a spacecraft on a
designated landing pad using **Proximal Policy Optimization (PPO)**
inside the Box2D-based `LunarLander-v3` physics simulator.

**Environment specs**
- **Observation space:** 8-D continuous vector — position, velocity, angle,
  angular velocity, and leg-ground contact flags.
- **Action space:** Discrete(4) — no-op, fire left engine, fire main engine,
  fire right engine.

**Hyperparameters**

| Parameter | Value |
|---|---|
| Policy network | MlpPolicy |
| Learning rate | 3e-4 |
| Discount factor (γ) | 0.99 |
| Batch size | 64 |
| Rollout steps | 2048 |
| Optimization epochs | 10 |
| Clip range | 0.2 |
| Total training timesteps | 500,000 |
        """
    )
    if model is None:
        st.warning(
            "No trained model found at `models/ppo_lunarlander.zip`. "
            "Add it to the repo to enable the Live Simulation and "
            "Evaluation pages."
        )
    else:
        st.success("Trained PPO model loaded and ready.")

# ------------------------------------------------------------------
# Live Simulation page
# ------------------------------------------------------------------
elif page == "Live Simulation":
    st.title("🎮 Live Simulation")
    st.markdown(
        "Run the trained agent for a single episode and watch the landing."
    )

    col1, col2 = st.columns(2)
    with col1:
        deterministic = st.toggle("Deterministic actions", value=True)
    with col2:
        seed = st.number_input(
            "Random seed (optional)", min_value=0, max_value=100000, value=0, step=1
        )

    if model is None:
        st.error("Model not found — cannot run simulation.")
    elif st.button("▶️ Run episode", type="primary"):
        with st.spinner("Simulating episode and rendering video..."):
            frames, total_reward, steps = run_episode(
                model, deterministic=deterministic, seed=int(seed)
            )
            video_bytes = frames_to_video_bytes(frames, fps=30)

        st.video(video_bytes)
        m1, m2 = st.columns(2)
        m1.metric("Total Reward", f"{total_reward:.2f}")
        m2.metric("Episode Length", f"{steps} steps")

        if total_reward >= 200:
            st.success("Excellent landing! 🎉 (score ≥ 200)")
        elif total_reward >= 100:
            st.info("Decent landing, but room to improve.")
        else:
            st.warning("Rough landing / crash.")

# ------------------------------------------------------------------
# Evaluation page
# ------------------------------------------------------------------
elif page == "Evaluation":
    st.title("📊 Quantitative Evaluation")
    st.markdown(
        "Run the deterministic policy over multiple episodes to estimate "
        "mean performance."
    )

    n_episodes = st.slider("Number of evaluation episodes", 1, 50, 20)

    if model is None:
        st.error("Model not found — cannot run evaluation.")
    elif st.button("Run evaluation", type="primary"):
        import gymnasium as gym
        from stable_baselines3.common.evaluation import evaluate_policy

        env = gym.make("LunarLander-v3")
        with st.spinner(f"Evaluating over {n_episodes} episodes..."):
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=n_episodes, deterministic=True
            )
        env.close()

        c1, c2 = st.columns(2)
        c1.metric("Mean Reward", f"{mean_reward:.2f}")
        c2.metric("Std Dev", f"{std_reward:.2f}")

        if mean_reward >= 200:
            st.success("Excellent operational rating — stable, centered landings.")
        elif mean_reward >= 100:
            st.info("Good performance, but not yet consistently excellent.")
        else:
            st.warning("Below target performance.")

# ------------------------------------------------------------------
# Training Analytics page
# ------------------------------------------------------------------
elif page == "Training Analytics":
    st.title("📈 Training Analytics")

    if os.path.exists(CURVE_IMG_PATH):
        st.image(CURVE_IMG_PATH, caption="Pre-rendered learning curve", use_container_width=True)
    else:
        st.info("No pre-rendered learning curve image found.")

    rewards = load_monitor_log(LOG_PATH)
    if rewards:
        st.subheader("Learning curve from training log")
        window = st.slider("Moving-average window", 5, 100, 20)
        rolling_mean = (
            np.convolve(rewards, np.ones(window) / window, mode="valid")
            if len(rewards) >= window
            else np.array(rewards)
        )

        import pandas as pd

        df = pd.DataFrame({"Episode Reward": rewards})
        df["Moving Average"] = pd.Series(rolling_mean).reindex(
            range(len(rewards))
        )
        st.line_chart(df)

        st.caption(
            f"{len(rewards)} episodes logged | "
            f"Final moving average: {rolling_mean[-1]:.2f}"
            if len(rolling_mean) > 0
            else ""
        )
    else:
        st.info(
            "No `logs/training_monitor.csv` found in the repo — "
            "add it to see the interactive learning curve here."
        )
