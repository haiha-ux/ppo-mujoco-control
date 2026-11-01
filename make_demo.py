"""Render the seed-0 deterministic policy of our PPO per env to MP4 and GIF. Needs MUJOCO_GL=egl (set below)."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import subprocess
import numpy as np, torch
from PIL import Image
from ppo import Agent
import gymnasium as gym
torch.set_num_threads(1)
SKIP, MAXF = 4, 200  # keep every 4th simulation step, at most 200 frames
envs = sys.argv[1:] or ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5"]
for env_id in envs:
    s = torch.load(f"ckpt/{env_id}_ours_s0.pt", weights_only=False)
    od = s["agent"]["actor.0.weight"].shape[1]; ad = s["agent"]["actor.4.weight"].shape[0]
    ag = Agent(od, ad); ag.load_state_dict(s["agent"])
    env = gym.make(env_id, render_mode="rgb_array", width=384, height=288)
    o, _ = env.reset(seed=12345)
    frames, R, i, done = [], 0.0, 0, False
    while not done and len(frames) < MAXF:
        n = np.clip((o - s["obs_mean"]) / np.sqrt(s["obs_var"] + 1e-8), -10, 10)
        with torch.no_grad():
            a = ag.act(torch.as_tensor(n, dtype=torch.float32), True).numpy()
        o, r, te, tr, _ = env.step(np.clip(a, env.action_space.low, env.action_space.high))
        R += r; done = te or tr; i += 1
        if i % SKIP == 0: frames.append(env.render())
    fps = round(1 / (env.unwrapped.dt * SKIP))
    print(env_id, "steps", i, "return so far", round(R, 1), "terminated" if done else "cut at frame cap", "fps", fps)
    h, w = frames[0].shape[:2]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", f"docs/{env_id.split('-')[0]}.mp4"],
                   input=np.stack(frames).tobytes(), check=True)
    small = [Image.fromarray(f).resize((256, 192)) for f in frames]
    small[0].save(f"docs/{env_id.split('-')[0]}.gif", save_all=True, append_images=small[1:], duration=int(1000 / fps), loop=0)
    for ext in ("gif", "mp4"): print("  ", ext, os.path.getsize(f"docs/{env_id.split('-')[0]}.{ext}") // 1024, "KB")
os._exit(0)  # skip noisy EGL teardown
