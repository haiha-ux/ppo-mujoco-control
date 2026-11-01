"""Train one run. python train.py --env Hopper-v5 --seed 0 [--impl ours|sb3] [--tag name] [--no-obs-norm] [--no-gae]"""
import argparse, csv, json, os, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--env", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--impl", default="ours", choices=["ours", "sb3"])
p.add_argument("--steps", type=int, default=1_000_000)
p.add_argument("--tag", default=None, help="run name suffix (ablations)")
p.add_argument("--no-obs-norm", action="store_true")
p.add_argument("--no-gae", action="store_true")
p.add_argument("--clip-vloss", action="store_true")
a = p.parse_args()
torch.set_num_threads(1)  # tiny MLPs: 1 thread is faster than 4 here (measured); machine cap is 4
name = f"{a.env}_{a.tag or a.impl}_s{a.seed}"
os.makedirs("logs", exist_ok=True); os.makedirs("ckpt", exist_ok=True)

if a.impl == "ours":
    import ppo
    state, log, wall = ppo.train(a.env, a.seed, a.steps, obs_norm=not a.no_obs_norm, use_gae=not a.no_gae,
                                 clip_vloss=a.clip_vloss)
    torch.save(state, f"ckpt/{name}.pt")
else:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback

    class Rec(BaseCallback):
        def __init__(self): super().__init__(); self.log = []
        def _on_step(self):
            for i in self.locals["infos"]:
                if "episode" in i:
                    self.log.append((self.num_timesteps, i["episode"]["r"], i["episode"]["l"]))
            return True

    # VecMonitor goes below VecNormalize so recorded returns are raw (unnormalised)
    env = VecNormalize(VecMonitor(DummyVecEnv([lambda: gym.make(a.env)] * 4)), norm_obs=True, norm_reward=True,
                       gamma=0.99, clip_obs=10, clip_reward=10)
    env.seed(a.seed)
    # same hyperparameters as ours: 4 envs x 512 steps, batch 64, 10 epochs, lr 3e-4 linear anneal, SB3 defaults otherwise
    m = PPO("MlpPolicy", env, n_steps=512, batch_size=64, n_epochs=10, learning_rate=lambda f: 3e-4 * f,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
            seed=a.seed, device="cpu", verbose=0)
    cb = Rec(); t0 = time.time()
    m.learn(a.steps - a.steps % 2048, callback=cb)
    wall = time.time() - t0; log = cb.log
    m.save(f"ckpt/{name}.zip"); env.save(f"ckpt/{name}_vecnorm.pkl")

with open(f"logs/{name}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["step", "episode_return", "episode_length"]); w.writerows(log)
json.dump({"name": name, "wall_s": wall, "episodes": len(log)}, open(f"logs/{name}.time.json", "w"))
print(f"done {name} wall {wall:.0f}s")
