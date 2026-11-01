"""Deterministic evaluation of every checkpoint in ckpt/: 10 episodes each, fresh env seeds.
Writes results/eval.json: {run_name: {"returns": [...], "mean": ..}}"""
import glob, json, os, sys
import numpy as np
import torch
from ppo import Agent
import gymnasium as gym
torch.set_num_threads(1)
N_EP = 10


def run_episodes(env_id, act_fn, seed):
    env = gym.make(env_id)
    rets = []
    for ep in range(N_EP):
        o, _ = env.reset(seed=10_000 + seed * 100 + ep)
        done, R = False, 0.0
        while not done:
            o, r, te, tr, _ = env.step(act_fn(o))
            R += r; done = te or tr
        rets.append(R)
    return rets


def eval_ours(path, env_id, seed):
    from ppo import Agent
    s = torch.load(path, weights_only=False)
    od = s["agent"]["actor.0.weight"].shape[1]; ad = s["agent"]["actor.4.weight"].shape[0]
    ag = Agent(od, ad); ag.load_state_dict(s["agent"])
    lo, hi = gym.make(env_id).action_space.low, gym.make(env_id).action_space.high

    def act(o):
        if s["obs_norm"]:
            o = np.clip((o - s["obs_mean"]) / np.sqrt(s["obs_var"] + 1e-8), -10, 10)
        with torch.no_grad():
            return np.clip(ag.act(torch.as_tensor(o, dtype=torch.float32), True).numpy(), lo, hi)
    return run_episodes(env_id, act, seed)


def eval_sb3(path, env_id, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    vn = VecNormalize.load(path.replace(".zip", "_vecnorm.pkl"), DummyVecEnv([lambda: gym.make(env_id)]))
    vn.training = False; vn.norm_reward = False
    m = PPO.load(path, device="cpu")
    act = lambda o: m.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]
    return run_episodes(env_id, act, seed)


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "*"
    out = json.load(open("results/eval.json")) if os.path.exists("results/eval.json") else {}
    for p in sorted(glob.glob(f"ckpt/{pat}.pt") + glob.glob(f"ckpt/{pat}.zip")):
        name = os.path.basename(p).rsplit(".", 1)[0]
        env_id, _, rest = name.partition("_")
        seed = int(rest.rsplit("_s", 1)[1])
        r = (eval_ours if p.endswith(".pt") else eval_sb3)(p, env_id, seed)
        out[name] = {"returns": r, "mean": float(np.mean(r))}
        print(name, f"{np.mean(r):.1f}", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/eval.json", "w"), indent=1)
