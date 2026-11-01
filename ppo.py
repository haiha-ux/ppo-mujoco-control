"""PPO from scratch (PyTorch + numpy) for continuous control. No RL library is used for the agent.

Contents: RunningMeanStd, compute_gae, Agent (Gaussian policy, state-independent log-std), train().
"""
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


class RunningMeanStd:
    """Parallel (Chan et al.) running mean/variance over batches."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        m2 = self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards, values, dones: (T, N). dones[t]=1 if the episode ended after step t (terminal or truncated,
    i.e. no bootstrapping across it). last_value: (N,) value of the observation after step T-1.
    Returns advantages (T, N); returns = advantages + values."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_value)
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv


def layer(i, o, std):
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, std)
    nn.init.zeros_(l.bias)
    return l


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        r2 = float(np.sqrt(2))
        self.actor = nn.Sequential(layer(obs_dim, hidden, r2), nn.Tanh(), layer(hidden, hidden, r2), nn.Tanh(),
                                   layer(hidden, act_dim, 0.01))
        self.critic = nn.Sequential(layer(obs_dim, hidden, r2), nn.Tanh(), layer(hidden, hidden, r2), nn.Tanh(),
                                    layer(hidden, 1, 1.0))
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x):
        return torch.distributions.Normal(self.actor(x), self.log_std.exp())

    def act(self, x, deterministic=False):
        if deterministic:
            return self.actor(x)
        return self.dist(x).sample()


def make_env(env_id, n, seed):
    envs = gym.make_vec(env_id, num_envs=n, vectorization_mode="sync",
                        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    envs.reset(seed=seed)
    return envs


def train(env_id, seed, total_steps=1_000_000, n_envs=4, n_steps=512, epochs=10, minibatch=64, lr=3e-4,
          gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
          obs_norm=True, rew_scale=True, adv_norm=True, anneal_lr=True, clip_vloss=False, use_gae=True,
          log_path=None, threads=1):
    torch.set_num_threads(threads)
    np.random.seed(seed)
    torch.manual_seed(seed)
    envs = make_env(env_id, n_envs, seed)
    od, ad = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
    lo, hi = envs.single_action_space.low, envs.single_action_space.high
    agent = Agent(od, ad)
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    if not use_gae:
        lam = 1.0  # lambda=1: advantage = Monte-Carlo return - V(s)
    orms, rrms = RunningMeanStd((od,)), RunningMeanStd(())
    ret_acc = np.zeros(n_envs)  # discounted return accumulator for reward scaling

    def norm_obs(o):
        if not obs_norm:
            return o.astype(np.float32)
        return np.clip((o - orms.mean) / np.sqrt(orms.var + 1e-8), -10, 10).astype(np.float32)

    obs_raw, _ = envs.reset(seed=seed)
    if obs_norm:
        orms.update(obs_raw)
    ep_ret, ep_len = np.zeros(n_envs), np.zeros(n_envs, int)
    log = []  # (global_step, episode_return, episode_length)
    batch = n_envs * n_steps
    n_updates = total_steps // batch
    gstep, t0 = 0, time.time()
    B = lambda *s: np.zeros(s, np.float32)
    for u in range(n_updates):
        if anneal_lr:
            for g in opt.param_groups:
                g["lr"] = lr * (1 - u / n_updates)
        O, A, LP, R, D, V = B(n_steps, n_envs, od), B(n_steps, n_envs, ad), B(n_steps, n_envs), B(n_steps, n_envs), \
            B(n_steps, n_envs), B(n_steps, n_envs)
        for t in range(n_steps):
            o = norm_obs(obs_raw)
            with torch.no_grad():
                ot = torch.as_tensor(o)
                d = agent.dist(ot)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
                v = agent.value(ot)
            O[t], A[t], LP[t], V[t] = o, a.numpy(), lp.numpy(), v.numpy()
            obs_raw, r, term, trunc, info = envs.step(np.clip(a.numpy(), lo, hi))
            gstep += n_envs
            ep_ret += r
            ep_len += 1
            r = r.astype(np.float64)
            done = term | trunc
            if rew_scale:  # divide by std of the running discounted return (no mean subtraction)
                ret_acc = ret_acc * gamma + r
                rrms.update(ret_acc)
                ret_acc[done] = 0.0
                r = np.clip(r / np.sqrt(rrms.var + 1e-8), -10, 10)
            if trunc.any():  # time-limit truncation: bootstrap from V(final observation)
                idx = np.where(trunc & ~term)[0]
                if len(idx):
                    fo = norm_obs(np.stack([info["final_obs"][i] for i in idx]))
                    with torch.no_grad():
                        fv = agent.value(torch.as_tensor(fo)).numpy()
                    r[idx] += gamma * fv
            if obs_norm:
                orms.update(obs_raw)  # includes reset obs after done; fine for running stats
            R[t], D[t] = r, done
            for i in np.where(done)[0]:
                log.append((gstep, ep_ret[i], ep_len[i]))
                ep_ret[i], ep_len[i] = 0.0, 0
        with torch.no_grad():
            lv = agent.value(torch.as_tensor(norm_obs(obs_raw))).numpy()
        adv = compute_gae(R, V, D, lv, gamma, lam)
        ret = adv + V
        f = lambda x: torch.as_tensor(x.reshape(batch, *x.shape[2:]))
        bO, bA, bLP, bAdv, bRet, bV = f(O), f(A), f(LP), f(adv), f(ret), f(V)
        for _ in range(epochs):
            perm = torch.randperm(batch)
            for s in range(0, batch, minibatch):
                mb = perm[s:s + minibatch]
                d = agent.dist(bO[mb])
                nlp = d.log_prob(bA[mb]).sum(-1)
                ratio = (nlp - bLP[mb]).exp()
                ad_ = bAdv[mb]
                if adv_norm:
                    ad_ = (ad_ - ad_.mean()) / (ad_.std() + 1e-8)
                pl = torch.max(-ad_ * ratio, -ad_ * ratio.clamp(1 - clip, 1 + clip)).mean()
                nv = agent.value(bO[mb])
                if clip_vloss:
                    vc = bV[mb] + (nv - bV[mb]).clamp(-clip, clip)
                    vl = 0.5 * torch.max((nv - bRet[mb]) ** 2, (vc - bRet[mb]) ** 2).mean()
                else:
                    vl = 0.5 * ((nv - bRet[mb]) ** 2).mean()
                loss = pl + vf_coef * vl - ent_coef * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                opt.step()
        if u % 20 == 0 or u == n_updates - 1:
            last = [x[1] for x in log[-20:]]
            print(f"{env_id} s{seed} step {gstep} ret20 {np.mean(last) if last else float('nan'):.1f} "
                  f"sps {gstep / (time.time() - t0):.0f}", flush=True)
    state = {"agent": agent.state_dict(), "obs_mean": orms.mean, "obs_var": orms.var, "obs_norm": obs_norm}
    return state, log, time.time() - t0
