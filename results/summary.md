| Env | Run | Seeds | Final eval return, mean +- std over seeds | Per-seed means | Train wall time / run | Training steps/s |
|---|---|---|---|---|---|---|
| HalfCheetah-v5 | ours | 3 | 2994 +- 1540 | 1762, 4721, 2498 | 4.9 min | 3417 |
| HalfCheetah-v5 | SB3 PPO | 3 | 2791 +- 1706 | 4760, 1844, 1768 | 5.3 min | 3134 |
| Hopper-v5 | ours, lambda=1 (no GAE) | 3 | 3359 +- 147 | 3377, 3204, 3495 | 5.3 min | 3127 |
| Hopper-v5 | ours, no obs. normalisation | 3 | 2602 +- 897 | 2123, 2045, 3636 | 5.2 min | 3203 |
| Hopper-v5 | ours | 3 | 3395 +- 211 | 3424, 3590, 3171 | 5.2 min | 3190 |
| Hopper-v5 | SB3 PPO | 3 | 2933 +- 999 | 3280, 3712, 1806 | 5.3 min | 3157 |
| Walker2d-v5 | ours | 3 | 3827 +- 511 | 4043, 3243, 4193 | 5.7 min | 2915 |
| Walker2d-v5 | SB3 PPO | 3 | 3231 +- 1889 | 4414, 1053, 4227 | 5.7 min | 2910 |
