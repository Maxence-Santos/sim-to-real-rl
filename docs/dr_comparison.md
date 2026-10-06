# Domain randomization: baseline versus randomized policy

Both policies are PPO runs of 10 million steps on the CPU build of PyTorch, eight MuJoCo processes, seed 0. `models/walk_baseline` was trained with domain randomization off. `models/walk_dr` was trained with `configs/train/domain_randomization.yaml` on. That second run took 4056 s and finished at about 2465 steps per second. Episodes stayed at 1000 control steps. The training return rose from about 1580 to 3090.

![PPO training return over 10 million steps](figures/training_return.png)

The blue trace finishes near 3170 and has several sharp drops, including one near zero around 8.3 million steps. Those drops are in the TensorBoard log. The orange trace is the randomized run.

Evaluation uses 16 episodes, seeds 1 through 16, and the deterministic actor. Each policy is normalized with its own saved observation statistics. A success is an episode that does not fall and whose mean absolute forward-speed error stays under 0.3 m/s. The same seed on the same condition gives both policies the same physical draw and the same command.

## One parameter at a time

On every row except `latency` and `all`, action delay is zero. `latency` and `all` draw 10–40 ms.

| Condition | Baseline error | Baseline success | DR error | DR success |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 0.014 m/s | 16/16 | 0.306 m/s | 7/16 |
| Mass ±20 % | 0.065 m/s | 15/16 | 0.462 m/s | 4/16 |
| Friction ×0.5–1.5 | 0.194 m/s | 11/16 | 0.471 m/s | 5/16 |
| Damping ×0.75–1.25 | 0.100 m/s | 14/16 | 0.474 m/s | 4/16 |
| Latency 10–40 ms | 0.088 m/s | 16/16 | 0.052 m/s | 16/16 |
| Sensor noise | 0.220 m/s | 13/16 | 0.282 m/s | 8/16 |
| All of the above | 0.399 m/s | 5/16 | 0.092 m/s | 16/16 |

Survival is 20 s everywhere except one baseline episode under friction alone, which fell at 2.46 s. The randomized policy did not fall in this set.

Read this table with the delay in mind. At 50 Hz the configured 10–40 ms rounds to 1 or 2 control steps, so a delay of zero is not in the training support of `walk_dr`. The nominal, mass, friction, damping, and sensor-noise rows turn latency off. On those rows the randomized policy walks at about 0.16 m/s whatever the command. The mass, friction, and damping numbers above are therefore not the effect of those parameters alone.

## Each parameter on top of the training latency

Delay is 1 or 2 control steps, which is the range the randomized policy actually saw. Fixed 20 ms is always 1 step. Fixed 40 ms is always 2 steps. Survival is 20 s on every episode below.

| Condition | Baseline error | Baseline success | DR error | DR success |
| --- | ---: | ---: | ---: | ---: |
| Latency fixed at 20 ms | 0.034 m/s | 16/16 | 0.048 m/s | 16/16 |
| Latency fixed at 40 ms | 0.131 m/s | 16/16 | 0.055 m/s | 16/16 |
| Mass, with latency | 0.189 m/s | 12/16 | 0.066 m/s | 16/16 |
| Friction, with latency | 0.337 m/s | 9/16 | 0.066 m/s | 16/16 |
| Damping, with latency | 0.362 m/s | 8/16 | 0.078 m/s | 16/16 |
| Sensor noise, with latency | 0.244 m/s | 10/16 | 0.063 m/s | 16/16 |
| All parameters | 0.399 m/s | 5/16 | 0.092 m/s | 16/16 |

![Forward-speed error with the training delay kept on](figures/dr_with_latency.png)

The dashed line is the 0.3 m/s success threshold.

For the randomized policy, the worst episode in the single-parameter rows is 0.121 m/s (damping). Under the full randomization the worst episode is 0.154 m/s. No single parameter accounts for the gap, and stacking them does not open a heavy tail.

For the baseline, sensor noise shifts every episode (minimum error 0.138 m/s once latency is on). Friction and damping add a tail that reaches about 0.77 m/s. Under the full set, every episode is worse than 0.20 m/s and the mean forward speed falls to about 0.24 m/s against commands whose mean is 0.64 m/s.

## What was not changed

Reward weights stayed at the values that produced the first walk: an alive bonus of 1 per step, velocity weight 1.5, yaw 0.5, upright 0.3, energy 0.0005, action rate 0.01, fall penalty 2. The randomized run did not learn to fall, and the return kept rising, so the weights were not the limit. The measured limit is the latency range: the randomized policy tracks inside the delay it was trained on, and it does not track when that delay is removed.

Commands to reproduce the two tables:

```bash
.venv/bin/python -m src.evaluation.compare_dr --model models/walk_baseline.zip --episodes 16 --seed 1
.venv/bin/python -m src.evaluation.compare_dr --model models/walk_dr.zip --episodes 16 --seed 1
.venv/bin/python -m src.evaluation.compare_dr --model models/walk_dr.zip --episodes 16 --seed 1 \
  --conditions mass_given_latency friction_given_latency damping_given_latency noise_given_latency latency_20ms latency_40ms
```

The Gazebo rollouts of these two policies are in [gazebo_transfer.md](gazebo_transfer.md).
