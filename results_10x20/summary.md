Value learning (linear TD, DQN) is trained on lines cleared, the CE objective.

| Method | Kind | Mean lines | Per-seed means | Games | Games at cap | Training steps |
|---|---|---:|---|---:|---:|---:|
| Heuristic | reference | 1453.4 | 1453.4 | 50 | 22 | 0 |
| Single-feature greedy | reference | 8.9 | 8.9 | 50 | 0 | 0 |
| Random | reference | 0.0 | 0.0 | 50 | 0 | 0 |
| CE · MLP | model | 1923.6 | 1842.0, 1998.0, 1930.8 | 150 | 139 | 15,080,527 |
| CE · linear | model | 1838.8 | 1871.5, 1705.4, 1939.6 | 150 | 121 | 2,819,196 |
| DQN | model | 1781.7 | 1888.0, 1459.7, 1997.4 | 150 | 123 | 50,000 |
| Linear TD | model | 10.5 | 14.0, 15.4, 2.1 | 150 | 0 | 50,000 |

Games that hit the piece cap are cut short, so their line counts and the means are lower bounds.
