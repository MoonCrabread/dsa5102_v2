Board shape after every placement, over the final-evaluation games (same seeds and 5,000-piece cap as `final_eval.csv`). Each value is averaged within a game, then over games; lower means further from losing. Per-seed values are in brackets.

| Model | Games | Mean lines | Mean max height | 95th pct max height | Peak max height | Mean column height | Mean holes | Mean bumpiness | Pieces with max height ≥ 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CE · MLP | 150 | 1923.6 | 4.71 (5.16, 3.86, 5.11) | 8.7 (9.7, 6.6, 9.6) | 14.0 (15.0, 11.5, 15.4) | 2.96 (3.45, 2.11, 3.32) | 1.52 (2.15, 0.49, 1.93) | 7.85 (7.82, 7.82, 7.91) | 4.1% (6.1%, 0.4%, 5.7%) |
| CE · linear | 150 | 1838.8 | 5.19 (5.04, 5.39, 5.14) | 10.1 (9.8, 10.8, 9.8) | 15.7 (15.2, 16.2, 15.7) | 3.60 (3.45, 3.92, 3.42) | 2.33 (2.17, 2.71, 2.11) | 7.63 (7.60, 7.50, 7.80) | 7.2% (6.0%, 9.3%, 6.3%) |
| DQN | 150 | 1781.7 | 4.76 (4.55, 5.06, 4.66) | 8.3 (7.7, 10.0, 7.2) | 13.1 (12.3, 15.4, 11.7) | 2.80 (2.64, 3.18, 2.59) | 0.81 (0.68, 1.36, 0.39) | 8.50 (8.40, 8.12, 8.97) | 3.2% (2.0%, 6.9%, 0.6%) |
| Heuristic | 50 | 1453.4 | 6.83 | 13.1 | 18.6 | 5.02 | 3.79 | 7.41 | 21.2% |
| Linear TD | 150 | 10.5 | 11.51 (9.97, 9.98, 14.58) | 18.3 (17.5, 17.6, 19.9) | 19.3 (18.9, 18.9, 19.9) | 9.16 (8.98, 8.98, 9.51) | 18.32 (16.00, 15.83, 23.15) | 12.20 (4.58, 4.85, 27.16) | 60.1% (52.1%, 52.7%, 75.6%) |
