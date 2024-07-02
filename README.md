# DialAttack

CommonVoice 18.0 has the following "macro-dialect" composition (L2 data removed):

|                        | Central | Balear | Nord  | Nord-Occidental | Valencià | Unknown |
|------------------------|---------|--------|-------|-----------------|----------|---------|
| Validated Sample Count | 954330  | 24052  | 29763 | 75606           | 84970    | 711528  |
| Duration (Hours)       | 1434.06 | 36.9   | 46.71 | 101.62          | 121.48   | 1074.82 |

Working in a way that tries to use as much data as possible, we aim need to train ASR models that vary in how biased toward the Central dialect they are. Givent that the Balear dialect represents the "lower limit" in terms of data, I worked from the assumption that at most, we can have ~28.8 hours of Balear training data (80% of 36), and 3.6 hours each of Balear development and test data (10% each). 

The following is one possible set of dialect compositions, ranging from a condition where all the fine-tuning data is in the Central dialect (Model 1) to a condition where the fine-tuning data is perfectly balanced (Model 4). 

In terms of number of hours:

__Train__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 144     | 0      | 0    | 0               | 0        | 144   |
| Model 2 (80% Central)  | 115.2   | 7.2    | 7.2  | 7.2             | 7.2      | 144   |
| Model 3 (50% Central)  | 72      | 18     | 18   | 18              | 18       | 144   |
| Model 4 (20% Central)  | 28.8    | 28.8   | 28.8 | 28.8            | 28.8     | 144   |

__Development__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 18      | 0      | 0    | 0               | 0        | 18    |
| Model 2 (80% Central)  | 14.4    | 0.9    | 0.9  | 0.9             | 0.9      | 18    |
| Model 3 (50% Central)  | 9       | 2.25   | 2.25 | 2.25            | 2.25     | 18    |
| Model 4 (20% Central)  | 3.6     | 3.6    | 3.6  | 3.6             | 3.6      | 18    |

__Evaluation__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 3.6     | 3.6    | 3.6  | 3.6             | 3.6      | 18    |
| Model 2 (80% Central)  | 3.6     | 3.6    | 3.6  | 3.6             | 3.6      | 18    |
| Model 3 (50% Central)  | 3.6     | 3.6    | 3.6  | 3.6             | 3.6      | 18    |
| Model 4 (20% Central)  | 3.6     | 3.6    | 3.6  | 3.6             | 3.6      | 18    |

And in terms of the number of data points that corresponds to:

__Train__

|                        | Central | Balear | Nord  | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|-------|-----------------|----------|-------|
| Model 1 (100% Central) | 96205   | 0      | 0     | 0               | 0        | 96205 |
| Model 2 (80% Central)  | 76964   | 4810   | 4810  | 4810            | 4810     | 96204 |
| Model 3 (50% Central)  | 48102   | 12025  | 12025 | 12025           | 12025    | 96202 |
| Model 4 (20% Central)  | 19241   | 19241  | 19241 | 19241           | 19241    | 96205 |

__Development__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 12025   | 0      | 0    | 0               | 0        | 12025 |
| Model 2 (80% Central)  | 9620    | 601    | 601  | 601             | 601      | 12024 |
| Model 3 (50% Central)  | 6012    | 1503   | 1503 | 1503            | 1503     | 12024 |
| Model 4 (20% Central)  | 2405    | 2405   | 2405 | 2405            | 2405     | 12025 |

__Evaluation__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 2405    | 2405   | 2405 | 2405            | 2405     | 12025 |
| Model 2 (80% Central)  | 2405    | 2405   | 2405 | 2405            | 2405     | 12025 |
| Model 3 (50% Central)  | 2405    | 2405   | 2405 | 2405            | 2405     | 12025 |
| Model 4 (20% Central)  | 2405    | 2405   | 2405 | 2405            | 2405     | 12025 |
