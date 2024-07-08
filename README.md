# DialAttack

CommonVoice 18.0 has the following "macro-dialect" composition (L2 data removed):

|                        | Central | Balear | Nord  | Nord-Occidental | Valencià | Unknown |
|------------------------|---------|--------|-------|-----------------|----------|---------|
| Validated Sample Count | 757873  | 23856  | 29623 | 74332           | 83254    | 598415  |
| Duration (Hours)       | 1150.93 | 36.66  | 46.49 | 100             | 119.26   | 912.23  |

**The values above are calculated after filtering out repeated recordings of the same sentence from the same macro-dialect. If the same sentence was recorded by speakers from different macro-dialects, all recordings of that sentence from different macro-dialects were left in the data. 

Working in a way that tries to use as much data as possible, we aim to train ASR models that vary in how biased toward the Central dialect they are. Given that the Balear dialect represents the "lower limit" in terms of data, I worked from the assumption that at most, we can have ~28.8 hours of Balear training data (80% of 36 hr), and 3.6 hours each of Balear development and test data (10% each). 

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
| Model 1 (100% Central) | 95420   | 0      | 0     | 0               | 0        | 95420 |
| Model 2 (80% Central)  | 76336   | 4771   | 4771  | 4771            | 4771     | 95420 |
| Model 3 (50% Central)  | 47710   | 11927  | 11927 | 11927           | 11927    | 95418 |
| Model 4 (20% Central)  | 19084   | 19084  | 19084 | 19084           | 19084    | 95420 |

__Development__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 11925   | 0      | 0    | 0               | 0        | 11925 |
| Model 2 (80% Central)  | 9540    | 596    | 596  | 596             | 596      | 11924 |
| Model 3 (50% Central)  | 5962    | 1490   | 1490 | 1490            | 1490     | 11922 |
| Model 4 (20% Central)  | 2385    | 2385   | 2385 | 2385            | 2385     | 11925 |

__Evaluation__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 2385    | 2385   | 2385 | 2385            | 2385     | 11925 |
| Model 2 (80% Central)  | 2385    | 2385   | 2385 | 2385            | 2385     | 11925 |
| Model 3 (50% Central)  | 2385    | 2385   | 2385 | 2385            | 2385     | 11925 |
| Model 4 (20% Central)  | 2385    | 2385   | 2385 | 2385            | 2385     | 11925 |

For now, this assumes that we'll only use the validated data with dialect annotations. If 144 hours of training data isn't enough to get good results on the ASR, we can look into making a dialect identifier. But, since the focus is more on the composition/bias in the train set rather than the quality of the ASR overall, I think we can proceed without using the unlabeled data. 
