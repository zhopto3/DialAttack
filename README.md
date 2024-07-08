# DialAttack

CommonVoice 18.0 has the following "macro-dialect" composition (L2 data removed):

|                        | Central | Balear | Nord  | Nord-Occidental | Valencià | Unknown |
|------------------------|---------|--------|-------|-----------------|----------|---------|
| Validated Sample Count | 761017  | 24864  | 29970 | 74452           | 83742    | 598786  |
| Duration (Hours)       | 1155.31 | 38.51  | 47.04 | 100.17          | 120.04   | 912.81  |

**The values above are calculated after filtering out repeated recordings of the same sentence from the same macro-dialect. If the same sentence was recorded by speakers from different macro-dialects, all recordings of that sentence from different macro-dialects were left in the data. 

Working in a way that tries to use as much data as possible, we aim to train ASR models that vary in how biased toward the Central dialect they are. Given that the Balear dialect represents the "lower limit" in terms of data, I worked from the assumption that at most, we can have ~30.4 hours of Balear training data (80% of 38 hr), and 3.8 hours each of Balear development and test data (10% each). 

The following is one possible set of dialect compositions, ranging from a condition where all the fine-tuning data is in the Central dialect (Model 1) to a condition where the fine-tuning data is perfectly balanced (Model 4). 

In terms of number of hours:

__Train__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 152     | 0      | 0    | 0               | 0        | 152   |
| Model 2 (80% Central)  | 121.6   | 7.6    | 7.6  | 7.6             | 7.6      | 152   |
| Model 3 (50% Central)  | 76      | 19     | 19   | 19              | 19       | 152   |
| Model 4 (20% Central)  | 30.4    | 30.4   | 30.4 | 30.4            | 30.4     | 152   |

__Development__

|                        | Central | Balear | Nord  | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|-------|-----------------|----------|-------|
| Model 1 (100% Central) | 19      | 0      | 0     | 0               | 0        | 19    |
| Model 2 (80% Central)  | 15.2    | 0.95   | 0.95  | 0.95            | 0.95     | 19    |
| Model 3 (50% Central)  | 9.5     | 2.375  | 2.375 | 2.375           | 2.375    | 19    |
| Model 4 (20% Central)  | 3.8     | 3.8    | 3.8   | 3.8             | 3.8      | 19    |

__Evaluation__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 3.8     | 3.8    | 3.8  | 3.8             | 3.8      | 19    |
| Model 2 (80% Central)  | 3.8     | 3.8    | 3.8  | 3.8             | 3.8      | 19    |
| Model 3 (50% Central)  | 3.8     | 3.8    | 3.8  | 3.8             | 3.8      | 19    |
| Model 4 (20% Central)  | 3.8     | 3.8    | 3.8  | 3.8             | 3.8      | 19    |

And in terms of the number of data points that corresponds to:

__Train__

|                        | Central | Balear | Nord  | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|-------|-----------------|----------|-------|
| Model 1 (100% Central) | 99455   | 0      | 0     | 0               | 0        | 99455 |
| Model 2 (80% Central)  | 79564   | 4972   | 4972  | 4972            | 4972     | 99452 |
| Model 3 (50% Central)  | 49727   | 12431  | 12431 | 12431           | 12431    | 99451 |
| Model 4 (20% Central)  | 19891   | 19891  | 19891 | 19891           | 19891    | 99455 |

__Development__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 12430   | 0      | 0    | 0               | 0        | 12430 |
| Model 2 (80% Central)  | 9944    | 621    | 621  | 621             | 621      | 12428 |
| Model 3 (50% Central)  | 6215    | 1553   | 1553 | 1553            | 1553     | 12427 |
| Model 4 (20% Central)  | 2486    | 2486   | 2486 | 2486            | 2486     | 12430 |

__Evaluation__

|                        | Central | Balear | Nord | Nord-Occidental | Valencià | Total |
|------------------------|---------|--------|------|-----------------|----------|-------|
| Model 1 (100% Central) | 2486    | 2486   | 2486 | 2486            | 2486     | 12430 |
| Model 2 (80% Central)  | 2486    | 2486   | 2486 | 2486            | 2486     | 12430 |
| Model 3 (50% Central)  | 2486    | 2486   | 2486 | 2486            | 2486     | 12430 |
| Model 4 (20% Central)  | 2486    | 2486   | 2486 | 2486            | 2486     | 12430 |

For now, this assumes that we'll only use the validated data with dialect annotations. If 152 hours of training data isn't enough to get good results on the ASR, we can look into making a dialect identifier. But, since the focus is more on the composition/bias in the train set rather than the quality of the ASR overall, I think we can proceed without using the unlabeled data. 
