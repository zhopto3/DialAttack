# The Impact of Dialect Varition on Robust Automatic Speech Recognition for Catalan

CommonVoice 18.0 has the following "macro-dialect" composition (L2 data removed):

|                        | Central | Balear | Nord  | Nord-Occidental | Valencià | Unknown |
|------------------------|---------|--------|-------|-----------------|----------|---------|
| Validated Sample Count | 761017  | 24864  | 29970 | 74452           | 83742    | 598786  |
| Duration (Hours)       | 1155.31 | 38.51  | 47.04 | 100.17          | 120.04   | 912.81  |

**The values above are calculated after filtering out repeated recordings of the same sentence from the same macro-dialect. If the same sentence was recorded by speakers from different macro-dialects, all recordings of that sentence from different macro-dialects were left in the data. 

Working in a way that tries to use as much data as possible, we aimed to train ASR models that vary in how biased toward the Central dialect they are. Given that the Balear dialect represents the "lower limit" in terms of data, we worked from the assumption that at most, we can have ~30.4 hours of Balear training data (80% of 38 hr), and 3.8 hours each of Balear development and test data (10% each). 

We trained four models using the following dialect compositions. Training data ranged from a condition where all the fine-tuning data is in the Central dialect (Model 1) to a condition where the fine-tuning data is perfectly balanced (Model 4). 

In terms of hours of data:

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

## To run model training: 

After splitting a sample from the common voice corpus, fine-tuning can be initialized with the script `./training/run_training.py`

For example:

`python3 ./training/run_training.py --experiment_name=central100_53m_v02 --model=XLSR53 --freeze_feature_extractor --prop_central=100`

## Adversarial Attacks

We train adversarial attacks on the fine-tuned models to study the impact of having multi-dialect fine-tuning data on adversarial robustness. Attacks can be trained using the script `./attack/launch_attack.py`:

`python3 ./attack/launch_attack.py --experiment_name=central100_53m_v02 --model=XLSR53 --lr=0.01 --regularizing_const=1.0`