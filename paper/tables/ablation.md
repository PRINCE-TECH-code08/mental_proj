| Experiment                     |   RF MAE |   RF sd |   RF dMAE vs E |   Ridge MAE |
|:-------------------------------|---------:|--------:|---------------:|------------:|
| A: Demographics                |   0.9803 |  0.0312 |         0.6334 |      1.0168 |
| B: Demographics + Social media |   0.3523 |  0.0116 |         0.0054 |      0.564  |
| C: Lifestyle                   |   0.5556 |  0.0182 |         0.2087 |      0.5735 |
| D: Social media + Lifestyle    |   0.3955 |  0.0149 |         0.0486 |      0.536  |
| E: All features (baseline set) |   0.3469 |  0.0108 |         0      |      0.5239 |
| F: All + engineered            |   0.3747 |  0.0116 |         0.0278 |      0.525  |
| G: All minus Stress_Level      |   0.3494 |  0.0121 |         0.0025 |      0.5258 |
| H: Stress_Level only           |   0.6223 |  0.019  |         0.2753 |      0.6212 |
| I: All minus Social media      |   0.419  |  0.0128 |         0.0721 |      0.5483 |
| J: All minus Lifestyle         |   0.3452 |  0.0123 |        -0.0017 |      0.5512 |
| K: All minus Demographics      |   0.3907 |  0.0143 |         0.0438 |      0.5345 |