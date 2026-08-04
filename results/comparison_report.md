# Hallucination Detection Results

## Overall

| Model | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| gemma2b | 473 | 0.524 | 0.201 | 0.620 | 0.303 |
| gpt-4-turbo | 478 | 0.745 | 0.336 | 0.537 | 0.413 |
| gpt-4o | 478 | 0.818 | 0.476 | 0.863 | 0.613 |
| llama1b | 478 | 0.462 | 0.133 | 0.400 | 0.199 |
| minicheck | 478 | 0.812 | 0.422 | 0.338 | 0.375 |
| our-model-retrained | 478 | 0.757 | 0.327 | 0.425 | 0.370 |
| our-model | 478 | 0.818 | 0.465 | 0.588 | 0.519 |
| summac | 478 | 0.730 | 0.212 | 0.225 | 0.218 |

## gemma2b by aspect

| Aspect | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Contribution | 106 | 0.547 | 0.308 | 0.870 | 0.455 |
| Main Results | 147 | 0.483 | 0.104 | 0.304 | 0.156 |
| Methodology | 137 | 0.569 | 0.164 | 0.556 | 0.253 |
| Motivation | 47 | 0.596 | 0.385 | 0.769 | 0.513 |
| Paper Goal | 36 | 0.361 | 0.080 | 1.000 | 0.148 |

## gpt-4-turbo by aspect

| Aspect | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Contribution | 108 | 0.852 | 0.643 | 0.750 | 0.692 |
| Main Results | 150 | 0.667 | 0.114 | 0.174 | 0.138 |
| Methodology | 137 | 0.715 | 0.244 | 0.556 | 0.339 |
| Motivation | 47 | 0.872 | 0.769 | 0.769 | 0.769 |
| Paper Goal | 36 | 0.694 | 0.091 | 0.500 | 0.154 |

## gpt-4o by aspect

| Aspect | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Contribution | 108 | 0.870 | 0.647 | 0.917 | 0.759 |
| Main Results | 150 | 0.853 | 0.512 | 0.913 | 0.656 |
| Methodology | 137 | 0.737 | 0.286 | 0.667 | 0.400 |
| Motivation | 47 | 0.957 | 0.923 | 0.923 | 0.923 |
| Paper Goal | 36 | 0.639 | 0.133 | 1.000 | 0.235 |

## llama1b by aspect

| Aspect | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Contribution | 108 | 0.426 | 0.183 | 0.458 | 0.262 |
| Main Results | 150 | 0.447 | 0.071 | 0.217 | 0.108 |
| Methodology | 137 | 0.482 | 0.156 | 0.667 | 0.253 |
| Motivation | 47 | 0.511 | 0.222 | 0.308 | 0.258 |
| Paper Goal | 36 | 0.500 | 0.000 | 0.000 | 0.000 |

## minicheck by aspect

| Aspect | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Contribution | 108 | 0.852 | 1.000 | 0.333 | 0.500 |
| Main Results | 150 | 0.840 | 0.462 | 0.261 | 0.333 |
| Methodology | 137 | 0.723 | 0.237 | 0.500 | 0.321 |
| Motivation | 47 | 0.787 | 0.800 | 0.308 | 0.444 |
| Paper Goal | 36 | 0.944 | 0.000 | 0.000 | 0.000 |

## our-model-retrained by aspect

| Aspect | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Contribution | 108 | 0.843 | 0.621 | 0.750 | 0.679 |
| Main Results | 150 | 0.733 | 0.207 | 0.261 | 0.231 |
| Methodology | 137 | 0.745 | 0.242 | 0.444 | 0.314 |
| Motivation | 47 | 0.681 | 0.333 | 0.154 | 0.211 |
| Paper Goal | 36 | 0.750 | 0.000 | 0.000 | 0.000 |

## our-model by aspect

| Aspect | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Contribution | 108 | 0.870 | 0.727 | 0.667 | 0.696 |
| Main Results | 150 | 0.827 | 0.455 | 0.652 | 0.536 |
| Methodology | 137 | 0.759 | 0.273 | 0.500 | 0.353 |
| Motivation | 47 | 0.745 | 0.545 | 0.462 | 0.500 |
| Paper Goal | 36 | 0.944 | 0.500 | 0.500 | 0.500 |

## summac by aspect

| Aspect | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Contribution | 108 | 0.778 | 0.500 | 0.250 | 0.333 |
| Main Results | 150 | 0.720 | 0.087 | 0.087 | 0.087 |
| Methodology | 137 | 0.650 | 0.188 | 0.500 | 0.273 |
| Motivation | 47 | 0.723 | 0.500 | 0.077 | 0.133 |
| Paper Goal | 36 | 0.944 | 0.000 | 0.000 | 0.000 |
