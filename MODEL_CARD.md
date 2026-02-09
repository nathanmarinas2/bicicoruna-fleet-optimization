# Model Card: BiciCoruna Empty-Station Classifier

## Overview
This model predicts whether a station will be empty within a short horizon.
It is trained with transfer learning and fine-tuning on Coruna data.

## Intended Use
- Support rebalancing decisions and early alerts.
- Provide operational insights for system health.

## Data
- External pretraining data: Barcelona and Washington DC datasets.
- Fine-tuning data: Coruna tracking data.

## Target Definition
- Empty station: bikes available < configured `empty_threshold`.
- Horizon: `horizon_minutes` (configured by `horizon_shifts`).

## Metrics
Reported in training scripts:
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC (classification)

## Limitations
- Short time coverage for Coruna data.
- Potential seasonality and weather bias.
- Station behavior can change over time.

## Ethical Considerations
- No personal data is used.
- Predictions should support, not replace, operator decisions.

## Configuration
See config.yaml for thresholds and horizon settings.
