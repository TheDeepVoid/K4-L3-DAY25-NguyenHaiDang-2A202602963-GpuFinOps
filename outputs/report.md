# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 4204.86 Wh
- Carbon per query: 12036.616 gCO2e
- Cheapest+cleanest region: europe-north1

### Reasoning Budget & Carbon Impact (Extension 4)

- Reasoning traffic share: 8.4% requests (16.5% tokens)
- Reasoning energy share: 29,787.7 Wh (94.0% total inference Wh)
- 10% traffic cap evaluation: traffic is currently 8.4% (<= 10% cap) => cap savings = $0.00 / 0.0 Wh
- Recommendation: Gate reasoning to complex evaluation tasks (eval, deep research); standard chat/search routes to cascade

_Figures are June-2026 as-of snapshots; re-baseline before acting._