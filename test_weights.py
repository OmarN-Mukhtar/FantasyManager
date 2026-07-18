"""Runnable check for the prediction and sentiment weighting math."""
from prediction.predictor import weighted_total
from sentiment.sentiment_analyzer import recency_weight

# Neutral difficulty (FDR 3) = pure 0.8/GW time decay
assert abs(weighted_total([(10, 3), (10, 3)]) - 18.0) < 1e-9

# Closer fixtures weigh more
assert weighted_total([(10, 3), (0, 3)]) > weighted_total([(0, 3), (10, 3)])

# Easier opponents weigh more
assert weighted_total([(10, 2)]) > weighted_total([(10, 3)]) > weighted_total([(10, 5)])

# Empty fixture list is worth nothing
assert weighted_total([]) == 0

# Fresh news outweighs old; half-life is 3 days
assert recency_weight(0) == 1.0
assert abs(recency_weight(3) - 0.5) < 1e-9
assert recency_weight(6) < recency_weight(1)
assert recency_weight(-1) == 1.0  # clock skew can't inflate a weight

print("weights OK")
