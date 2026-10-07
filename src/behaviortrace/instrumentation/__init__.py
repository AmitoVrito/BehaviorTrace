"""the pipeline - training-time instrumentation.


Goal: log per-step rollouts, advantages/rewards, compact per-rollout
gradient embeddings, and periodic checkpoint references at sub-few-percent
overhead (an earlier experiment target: < 5% wall-clock).
"""
