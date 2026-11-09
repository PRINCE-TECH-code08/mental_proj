# Research gap - CANDIDATE, pending completion of the literature survey

Novelty is **not claimed**. What this project contributes empirically, on this one dataset:
1. A **reproducible protocol** around a student baseline: exact reproduction, leak-free pipeline, nested CV (the baseline compared models on the test set).
2. **Honest negative results**: engineered ratio features did not help; tuning gave a modest gain within CV noise; Stress_Level is redundant for the model.
3. **Uncertainty** via cross-conformal intervals with empirical marginal and conditional coverage checks.
4. **Model-behaviour analyses**: Shapley attribution, sensitivity, and an input-noise floor showing why small changes in repeated predictions must not be read as change.
5. **Subgroup reporting** with sample-size flags and an explicit refusal to declare fairness.

Whether the combination (explainability + uncertainty + subgroup + noise-floor) is missing from prior student-wellbeing ML work is **NOT ESTABLISHED** until the survey in `docs/literature_review.md` is completed.
