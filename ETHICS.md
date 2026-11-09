# Ethics and Responsible Use

**Purpose.** Educational/research demonstration. It predicts a dataset-specific wellbeing score; it does not diagnose depression, anxiety or any disorder, does not replace clinicians, and gives no treatment or medication advice.

| Topic | Position / mitigation |
|---|---|
| Privacy | Inputs are sent to the API to compute a score; request bodies are not logged and nothing is stored server-side. History lives in the user's browser only (localStorage), stores no age/country/gender, and can be exported/deleted. No accounts or identifiers. |
| Consent / provenance | Training-data consent and origin are NOT ESTABLISHED (see DATA_CARD). Do not present results as findings about real students. |
| Sensitive data | Health-related self-reports. Use HTTPS in deployment; keep `ALLOWED_ORIGINS` restricted; do not add analytics that capture inputs. |
| Bias | Only two gender values, ages 18-24, imbalanced countries/platforms. Country has non-trivial model importance (demographic proxy) - consider dropping it in any real use. Subgroup errors are reported, **fairness is not claimed** (`docs/fairness.md`). |
| False reassurance | A "higher than most" position or a tight interval is not evidence of good health. Interval coverage is marginal and not guaranteed for every group (e.g. India 0.837 on 129 test rows). |
| False alarm | A "lower" position may distress users; wording is descriptive/relative and always includes a support message. |
| Causal misreading | Explanations and sensitivity results are associations in a model, not advice that changing a habit changes wellbeing. Causal evidence needs designs such as the quasi-experimental study of Braghieri et al. (2022) - not this dataset. |
| Trend misreading | Predictions vary with input noise (p95 change 0.55 for 0.25 h noise, assumption-based); small changes are within noise and never a clinical change. |
| Misuse | Must not be used for screening, admissions, employment, insurance or surveillance of students. |
| Responsible deployment | Keep disclaimers visible; show interval + warnings; retrain/re-validate before any change of population; add crisis resources appropriate to the deployment region. |
