## 2026-03-14 - Enhanced CLI Feedback and Visual Results
**Learning:** In CLI-based ML pipelines, standard library logs (like Keras progress bars) can clash with custom reporting. Silencing internal logs and implementing custom single-line progress updates (`\r`) significantly improves tool professionalism and usability.
**Action:** Always set `verbose=0` on internal model calls when building a custom CLI interface to maintain control over terminal output.

**Learning:** Presenting probability distributions with simple text can be hard to scan. Visual aids like ASCII bar charts provide immediate intuitive feedback on model confidence.
**Action:** Use ASCII/UTF-8 characters (like █ and ░) to create simple bar charts for probability distributions in CLI tools.

## 2026-03-20 - Multi-Tiered Confidence Indicators
**Learning:** Generic "Result" outputs in ML tools can be misleading if the model is uncertain. Categorizing results into visual tiers (Confident, Uncertain, Low Confidence) using color and text labels improves interpretability and accessibility.
**Action:** Always provide a confidence-aware status indicator for ML predictions in CLI outputs.
