# CV Bullets

- Designed and shipped **Tiny Latent Control Lab**: a zero-generation system that routes natural language to sandboxed application actions exclusively via frozen LLM hidden-state vectors and a lightweight probe (no `model.generate()`, no tool-call parsing).
- Built rich interactive demos (Streamlit + Gradio) featuring live HTML grid visualization with movement trails, side-by-side latent-router vs. simulated generation-router safety comparison, and interactive latent-space explorer.
- Extended the work with activation-space concept vectors: computed contrastive directional vectors in representation space and demonstrated steering effects on probe outputs (analysis-only, LLM remains frozen).
- Delivered end-to-end reproducible pipeline (dataset → features → calibrated probe → visuals + GIFs) on CPU with `distilgpt2`, plus strong ABSTAIN OOD behavior (0.964 P/R) and comprehensive guard tests.
- Produced portfolio-grade assets: animated routing GIFs, composite diagrams, model/probe cards, and rigorous evaluation (baselines, prompt robustness, OOD, data efficiency).
