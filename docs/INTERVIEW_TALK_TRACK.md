# Interview Talk Track

## 30-Second Explanation

I built a prototype that routes natural-language requests into a small task app
without asking the LLM to generate commands. It runs a frozen Hugging Face causal
LM in a forward pass, captures the pre-`lm_head` hidden state with a PyTorch
hook, feeds that vector into a linear probe, and executes only typed enum
actions in a sandboxed state machine.

## 2-Minute Explanation

Most agent stacks generate JSON or tool-call text and then parse it. I wanted to
test a different interface: can software react directly to latent model states?
The repo implements a bounded task controller with five actions and `ABSTAIN`.
The runtime path is `text -> tokenizer -> frozen LM -> hook -> vector -> probe
-> OOD gate -> Action enum -> TaskFlowKernel`. I verified it locally with
`distilgpt2`, then hardened the evaluation after discovering the first dataset
was too template-leaky. The final package includes a cleaner V2 dataset, prompt
augmentation, calibrated OOD gates, TF-IDF baselines, model comparison, and
Colab instructions for Gemma.

## Architecture Walkthrough

1. Tokenize the user request with a fixed router prompt.
2. Run a frozen causal LM forward pass.
3. Hook the tensor entering `lm_head`.
4. Select the final-token hidden vector.
5. Classify with a lightweight linear probe.
6. Apply confidence/OOD gates.
7. Execute only a typed enum action through `TaskFlowKernel`.

## Why No Generation?

Generation introduces a text protocol: the model emits JSON, code, SQL, or tool
calls, and the app has to parse them. This project tests whether a vector-facing
boundary can avoid that text-command layer for a bounded action space.

## Why Hooks?

Hooks make the feature boundary explicit. The app never needs logits or sampled
tokens; it needs the hidden representation immediately before unembedding.

## Why Linear Probes?

A linear probe is intentionally simple. If action labels are separable with a
small projection layer, the experiment is easier to inspect and harder to
overclaim than a large fine-tuned classifier.

## Why Interesting If TF-IDF Is Competitive?

TF-IDF being competitive is important evidence that the toy dataset is small and
surface-form easy. The interesting part is not benchmark dominance; it is the
architecture: a real hidden-state route into typed software actions with no
generated command parser.

## Limitations

- Small action space.
- Small synthetic and curated datasets.
- Prompt template still matters.
- V2 improves hard-negative rejection but still struggles with executable hard
  paraphrases.
- Gemma is documented as a Colab path, not claimed as locally run.

## Likely Questions

**Q: Is this safer than tool calling?**  
It removes generated command parsing, but safety still depends on the action
space, calibration, and evaluation. I kept the app sandboxed and tiny.

**Q: Why not just use a text classifier?**  
For this dataset, TF-IDF is competitive. The point was to explore a latent
software interface and evaluate it honestly, not to claim text baselines are
obsolete.

**Q: What would you improve next?**  
Larger human-authored hard negatives, Gemma-scale runs, better OOD calibration,
and tests for prompt/layer invariance.

**Q: How do you know it does not call `generate()`?**  
There is an AST guard test over the core route, and the extractor uses
`model(**enc, use_cache=False)` under `torch.inference_mode()`.
