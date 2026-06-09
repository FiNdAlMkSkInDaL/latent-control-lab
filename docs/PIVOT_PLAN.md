# Pivot Plan

## Why simplify

The earlier repo proved the zero-generation routing architecture, but too much
of the story was wrapped around broad task-management language, larger model
ambitions, and optional Gemma/Colab paths. The new direction is deliberately
smaller and more visual: make the latent-vector control path obvious in one
screen, runnable on CPU, and easy to explain in a portfolio review.

## What is removed or de-emphasized

- Gemma, gated models, GPU-only paths, browser automation, and Colab are no
  longer part of the main narrative.
- The existing TaskFlow app remains as historical scaffolding and regression
  coverage, but it is no longer the primary demo.
- Text baselines and hard-eval utilities can stay where useful, but they should
  support honest comparison rather than define the product.
- Large local artifacts such as feature arrays, probe bundles, and model caches
  remain ignored.

## New visual demo

The primary demo is VectorBot: a small 2D grid-world controller. A command like
`go north` is fed through a frozen tiny transformer forward pass. A PyTorch hook
captures the pre-lm-head hidden state, the final non-padding token vector is
projected by a lightweight probe, and the gated enum action updates a sandboxed
grid state.

Runtime action selection follows only this path:

```text
natural language -> tokenizer -> frozen model forward pass -> hidden vector
-> linear probe/router -> confidence/OOD gate -> VectorBot action enum
-> VectorBotKernel.execute()
```

VectorBot supports `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`,
`TOGGLE_LIGHT`, `RESET`, and `ABSTAIN`/`NO_OP`.

## Why tiny models

Tiny and lightweight causal language models make the project demoable without a
GPU, gated-model license, or long setup. `distilgpt2` is the primary live model;
`sshleifer/tiny-gpt2` and `hf-internal-testing/tiny-random-gpt2` are plumbing
targets for CI and smoke checks. The goal is not state-of-the-art language
understanding. The goal is a clean, inspectable proof that transformer
activations can be used directly as a control signal.

## What counts as success

- The default README and scripts center VectorBot, not Gemma or TaskFlow.
- The core invariant is guarded: no `model.generate()` in routing, no generated
  commands, no JSON/tool-call parsing, and no keyword/regex action parser.
- A deterministic VectorBot kernel exposes clear state diffs, ASCII rendering,
  action history, and JSON-serializable snapshots.
- Dataset generation, feature extraction, probe training, scripted demo, and
  visual asset generation are one-command runnable on CPU.
- If model download is blocked, the repo still produces clearly marked fallback
  plumbing artifacts and replay/static visuals without claiming real-model
  metrics.
- A GitHub visitor sees a visual grid demo, confidence bars, a latent scatter
  plot, a transcript, and honest limitations.
