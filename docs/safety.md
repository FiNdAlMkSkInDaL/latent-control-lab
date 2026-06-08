# Safety Boundary

This MVP deliberately avoids dangerous actuator surfaces.

- No arbitrary shell execution.
- No host filesystem actions beyond writing local datasets/artifacts during development.
- No network actions in the toy application.
- No model-generated commands.
- No parser-driven command execution.
- Low-confidence, low-margin, and high-distance predictions can abstain.

The toy application is intentionally a bounded state machine rather than a real system controller.
