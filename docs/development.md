# Development rules

1. Do not modify frozen datasets in place.
2. Do not commit model weights or local caches.
3. Keep generated results separate from source code.
4. Record the configuration and environment for reported experiments.
5. Do not change the pinned model revision silently.
6. Do not change the Stage 1 scoring protocol without updating the experiment documentation.
7. `legacy/` is archival; new work belongs in the canonical structure.
