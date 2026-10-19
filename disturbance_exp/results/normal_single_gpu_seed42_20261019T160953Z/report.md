# Interrupted initial dry-run

Status: `failed`.

The first implementation initialized the tokenizer through Transformers and remained CPU-bound for over one minute. The process was stopped before sending requests or starting injectors. Tokenizer loading was changed to use the local `tokenizer.json` directly; subsequent scenario dry-runs completed successfully.
