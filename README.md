# lm_eval
Custom LLM Eval tasks. Intended to be run against a local llama-server endpoint.

## tmp_handling

This eval tests that the model will prefer safe temporary files - either saved
outside of systemwide `/tmp/`, or atomically-created and randomly-named using
the appropriate methods.

