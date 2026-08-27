"""
ComfyUI-TTFAnalysis — analysis-layer node package.

The node wrapper lands in P3. Until then this file deliberately registers
nothing and imports nothing: the runner beneath it must stay importable and
runnable without ComfyUI present, which is what lets the harness be tested
outside the container.
"""
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
