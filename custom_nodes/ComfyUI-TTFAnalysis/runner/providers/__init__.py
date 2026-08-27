"""
Signal providers. One module per signal; each calls registry.register() at
import time and registry.load_all() imports every module here.

There is deliberately no list of signals in this file. Adding a provider is
adding a file — if you find yourself editing anything outside providers/ to add
one, the abstraction has leaked and that is the bug.

Registration order = module iteration order (alphabetical by module name), and
the scheduler uses it as the tie-break in its topological sort. It is therefore
part of the byte-identity contract, not cosmetic.
"""
