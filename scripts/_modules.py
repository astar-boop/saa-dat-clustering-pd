"""Load a script of the chain as a module, to reuse its functions and constants.

Script names begin with a number and cannot be imported with `import`.
The scripts that are reused keep their body in main(), under the
`if __name__ == "__main__"` guard: loading them here defines functions and constants
and does not run the analysis.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_script(name_file, name):
    """The file `name_file` of this folder as a module named `name`."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
