"""osc2carla: an OpenSCENARIO 2.1 to CARLA compiler.

The package follows the three-stage pipeline of the paper:
- ``osc2carla.frontend`` - ANTLR4 parsing + typed AST construction.
- ``osc2carla.middle``   - two-pass semantic analyser (symbol resolution).
- ``osc2carla.backend``  - py_trees behaviour-tree generation and CARLA glue.
"""

__version__ = "0.1.0"
