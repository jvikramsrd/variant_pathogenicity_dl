"""vpdl — variant pathogenicity, dataset-separable.

The organising constraint: every data source is independently loadable,
trainable and evaluable, because the question this codebase exists to answer is
whether pooling sources beats training on each alone.

See docs/v2/ARCHITECTURE.md for the layering, and
tests/regression/test_landmines.py for the defects this design is required to
make impossible.
"""

__version__ = "0.1.0.dev0"
