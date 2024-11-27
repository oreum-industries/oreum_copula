# src.engine.base_handler.py
"""Common handler operations"""

from enum import Enum
from pathlib import Path

from oreum_core import curate


class BaseKind(str, Enum):
    """Establish the supported base kinds"""

    handler = "handler"  # a general handler for R&D, not part of the engine
    cleaner = "cleaner"
    explorer = "explorer"
    preparer = "preparer"
    trainer = "trainer"
    forecaster = "forecaster"


class BaseHandler:
    """Base handler for CLI actions to be inherited
    e.g. `super().__init__(*args, **kwargs)`
    """

    def __init__(self, kind: BaseKind, fqns: dict[str, Path], **kwargs):
        """
        Usage note: override kws directly on downstream instance e.g.
        def __init__(self, **kwargs):
            super().__init__(*args, **kwargs)
            ...
        """
        self.kind = kind
        self.fqns = fqns
        self.pltp = kwargs.pop("pltp", None)
        self.txtio = curate.SimpleStringIO(kind="txt")
        self.ppqio = curate.PandasParquetIO()
        self.mdl = None
        self.yhats = {}
