# engine.__init__.py
"""engine"""

# from collections import abc
# from pathlib import Path

# __all__ = ['get_project_rootdir']


# def _takewhile_inclusive(predicate, it: abc.Iterable):
#     """Inclusive version of itertools.takewhile
#     see https://stackoverflow.com/a/70762559/1165112
#     """
#     for x in it:
#         if predicate(x):
#             yield x
#         else:
#             yield x
#             break


# def get_project_rootdir() -> Path:
#     """Get fully resolved rootdir of the project e.g. /foo/resevol/"""

#     return Path(
#         *list(
#             _takewhile_inclusive(lambda x: x != "resevol", Path.cwd().resolve().parts)
#         )
#     )
