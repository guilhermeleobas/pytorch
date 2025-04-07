"""
Python polyfills for sys
"""

from __future__ import annotations

import sys

from ..decorators import substitute_in_graph


__all__ = [
    "intern",
    "getrecursionlimit",
]


@substitute_in_graph(sys.intern, can_constant_fold_through=True)
def intern(string: str, /) -> str:
    return string


@substitute_in_graph(sys.getrecursionlimit, can_constant_fold_through=True)
def getrecursionlimit() -> int:
    return sys.getrecursionlimit()


@substitute_in_graph(sys.gettrace, can_constant_fold_through=True)
def gettrace():
    return sys.gettrace()


@substitute_in_graph(sys.settrace, can_constant_fold_through=True)
def settrace(function, /) -> None:
    return sys.settrace(function)
