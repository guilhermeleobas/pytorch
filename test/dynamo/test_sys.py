# Owner(s): ["module: dynamo"]
import sys

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import EagerAndRecordGraphs

from .utils import make_dynamo_test


class SysTests(torch._dynamo.test_case.TestCase):
    def test_exc_info(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise ValueError
            except Exception:
                typ, _, _ = sys.exc_info()
                if typ is ValueError:
                    return t.sin()
                else:
                    return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())


class CPythonActiveExceptionTests(torch._dynamo.test_case.TestCase):
    @make_dynamo_test
    def test_exc_info_no_exception(self):
        self.assertEqual(sys.exc_info(), (None, None, None))

    @make_dynamo_test
    def test_sys_exception_no_exception(self):
        self.assertEqual(sys.exception(), None)

    @make_dynamo_test
    def test_exc_info_with_exception_instance(self):
        def f():
            raise ValueError(42)

        try:
            f()
        except Exception as e_:
            e = e_
            exc_info = sys.exc_info()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc_info[0], ValueError)
        self.assertIs(exc_info[1], e)
        self.assertIs(exc_info[2], e.__traceback__)

    @make_dynamo_test
    def test_exc_info_with_exception_type(self):
        def f():
            raise ValueError

        try:
            f()
        except Exception as e_:
            e = e_
            exc_info = sys.exc_info()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc_info[0], ValueError)
        self.assertIs(exc_info[1], e)
        self.assertIs(exc_info[2], e.__traceback__)

    @make_dynamo_test
    def test_sys_exception_with_exception_instance(self):
        def f():
            raise ValueError(42)

        try:
            f()
        except Exception as e_:
            e = e_
            exc = sys.exception()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc, e)

    @make_dynamo_test
    def test_sys_exception_with_exception_type(self):
        def f():
            raise ValueError

        try:
            f()
        except Exception as e_:
            e = e_
            exc = sys.exception()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc, e)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
