
import torch

class Foo():

    def __init__(self, a):
        self.a = a

    def __getattribute__(self, name):
        return super().__getattribute__(name)

@torch.compile(backend="eager", fullgraph=True)
def fn(t):
    f = Foo(3)
    return t.sin()

t = torch.randn(2)
fn(t)
