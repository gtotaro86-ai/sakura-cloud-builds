"""Release evaluated upstream lazy graphs after one tensor, retaining small metadata."""
from contextlib import contextmanager

@contextmanager
def bounded_eager(tensor):
    from gguf.lazy import LazyBase
    nodes=[];seen=set()
    def visit(value):
        if isinstance(value,LazyBase):
            if id(value) in seen:return
            seen.add(id(value));nodes.append((value,value._args,value._data))
            visit(value._args);visit(value._kwargs)
        elif isinstance(value,(list,tuple)):
            for v in value:visit(v)
        elif isinstance(value,dict):
            for v in value.values():visit(v)
    visit(tensor)
    try:yield LazyBase.to_eager(tensor)
    finally:
        for node,args,data in nodes:node._args=args;node._data=data
