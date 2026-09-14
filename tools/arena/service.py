"""One bounded tick, then publish summaries. Invoked by the host timer."""
import sys
from pathlib import Path
from . import controller, store
from .publish import publish


def main(root):
    root=Path(root)
    with store.locked(root):
        from .continuous import sync
        sync(root)
        print(controller.tick(root),flush=True)
        publish(root)


if __name__=='__main__':
    main(sys.argv[1])
