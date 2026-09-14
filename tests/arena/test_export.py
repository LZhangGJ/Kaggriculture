import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from tools.arena.kaggle_export import unpack, pack


class ExportTest(unittest.TestCase):
    def tar(self, name, kind=tarfile.REGTYPE):
        output=io.BytesIO()
        with tarfile.open(fileobj=output,mode='w:gz') as archive:
            item=tarfile.TarInfo(name);item.type=kind
            item.size=1 if kind==tarfile.REGTYPE else 0
            archive.addfile(item,io.BytesIO(b'x'))
        return output.getvalue()

    def test_unsafe_tar(self):
        for name,kind in [('../main.py',tarfile.REGTYPE),('main.py',tarfile.SYMTYPE)]:
            with self.assertRaises(ValueError):unpack(self.tar(name,kind))

    def test_nested_main(self):
        self.assertEqual(unpack(self.tar('agent/main.py')),{'main.py':b'x'})

    def test_reproducible_package(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.zip',Path(d)/'b.zip'
            pack({'main.py':b'agent = None'},a);pack({'main.py':b'agent = None'},b)
            self.assertEqual(a.read_bytes(),b.read_bytes())
