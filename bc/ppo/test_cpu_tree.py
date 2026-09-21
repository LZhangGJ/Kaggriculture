import unittest,os
import torch
import numpy as np
from ppo.transport import cpu_tree, BatchTransfer

class PackedTraceTests(unittest.TestCase):
    def test_types_views_and_storage_lifetime(self):
        device=os.environ.get('PARITY_DEVICE','cpu')
        source=torch.arange(24,device=device,dtype=torch.float32).reshape(4,6)
        tree=[({'float':source[:,::2],'int':torch.tensor([2**40,7],device=device),
                'bool':torch.tensor([True,False],device=device)},source[0,0],
                torch.empty(0,device=device,dtype=torch.float64))]
        result=cpu_tree(tree)
        expected=source[:,::2].cpu().numpy().copy()
        source.fill_(-1)
        np.testing.assert_array_equal(result[0][0]['float'],expected)
        np.testing.assert_array_equal(result[0][0]['int'],[2**40,7])
        self.assertEqual(result[0][0]['bool'].dtype,np.dtype(bool))
        self.assertEqual(result[0][1].shape,())
        self.assertEqual(result[0][2].shape,(0,))
        self.assertIsInstance(result[0],tuple)

class PackedObservationTests(unittest.TestCase):
    def test_padding_dtype_and_reuse(self):
        transfer=BatchTransfer(os.environ.get('PARITY_DEVICE','cpu'))
        rows=[dict(x=np.arange(12,dtype=np.float32).reshape(3,4),
                   workers=np.ones((n,4),np.float32),valid=np.ones(n,bool),
                   count=np.asarray(n,np.int64)) for n in (1,3)]
        first=transfer.mapping(rows,pad_names=('workers','valid'))
        rows[0]['x'].fill(99);rows[1]['x'].fill(88)
        second=transfer.mapping(rows,pad_names=('workers','valid'))
        np.testing.assert_array_equal(first['x'][0].cpu(),np.arange(12).reshape(3,4))
        np.testing.assert_array_equal(second['x'][0].cpu(),np.full((3,4),99))
        np.testing.assert_array_equal(first['valid'].cpu(),[[True,False,False],[True,True,True]])
        self.assertEqual(first['count'].dtype,torch.int64)
        self.assertEqual(first['workers'].shape,(2,3,4))

if __name__=='__main__':unittest.main()
