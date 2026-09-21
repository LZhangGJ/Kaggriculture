"""One reusable shared-memory arena per CPU actor; queues carry descriptors only."""
from multiprocessing import shared_memory
import numpy as np


class ArrayArena:
    def __init__(self, name=None, size=32*1024*1024):
        self.owner = name is None
        self.shm = shared_memory.SharedMemory(name=name, create=self.owner, size=size)

    def write(self, value):
        offset = 0
        def encode(v):
            nonlocal offset
            if isinstance(v, np.ndarray):
                a = np.ascontiguousarray(v)
                offset = (offset+63)//64*64
                if offset+a.nbytes > self.shm.size:
                    raise BufferError('Actor numeric response exceeds shared arena; increase --arena-mib')
                spec = {'__array__': (offset, a.shape, a.dtype.str)}
                np.ndarray(a.shape, a.dtype, buffer=self.shm.buf, offset=offset)[...] = a
                offset += a.nbytes
                return spec
            if isinstance(v, np.generic): return v.item()
            if isinstance(v, dict): return {k: encode(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)): return [encode(x) for x in v]
            return v
        spec = encode(value)
        return spec, offset

    def read(self, spec, copy=True):
        if isinstance(spec, dict) and '__array__' in spec:
            offset, shape, dtype = spec['__array__']
            view = np.ndarray(shape, dtype, buffer=self.shm.buf, offset=offset)
            return view.copy() if copy else view
        if isinstance(spec, dict): return {k:self.read(v, copy=copy) for k,v in spec.items()}
        if isinstance(spec, list): return [self.read(v, copy=copy) for v in spec]
        return spec

    def close(self):
        self.shm.close()
        if self.owner: self.shm.unlink()


class BatchTransfer:
    """Reusable pinned packing buffers; event protects each buffer from overwrite."""
    def __init__(self, device):
        import torch
        self.device=torch.device(device); self.buffers={}

    def array(self, rows, pad=False):
        import torch
        a=np.asarray(rows[0]); shape=(len(rows),*a.shape)
        if pad: shape=(len(rows),max(len(r) for r in rows),*a.shape[1:])
        if self.device.type!='cuda':
            if not pad: return torch.as_tensor(np.stack(rows),device=self.device)
            out=np.zeros(shape,a.dtype)
            for i,row in enumerate(rows): out[i,:len(row)]=row
            return torch.as_tensor(out,device=self.device)
        key=(shape,a.dtype.str)
        if key not in self.buffers:
            buf=torch.empty(shape,dtype=torch.from_numpy(np.empty((),a.dtype)).dtype,pin_memory=True)
            self.buffers[key]=(buf,torch.cuda.Event())
        buf,event=self.buffers[key]
        event.synchronize()
        if pad:
            out=buf.numpy();out.fill(0)
            for i,row in enumerate(rows): out[i,:len(row)]=row
        else: np.stack(rows,out=buf.numpy())
        result=buf.to(self.device,non_blocking=True)
        event.record()
        return result

    def mapping(self, rows, pad_names=()):
        import torch
        if self.device.type!='cuda':
            return {k:self.array([r[k] for r in rows],pad=k in pad_names) for k in rows[0]}
        groups={}
        for name in rows[0]:
            first=np.asarray(rows[0][name]);shape=(len(rows),*first.shape)
            if name in pad_names:
                shape=(len(rows),max(len(r[name]) for r in rows),*first.shape[1:])
            groups.setdefault(first.dtype.str,[]).append((name,shape))
        result={}
        for dtype,fields in groups.items():
            key=('packed',dtype,tuple(fields))
            if key not in self.buffers:
                count=sum(int(np.prod(shape)) for _,shape in fields)
                buf=torch.empty(count,dtype=torch.from_numpy(np.empty((),np.dtype(dtype))).dtype,pin_memory=True)
                self.buffers[key]=(buf,torch.cuda.Event())
            buf,event=self.buffers[key];event.synchronize()
            host=buf.numpy();offset=0
            for name,shape in fields:
                size=int(np.prod(shape));out=host[offset:offset+size].reshape(shape)
                if name in pad_names:
                    out.fill(0)
                    for i,row in enumerate(rows):out[i,:len(row[name])]=row[name]
                else:np.stack([r[name] for r in rows],out=out)
                offset+=size
            device=buf.to(self.device,non_blocking=True);event.record();offset=0
            for name,shape in fields:
                size=int(np.prod(shape));result[name]=device[offset:offset+size].reshape(shape);offset+=size
        return result

    def observations(self, rows):
        return self.mapping(rows,pad_names=('workers','worker_valid'))


def cpu_tree(tree):
    """Copy a tensor tree once per dtype; returned arrays own their CPU storage."""
    import torch
    groups={}; leaves=[]
    def gather(value):
        if isinstance(value,torch.Tensor):
            i=len(leaves);leaves.append(value)
            groups.setdefault((value.device,value.dtype),[]).append(i)
        elif isinstance(value,dict):
            for v in value.values():gather(v)
        elif isinstance(value,(list,tuple)):
            for v in value:gather(v)
        else:raise TypeError(type(value))
    gather(tree); arrays=[None]*len(leaves)
    for indices in groups.values():
        flat=torch.cat([leaves[i].reshape(-1) for i in indices]).cpu().numpy()
        offset=0
        for i in indices:
            size=leaves[i].numel()
            arrays[i]=flat[offset:offset+size].reshape(tuple(leaves[i].shape))
            offset+=size
    iterator=iter(arrays)
    def restore(value):
        if isinstance(value,torch.Tensor):return next(iterator)
        if isinstance(value,dict):return {k:restore(v) for k,v in value.items()}
        return type(value)(restore(v) for v in value)
    return restore(tree)
