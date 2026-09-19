"""Windows job CPU hard cap plus whole-machine headroom feedback.

Only this process and its descendants are limited. Unrelated programs can still
raise whole-machine usage above 70%; sampling cannot guarantee an instant bound.
Win32 references: https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemtimes
"""
import ctypes,json,os,threading,time
from ctypes import wintypes
from pathlib import Path


class Rate(ctypes.Structure):
    _fields_=[('flags',wintypes.DWORD),('rate',wintypes.DWORD)]


class CpuJob:
    def __init__(self,percent=.1):
        if os.name!='nt': raise RuntimeError('This CPU cap requires Windows')
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype=wintypes.HANDLE
        self.api.GetCurrentProcess.restype=wintypes.HANDLE
        self.api.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype=wintypes.BOOL
        self.api.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        self.api.SetInformationJobObject.restype=wintypes.BOOL
        self.api.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
        self.api.QueryInformationJobObject.restype=wintypes.BOOL
        self.api.GetSystemTimes.argtypes=[ctypes.POINTER(wintypes.FILETIME)]*3
        self.api.GetSystemTimes.restype=wintypes.BOOL
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        self.set_rate(percent)
        if not self.api.AssignProcessToJobObject(self.handle,self.api.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
        # Keep the handle open until process exit. No unrelated process is assigned.

    def set_rate(self,percent):
        assert .01<=percent<=70
        info=Rate(0x1|0x4,max(1,int(percent*100)))
        if not self.api.SetInformationJobObject(self.handle,15,ctypes.byref(info),ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.rate=info.rate/100

    def query_rate(self):
        info=Rate()
        if not self.api.QueryInformationJobObject(self.handle,15,ctypes.byref(info),ctypes.sizeof(info),None):
            raise ctypes.WinError(ctypes.get_last_error())
        assert info.flags==5 and 1<=info.rate<=7000
        return info.rate/100

    def times(self):
        idle,kernel,user=wintypes.FILETIME(),wintypes.FILETIME(),wintypes.FILETIME()
        if not self.api.GetSystemTimes(ctypes.byref(idle),ctypes.byref(kernel),ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        def ticks(v): return (v.dwHighDateTime<<32)|v.dwLowDateTime
        return ticks(idle),ticks(kernel)+ticks(user),time.process_time()*10_000_000


def next_rate(total,own,running=False):
    other=max(0.,total-own)
    # Reserve fifteen points for unrelated bursts. Hysteresis prevents rapid restart
    # when an unrelated workload stays close to the user's 70% ceiling.
    proceed=total<(60 if running else 55)
    return (max(.1,min(70.,55.-other)) if proceed else .1),proceed


class CpuBudget:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.job=CpuJob(.1);self.gate=threading.Event();self.stop=threading.Event()
        self.error=None;self.wait_seconds=0.;self.thread=threading.Thread(target=self.monitor,daemon=True)
        self.thread.start()

    def record(self,status):
        temp=self.root/'CPU_LIMIT.tmp'
        temp.write_text(json.dumps(status,indent=2),encoding='utf8');temp.replace(self.root/'CPU_LIMIT.json')

    def monitor(self):
        try:
            previous=self.job.times();sample=0;running=False
            while not self.stop.wait(.5):
                current=self.job.times();dt=current[1]-previous[1]
                if dt<=0: continue
                total=min(100.,max(0.,100*(1-(current[0]-previous[0])/dt)))
                own=max(0.,100*(current[2]-previous[2])/dt);previous=current
                rate,proceed=next_rate(total,own,running);running=proceed;self.job.set_rate(rate)
                actual=self.job.query_rate()
                if proceed: self.gate.set()
                else: self.gate.clear()
                status=dict(status='RUNNING_WITH_CPU_LIMIT' if proceed else 'WAITING_FOR_CPU',pid=os.getpid(),
                    limit_percent=70,whole_machine_target_percent=55,pause_threshold_percent=60,resume_below_percent=55,total_cpu_percent=round(total,2),
                    training_cpu_percent=round(own,2),job_hard_cap_percent=actual,
                    verified_native_flags=5,logical_processors=os.cpu_count(),sample_seconds=.5,
                    heartbeat_unix=time.time(),waiting_seconds=round(self.wait_seconds,2),
                    scope='Job hard cap covers this process and children; other processes remain independent.')
                self.record(status)
                sample+=1
                if sample%10==0:
                    with (self.root/'CPU_USAGE.jsonl').open('a',encoding='utf8') as log: log.write(json.dumps(status)+'\n')
        except BaseException as exc:
            self.error=repr(exc)
            try: self.job.set_rate(.1)
            finally: self.gate.set()
            self.record(dict(status='CPU_LIMIT_FAILED',pid=os.getpid(),error=self.error,heartbeat_unix=time.time()))

    def wait(self):
        started=time.monotonic()
        while not self.gate.wait(1):
            if not self.thread.is_alive(): raise RuntimeError('CPU limit monitor stopped')
        self.wait_seconds+=time.monotonic()-started
        if self.error: raise RuntimeError(self.error)


if __name__=='__main__':
    # A deliberately small CPU load verifies actual OS throttling, not just flags.
    assert next_rate(75,5)==(.1,False)
    assert next_rate(55,5,True)==(5.,True)
    assert next_rate(59,10,True)==(6.,True)
    assert next_rate(59,0,False)==(.1,False)
    assert next_rate(54,0,False)==(1.,True)
    job=CpuJob(.5);assert job.query_rate()==.5
    first=job.times();started=time.monotonic();cpu_start=time.process_time()
    value=1
    while time.monotonic()-started<3:
        for _ in range(1000): value=(value*13+7)%1000000007
    elapsed=time.monotonic()-started;second=job.times()
    observed=(time.process_time()-cpu_start)/elapsed/(os.cpu_count() or 1)*100
    assert second[1]>first[1] and second[0]>=first[0]
    assert observed<2.,f'CPU cap not effective: {observed:.2f}%'
    print(json.dumps(dict(status='PASS',requested_cap_percent=.5,queried_cap_percent=job.query_rate(),
                         observed_cpu_percent=observed,seconds=elapsed,logical_processors=os.cpu_count())),flush=True)
