"""Fixed 18ms publication phase within unchanged 20ms cycle; own-core spin releases Python GIL.

This consumes idle CPU instead of relying on sub-ms sleep wakeups. It is not
a hard-real-time guarantee. Complete-cycle wall duration remains mandatory.
"""
from pathlib import Path
import ctypes,time
lib=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'build/libperiodic_publish75.so'))
lib.cbf_wait_until.argtypes=[ctypes.c_double];lib.cbf_wait_until.restype=None
def wait_until(deadline):
    begin=time.monotonic();cpu=time.thread_time_ns()
    lib.cbf_wait_until(float(deadline))
    return dict(cpu_ms=(time.thread_time_ns()-cpu)*1e-6,wall_ms=(time.monotonic()-begin)*1000)
