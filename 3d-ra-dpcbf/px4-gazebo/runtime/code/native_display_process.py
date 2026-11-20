"""Keep GUI geometry and protobuf allocation outside the control process."""
import multiprocessing as mp
import queue,time

def _display(out,start,goal,parameters,mailbox,ready,initial):
    from native_overlay import NativeOverlay
    overlay=NativeOverlay(out,start,goal,parameters)
    if initial is not None:overlay.prepare(*initial)
    ready.set()
    try:
        while True:
            item=mailbox.get()
            if item[0]=='close':overlay.close(item[1],item[2]);break
            overlay.update(*item[1])
    finally:pass

class DisplayProcess:
    def __init__(self,out,start,goal,parameters,initial=None):
        context=mp.get_context('spawn');self.queue=context.Queue(maxsize=1);self.last=-1.;self.closed=False;ready=context.Event()
        # Not daemon: this owned process has its own isolated RPC child.
        self.process=context.Process(target=_display,args=(str(out),start,goal,parameters,self.queue,ready,initial));self.process.start()
        deadline=time.monotonic()+120.
        while time.monotonic()<deadline and self.process.is_alive():
            if ready.wait(.25):break
        if not ready.is_set():raise RuntimeError("Native display initialization did not finish")

    def update(self,t,*args):
        if t-self.last<.049:return
        self.last=t;item=('update',(t,*args))
        try:self.queue.put_nowait(item)
        except queue.Full:
            try:self.queue.get_nowait()
            except queue.Empty:pass
            try:self.queue.put_nowait(item)
            except queue.Full:pass

    def close(self,completed=False,failure=None):
        if self.closed:return
        self.closed=True
        while self.process.is_alive():
            try:self.queue.put(('close',completed,failure),timeout=.15);break
            except queue.Full:
                try:self.queue.get_nowait()
                except queue.Empty:pass
        self.process.join(timeout=5.)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=1.)
        self.queue.close();self.queue.join_thread()
