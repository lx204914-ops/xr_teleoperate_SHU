"""Isolate blocking Gazebo service calls from Python telemetry callbacks.

Transport's Python binding can retain the GIL during synchronous requests.
This worker has no Python topic subscriptions; parent poll releases the GIL.
Request bytes, service types and physical clock verification are unchanged.
"""
import multiprocessing as mp
import sys,threading

def _serve(pipe):
    sys.path.append('/usr/lib/python3/dist-packages')
    from gz.transport13 import Node
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.double_v_pb2 import Double_V
    from gz.msgs10.world_control_pb2 import WorldControl
    from gz.msgs10.gui_camera_pb2 import GUICamera
    from gz.msgs10.marker_v_pb2 import Marker_V
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    types={t.DESCRIPTOR.full_name:t for t in (Boolean,Double_V,WorldControl,GUICamera,Marker_V,Pose_V,Pose,EntityFactory)}
    node=Node()
    try:
        while True:
            task=pipe.recv()
            if task is None:break
            service,data,reqname,repname,timeout=task
            try:
                req=types[reqname]();req.ParseFromString(data)
                ok,rep=node.request(service,req,types[reqname],types[repname],timeout)
                pipe.send((ok,rep.SerializeToString(),None))
            except Exception as exc:pipe.send((False,b'',repr(exc)))
    except EOFError:pass
    finally:pipe.close()

class RpcClient:
    def __init__(self):
        context=mp.get_context('spawn');self.pipe,child=context.Pipe()
        self.process=context.Process(target=_serve,args=(child,),daemon=True);self.process.start();child.close()
        self.lock=threading.Lock();self.closed=False
    def request(self,service,request,request_type,response_type,timeout):
        with self.lock:
            self.pipe.send((service,request.SerializeToString(),request_type.DESCRIPTOR.full_name,response_type.DESCRIPTOR.full_name,int(timeout)))
            if not self.pipe.poll(max(5.,timeout*.001+1.)):
                raise RuntimeError('Isolated Gazebo RPC worker did not respond: '+service)
            ok,data,error=self.pipe.recv()
        if error:raise RuntimeError(error)
        result=response_type();result.ParseFromString(data);return ok,result
    def close(self):
        if self.closed:return
        self.closed=True
        try:self.pipe.send(None)
        except (BrokenPipeError,EOFError,OSError):pass
        self.process.join(timeout=1.)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=1.)
        self.pipe.close()
