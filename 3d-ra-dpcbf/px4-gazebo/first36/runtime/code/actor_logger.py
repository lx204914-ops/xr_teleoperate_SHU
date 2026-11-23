"""Independent read-only Gazebo obstacle-entity recorder."""
from pathlib import Path
import sys,os,time,signal
import numpy as np
sys.path.append('/usr/lib/python3/dist-packages')
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
out=Path(sys.argv[1]);f=(out/'actual_obstacle_entities.bin').open('wb',buffering=1024*1024)
node=Node();running=True
def finish(*args):
    global running
    running=False
def receive(msg):
    t=msg.header.stamp.sec+msg.header.stamp.nsec*1e-9
    rows=[]
    for p in msg.pose:
        if p.name.startswith('ra_obstacle_'):rows.append([t,int(p.name.rsplit('_',1)[1]),p.position.x,p.position.y,p.position.z])
    if rows:f.write(np.asarray(rows,dtype='<f8').tobytes())
node.subscribe(Pose_V,'/ra/obstacle_actual',receive)
signal.signal(signal.SIGTERM,finish);signal.signal(signal.SIGINT,finish)
(out/'actor_logger_ready').write_text(str(os.getpid()))
while running:time.sleep(.02)
node.unsubscribe('/ra/obstacle_actual');f.flush();f.close()
