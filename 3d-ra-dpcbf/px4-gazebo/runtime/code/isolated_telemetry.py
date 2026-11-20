from pathlib import Path
import os,subprocess,time,numpy as np
from continuous_instrumentation import TimedGazebo
ROOT=Path(__file__).resolve().parents[1]
class IsolatedGazebo(TimedGazebo):
    def __init__(self):
        super().__init__()
        self.node.unsubscribe('/world/ra_lab/pose/info');self.node.unsubscribe('/ra/obstacle_actual')
        self.capture_folder=Path(os.environ['CBF_REAL_RUN_FOLDER'])
        self.actor_log=(self.capture_folder/'actor_logger.log').open('wb')
        self.actor_process=subprocess.Popen([os.environ.get('CBF_PYTHON',str(Path.home()/'cbf_px4/.venv/bin/python')),str(ROOT/'code/actor_logger.py'),str(self.capture_folder)],stdout=self.actor_log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+8.
        while not (self.capture_folder/'actor_logger_ready').exists() and time.monotonic()<deadline and self.actor_process.poll() is None:time.sleep(.02)
        if not (self.capture_folder/'actor_logger_ready').exists():raise RuntimeError('Read-only obstacle entity logger did not start')
    def finish_actor_logging(self):
        if self.actor_process.poll() is None:self.actor_process.terminate()
        self.actor_process.wait(timeout=8);self.actor_log.close()
        rows=np.fromfile(self.capture_folder/'actual_obstacle_entities.bin',dtype='<f8')
        if len(rows)%5:raise RuntimeError('Incomplete physical obstacle entity record')
        self.obstacle_pose_samples=rows.reshape(-1,5)
