"""Terminate virtual SITL at first arrival; restore parameter files after exit."""
from pathlib import Path
import hashlib,json,os,signal
from platform_session import Platform as OriginalPlatform,ROOT,PX4

class Platform(OriginalPlatform):
    def snapshot_parameters(self):
        self.saved_parameters={}
        root=PX4/'build/px4_sitl_default/rootfs'
        for name in ('parameters.bson','parameters_backup.bson'):
            p=root/name;self.saved_parameters[p]=p.read_bytes() if p.exists() else None
    def restore_parameters(self):
        receipt=[]
        for p,data in self.saved_parameters.items():
            assert p.parent.resolve()==(PX4/'build/px4_sitl_default/rootfs').resolve()
            if data is None:p.unlink(missing_ok=True)
            else:p.write_bytes(data)
            assert (not p.exists()) if data is None else p.read_bytes()==data
            receipt.append(dict(path=str(p),original_sha256=hashlib.sha256(data).hexdigest() if data else None,restored=True))
        if self.out.exists():
            (self.out/'parameter_file_restoration.json').write_text(json.dumps(dict(restored=True,mode='Restore exact pre-run parameter file bytes after virtual PX4 exits; no landing required',files=receipt),indent=2))
            f=self.out/'flight_tracking_profile.json'
            if f.exists():
                record=json.loads(f.read_text());record.update(restored=True,restore_mode='Exact original parameter files after SITL termination',restore_errors=[]);f.write_text(json.dumps(record,indent=2))
    def __enter__(self):
        self.snapshot_parameters()
        try:return super().__enter__()
        except BaseException:
            if self.proc is not None:self.restore_parameters()
            raise
    def __exit__(self,*exc):
        try:return super().__exit__(*exc)
        finally:self.restore_parameters()
