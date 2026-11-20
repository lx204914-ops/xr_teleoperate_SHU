"""Run only the two existing frozen cases with the accepted deployment code."""
from pathlib import Path
import sys,json,hashlib,traceback,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_quadrotor
ROOT=HERE.parent
OUT=HERE/"stage7_existing_cases";OUT.mkdir(exist_ok=True)
EXPECTED={10:"d957aaa7801217ab47df6b3457929af474cec161ade04ba5d2890593bd89bdad",
50:"00ba192a5a4f06fb62adc3fe745a527e86aac041c8e0949ef773399d5fdc425a"}
records=[]
for n,digest in EXPECTED.items():
    scene=ROOT/"tracked_interface/inputs"/f"smooth_N{n:03d}_S2000.npz"
    assert hashlib.sha256(scene.read_bytes()).hexdigest()==digest
    output=HERE/"results"/f"{scene.stem}_ra_prioritized_execution"
    print(f"CASE_START N={n}",flush=True)
    tick=time.perf_counter()
    try:
        if (output/"summary.json").exists():
            receipt=json.loads((output/"summary.json").read_text("utf8"))
            assert receipt["scene_sha256"]==digest
            record={"obstacles":n,"reused_existing_complete_run":True,"result":receipt,"folder":str(output)}
        elif output.exists():
            raise RuntimeError("Existing incomplete output preserved; will not overwrite")
        else:
            run_quadrotor.run(scene,horizon=24.,method="ra")
            receipt=json.loads((output/"summary.json").read_text("utf8"))
            record={"obstacles":n,"reused_existing_complete_run":False,"result":receipt,"folder":str(output)}
    except Exception as exc:
        record={"obstacles":n,"failure":repr(exc),"traceback":traceback.format_exc(),"folder":str(output)}
    record["wall_seconds"]=time.perf_counter()-tick
    records.append(record)
    (OUT/"run_status.json").write_text(json.dumps({"cases":records,"completed_cases":len(records),"planned_cases":2},indent=2,allow_nan=False),encoding="utf8")
    print(f"CASE_END N={n} "+json.dumps({k:v for k,v in record.items() if k not in ("traceback","result")}),flush=True)
print("STAGE7_RUNS_FINISHED",flush=True)
