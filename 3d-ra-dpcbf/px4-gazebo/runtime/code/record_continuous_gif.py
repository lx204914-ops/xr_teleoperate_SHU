"""Record actual X11 Gazebo pixels during a fresh PX4 closed-loop flight."""
from pathlib import Path
import os,subprocess,json,time,struct,hashlib
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
PY=os.environ.get('CBF_PYTHON',str(Path.home()/'cbf_px4/.venv/bin/python'))

def screenshot():
    gui=json.loads((ROOT/'native_gui.json').read_text())
    data=subprocess.check_output(['xwd','-id',str(gui['window_id']),'-silent','-display',gui['display']],env={**os.environ,'XAUTHORITY':'/run/user/1000/gdm/Xauthority'},timeout=8)
    h=struct.unpack('>25I',data[:100]);size,version,fmt,depth,w,height,xoff,order,unit,bitorder,pad,bpp,stride,visual,rm,gm,bm,bits,cmap,ncolors,*_=h
    assert version==7 and bpp in (16,24,32),(version,bpp,h)
    buf=np.frombuffer(data[size+ncolors*12:],dtype='u1').reshape(height,stride)
    if bpp in (16,32):pixels=buf.view(('<' if order==0 else '>')+('u2' if bpp==16 else 'u4'))[:,:w]
    else:
        b=buf[:,:w*3].reshape(height,w,3).astype('uint32')
        pixels=(b[:,:,0]|(b[:,:,1]<<8)|(b[:,:,2]<<16)) if order==0 else ((b[:,:,0]<<16)|(b[:,:,1]<<8)|b[:,:,2])
    channels=[]
    for mask in (rm,gm,bm):
        shift=(mask&-mask).bit_length()-1;maximum=mask>>shift
        channels.append(np.round(((pixels&mask)>>shift).astype('float32')*255/maximum).astype('u1'))
    rgb=np.stack(channels,axis=2)
    return Image.fromarray(rgb).resize((1280,720),Image.Resampling.LANCZOS)

def main():
    os.environ['GZ_IP']='127.0.0.1'
    os.environ.setdefault('GZ_PARTITION','cbf_dense_tracking')
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--attach-session',required=True);args=ap.parse_args()
    name=args.attach_session;run=ROOT/'runs'/name
    folder=ROOT/'native_recording'/name;folder.mkdir(parents=True,exist_ok=True)
    (folder/'recording_status.json').write_text(json.dumps(dict(session=name,phase='waiting_for_real_GUI',pid=os.getpid())))
    from native_audit_reader import records as read_native_records
    native_offset=0;native_latest=None
    frames=[];records=[];begin=time.monotonic();last=-1.;last_new=begin
    while time.monotonic()-begin<180:
        path=run/'native_view_status.json'
        view=json.loads(path.read_text()) if path.exists() else {}
        batch,native_offset=read_native_records(run/'native_frames.bin',native_offset)
        if batch:native_latest=batch[-1]
        before_native=native_latest.copy() if native_latest is not None else None
        if before_native is not None and before_native[11]!=0:break
        t=float(before_native[1]) if before_native is not None else -1.
        gui_path=ROOT/'native_gui.json'
        try: gui=json.loads(gui_path.read_text()) if gui_path.exists() else {}
        except (OSError,ValueError): gui={}
        if t>last and view.get('marker_batches',0)>2 and gui.get('session')==name and gui.get('window_id'):
            if not frames and t>3.:raise RuntimeError('Recording missed initial motion; do not call it a full live recording')
            tic=time.monotonic()
            try: frame=screenshot()
            except subprocess.CalledProcessError as exc:
                if not frames and t<3.:
                    time.sleep(.1);continue
                raise
            after=json.loads(path.read_text())
            batch,native_offset=read_native_records(run/'native_frames.bin',native_offset)
            if batch:native_latest=batch[-1]
            after_native=native_latest.copy() if native_latest is not None else before_native
            if after_native[11]!=0:break
            n=int(before_native[6]);native_params=before_native[12:].reshape(n,33)
            frames.append(frame);records.append(dict(sim_before_s=t,sim_after_s=float(after_native[1]),wall_s=tic-begin,native_frame_before=int(before_native[2]),native_frame_after=int(after_native[2]),controller_stamp_before=float(native_params[0,9]),
                rendered_batches=view['marker_batches'],native_checks=view.get('native_checks',{}),curvature_vertex=np.c_[native_params[:,5],native_params[:,4]].tolist(),risk_radius=native_params[:,6].tolist(),
                raw_RGB_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
            last=t;last_new=time.monotonic()
            if len(frames)>=600:raise RuntimeError('Bounded recorder frame limit reached')
            (folder/'recording_status.json').write_text(json.dumps(dict(session=name,phase='capturing_actual_Gazebo',frames=len(frames),sim_s=t)))
        state=json.loads((ROOT/'active_session.json').read_text()) if (ROOT/'active_session.json').exists() else {}
        if frames and last>14 and time.monotonic()-last_new>3:break
        if state.get('session')==name and state.get('completed'):
            if not frames:raise RuntimeError('No live native frames captured')
            break
        time.sleep(.2)
    assert len(frames)>1 and records[0]['sim_before_s']<3
    assert len({r['raw_RGB_sha256'] for r in records})>1
    # A shared palette prevents per-frame color flicker. Nothing is redrawn.
    ids=np.linspace(0,len(frames)-1,min(12,len(frames)),dtype=int)
    atlas=Image.new('RGB',(320*4,180*3),(230,230,230))
    for j,k in enumerate(ids):atlas.paste(frames[k].resize((320,180)),((j%4)*320,(j//4)*180))
    palette=atlas.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    quantized=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
    # Preserve the simulator's measured time scale, removing wall-clock stalls.
    # Quantization is cumulative in GIF 10ms units, so playback time stays honest.
    times=np.array([r['sim_before_s'] for r in records]);edges=np.r_[times-times[0],times[-1]-times[0]+.5]
    edges=np.round(edges*100).astype(int);duration=np.maximum(1,np.diff(edges))*10
    target=folder/'gazebo_native_live.gif'
    quantized[0].save(target,save_all=True,append_images=quantized[1:],duration=duration.tolist(),loop=0,optimize=False,disposal=2)
    frames[len(frames)//2].save(folder/'native_midframe.png')
    meta=dict(session=name,source='Direct XWD pixels of native Gazebo window during a NEW official PX4+Gazebo closed loop; not Matplotlib, not a trajectory replay',
        playback='GIF durations follow displayed simulation time; wall-clock stalls are removed, without changing physical trajectories or speeds',
        native_size=[1600,900],gif_size=[1280,720],frames=len(frames),sim_start_s=float(times[0]),sim_end_s=float(times[-1]),
        playback_duration_s=float(duration.sum()/1000),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,records=records)
    (folder/'recording_receipt.json').write_text(json.dumps(meta,indent=2))
    deadline=time.monotonic()+100
    while time.monotonic()<deadline:
        state=json.loads((ROOT/'active_session.json').read_text()) if (ROOT/'active_session.json').exists() else {}
        if state.get('session')==name and state.get('completed'):break
        time.sleep(.5)
    meta['physical_check']=state.get('audit');meta['flight_summary']=state.get('summary');meta['continuous_check']=state.get('continuous_audit');meta['display_check']=state.get('display_audit')
    assert not meta['flight_summary']['synchronous_physics']
    (folder/'recording_receipt.json').write_text(json.dumps(meta,indent=2))
    (folder/'recording_status.json').write_text(json.dumps(dict(session=name,phase='completed_real_flight_and_GIF',frames=len(frames),gif_bytes=target.stat().st_size,physical_check_passed=bool(meta['physical_check'] and meta['physical_check']['passed']))))
    print(json.dumps({k:v for k,v in meta.items() if k not in ('records','physical_check','flight_summary')},indent=2))

if __name__=='__main__':main()
