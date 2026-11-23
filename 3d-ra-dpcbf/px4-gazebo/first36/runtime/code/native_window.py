"""Locate and capture only this project's Gazebo window on the existing desktop."""
from pathlib import Path
import time,json,os
from Xlib import X,display
def descendants(pid):
    result={pid}
    for _ in range(5):
        for p in Path('/proc').glob('[0-9]*'):
            try:
                if int((p/'stat').read_text().split(') ')[1].split()[1]) in result:result.add(int(p.name))
            except (OSError,ValueError,IndexError):pass
    return result
def configure(pid,name):
    d=display.Display(name);owned=descendants(pid);atom=d.intern_atom('_NET_WM_PID')
    def walk(w,depth=0):
        if depth>4:return None
        try:
            prop=w.get_full_property(atom,X.AnyPropertyType)
            if prop is not None and int(prop.value[0]) in owned and 'Gazebo' in (w.get_wm_name() or '') and w.get_attributes().map_state==X.IsViewable:return w
            for c in w.query_tree().children:
                r=walk(c,depth+1)
                if r is not None:return r
        except Exception:pass
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        w=walk(d.screen().root)
        if w is not None:
            w.configure(width=1600,height=900,stack_mode=X.Above);d.sync()
            time.sleep(.2)
            g=w.get_geometry();p=d.screen().root.translate_coords(w,0,0)
            if w.get_attributes().map_state==X.IsViewable and p.x>=0 and p.y>=0 and p.x+g.width<=d.screen().width_in_pixels and p.y+g.height<=d.screen().height_in_pixels:
                result=w.id;d.close();return result
        time.sleep(.25)
    d.close();raise RuntimeError('No window belonging to this Gazebo client; refusing desktop capture')
