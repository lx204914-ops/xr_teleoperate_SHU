"""Read complete native C++ render records, never synthesize a frame."""
import struct
from pathlib import Path
import numpy as np

def records(path,offset=0):
    path=Path(path);out=[]
    if not path.exists():return out,offset
    with path.open('rb') as f:
        f.seek(offset)
        while True:
            start=f.tell();head=f.read(4)
            if len(head)!=4:break
            size=struct.unpack('<I',head)[0]
            if not 12<=size<=6612:raise ValueError('Invalid native audit record length')
            data=f.read(size*8)
            if len(data)!=size*8:break
            out.append(np.frombuffer(data,dtype='<f8').copy());offset=f.tell()
    return out,offset
