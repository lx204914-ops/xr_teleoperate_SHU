"""Actual PX4 receiver confirmations, independent from corrected actuator references."""
import numpy as np

def supplement(sent, matched, first, rx_t, payload, trace, summary):
    rx_t=np.asarray(rx_t);payload=np.asarray(payload)
    initial_missing=[i for i in range(len(sent)) if i not in first]
    confirmed=[];used=set();end=float(summary['mission_end_sim'])
    echoes=[]
    for k,v in enumerate(trace.get('receive',[])):
        if v.get('kind')!='DEBUG_VECT' or v.get('system')!=1 or v.get('component')!=1:continue
        if v.get('accepted_wire_sequence') is None or v.get('stamp') is None:continue
        a=np.asarray(v.get('acceleration_enu',[]),dtype=np.float32)
        if a.shape!=(3,) or not np.isfinite(a).all():continue
        t=float(v['stamp'])
        if t>end+1e-9:continue
        echoes.append((t,k,int(v['accepted_wire_sequence']),a))
    echoes.sort(key=lambda x:x[0])
    for i in initial_missing:
        if sent[i].get('wire_sequence') is None:continue
        a=np.asarray(sent[i]['acceleration_enu'],dtype=np.float32)
        low=float(sent[i]['state_stamp'])-.004;high=min(float(sent[i]['state_stamp'])+.08,end)
        before=[j for j in first if j<i];after=[j for j in first if j>i]
        if before:low=max(low,float(rx_t[first[max(before)][0]])+1e-10)
        if after:high=min(high,float(rx_t[first[min(after)][0]])-1e-10)
        candidates=[(t,k,b) for t,k,seq,b in echoes if k not in used and low<=t<=high and seq==int(sent[i]['wire_sequence']) and np.array_equal(a,b)]
        if len(candidates)!=1:continue
        t,k,b=candidates[0];row=np.r_[np.zeros(3),b.astype(float)];j=len(rx_t)
        error=float(np.max(abs(np.asarray(sent[i]['velocity_enu']+sent[i]['acceleration_enu'])-row)))
        rx_t=np.r_[rx_t,t];payload=np.vstack([payload,row]);matched.append((i,j,error));first[i]=(j,error);used.add(k)
        confirmed.append(dict(controller_index=i,confirmation_stamp=t,receive_trace_index=k,wire_sequence=int(sent[i]['wire_sequence']),source='Actual PX4 MavlinkReceiver publication confirmation',maximum_payload_error=error))
    return matched,first,rx_t,payload,dict(raw_ULog_missing=initial_missing,receiver_confirmed=confirmed,remaining=[i for i in range(len(sent)) if i not in first],scope='Receiver accepted input only; no actuator or tracking implication')
