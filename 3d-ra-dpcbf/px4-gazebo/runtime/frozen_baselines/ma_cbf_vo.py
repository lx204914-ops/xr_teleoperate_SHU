"""MA-CBF-VO: soft VO objective + hard braking-distance barrier, 3D adaptation.
Roncero et al., arXiv:2409.10117, Eqs.16--19 and 24, Table I.
Only current states enter control. No four-plane or branch approximation.
"""
import numpy as np
from numba import njit
from baseline_coefficients import coefficients
from sampled_baselines import tube_jets,constant,jadd,jsquare,jmul,jinv,jsqrt,add,sub,mul,scale,square
KV=10.;KC=10.;ABR=1.2;KVO=1000.

@njit(cache=True)
def terms(x,obs,par):
    n=len(obs);R=par[21]+par[22]+obs[:,9];z=np.zeros(n);pv=par.copy();pv[3]=KV
    bv,cv,lv=coefficients(x,obs,R,z,R,pv,2)
    bh=np.zeros((n,3));ch=np.zeros(n);lh=lv.copy();weights=np.zeros(n);ttc=np.full(n,np.inf)
    for i in range(n):
        r=obs[i,:3]-x[:3];w=obs[i,3:6]-x[3:];ao=obs[i,6:9]
        D=np.linalg.norm(r);e=r/D;q=e@w;Z=w@w-q*q;c=min(q,0.)
        hc=D-R[i]-c*c/(2*ABR)
        bh[i]=(c/ABR)*e
        ch[i]=q-(c/ABR)*(Z/D+e@ao)+KC*hc
        lh[i,7]=hc;lh[i,8]=ch[i];lh[i,9:12]=bh[i]
        ww=w@w;rw=r@w;dd=D*D-R[i]*R[i];disc=rw*rw-ww*dd
        if dd<=0.:ttc[i]=0.;weights[i]=1e12
        elif ww>1e-16 and rw<0. and disc>=0.:
            # Stable positive entry root; no nominal-input or future-state TTC.
            entry=dd/(-rw+np.sqrt(disc))
            if entry>0.:ttc[i]=entry;weights[i]=1./entry
    return bv,cv,lv,bh,ch,lh,weights,ttc

@njit(cache=True)
def hard_hold(x,obs,par,dt):
    n=len(obs);m=np.full(n,np.inf);details=np.full((n,4),np.nan)
    for i in range(n):
        r=obs[i,:3]-x[:3];w=obs[i,3:6]-x[3:];ao=obs[i,6:9]
        if np.linalg.norm(ao)>.8+1e-10:continue
        rj,wj=tube_jets(r,w,ao,dt);d2=constant(0.);rw=constant(0.)
        for l in range(3):d2=jadd(d2,jsquare(rj[l]));rw=jadd(rw,jmul(rj[l],wj[l]))
        if d2[0,0]<=0.:continue
        q=jmul(rw,jinv(jsqrt(d2)))
        c=np.minimum(q[0],0.)
        hd=sub(q[0],scale(mul(c,q[1]),1./ABR))
        # h_c is C1. The q<0 second derivative applies almost everywhere;
        # across q=0 the indicator lies in [0,1]. This encloses both sides.
        loss=0. if q[0,0]>=0. else square(q[1])[1]
        hdd_lo=np.nextafter(q[1,0]-(loss+mul(c,q[2])[1])/ABR,-np.inf)
        psi_lo=add(np.array([hdd_lo,hdd_lo]),scale(hd,KC))[0]
        m[i]=np.nextafter(dt*max(0.,-psi_lo)+2e-7,np.inf)
        details[i]=np.array([psi_lo,q[0,0],q[0,1],np.sqrt(d2[0,0])])
    return m,details

class SoftVOQP:
    """Reusable convex QP with an exact O(N) CSC sparsity pattern.

    The earlier comparison prototype initialized OSQP with a dense all-ones
    constraint matrix.  Although the mathematical QP had O(N) nonzeros, that
    allocation stored O(N^2) entries and invalidated scaling measurements.
    Here every stored entry belongs to one of the hard rows, soft rows, or
    variable-bound rows, so nnz(A)=8N+3 exactly.
    """
    def __init__(self,n,smooth):
        import scipy.sparse as sp
        import osqp
        self.n=n;self.smooth=smooth;self.sp=sp;self.solver=osqp.OSQP();self.fallbacks=0
        nv=3+n;nr=2*n+nv
        rows=[];cols=[];values=[]
        for j in range(3):
            rows.extend(range(0,n));cols.extend([j]*n);values.extend([1.]*n)
            rows.extend(range(n,2*n));cols.extend([j]*n);values.extend([1.]*n)
            rows.append(2*n+j);cols.append(j);values.append(1.)
        for i in range(n):
            rows.extend([n+i,2*n+3+i]);cols.extend([3+i,3+i]);values.extend([1.,1.])
        pattern=sp.csc_matrix((np.asarray(values),(np.asarray(rows),np.asarray(cols))),shape=(nr,nv))
        self.pattern_rows=np.asarray(rows,dtype=np.int64)
        self.pattern_cols=np.asarray(cols,dtype=np.int64)
        self.nnz=int(pattern.nnz)
        if self.nnz!=8*n+3:raise RuntimeError('Unexpected MA-CBF-VO QP sparsity pattern')
        self.solver.setup(P=sp.diags(np.ones(nv),format='csc'),q=np.zeros(nv),
            A=pattern,l=np.full(nr,-np.inf),u=np.full(nr,np.inf),
            verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,polishing=False,adaptive_rho=True)
    def solve(self,nom,prev,bv,cv,bh,ch,m,w,force_fallback=False):
        n=self.n;nv=3+n;nr=2*n+nv
        values=[]
        for j in range(3):
            values.extend(bh[:,j]);values.extend(bv[:,j]);values.append(1.)
        for i in range(n):values.extend([1.,1.])
        A=self.sp.csc_matrix((np.asarray(values),(self.pattern_rows,self.pattern_cols)),shape=(nr,nv))
        if A.nnz!=self.nnz:raise RuntimeError('MA-CBF-VO QP sparsity changed at runtime')
        lo=np.r_[m-ch,-cv,[-2.]*3,[0.]*n];hi=np.r_[[np.inf]*(2*n),[2.]*3,[np.inf]*n]
        diag=np.r_[np.full(3,2*(1+self.smooth)),2*KVO*w]
        linear=np.r_[-2*(nom+self.smooth*prev),np.zeros(n)]
        self.solver.update(Px=diag,q=linear,Ax=A.data,l=lo,u=hi)
        result=self.solver.solve(raise_error=False)
        good=(not force_fallback) and result.info.status_val in (1,2) and result.x is not None
        if good:
            sol=result.x;good=np.min(bh@sol[:3]+ch-m)>=-2e-7 and np.max(abs(sol[:3]))<=2+1e-7 and np.min(bv@sol[:3]+cv+sol[3:])>=-2e-7
        if not good:
            sol=clarabel_qp(nom,prev,bv,cv,bh,ch,m,w,self.smooth)
            self.fallbacks+=1
            if sol is None:return np.zeros(3),False,int(result.info.iter),np.full(n,np.nan)
            good=np.min(bh@sol[:3]+ch-m)>=-2e-7 and np.max(abs(sol[:3]))<=2+1e-7
        # Canonical minimum feasible slack; zero-weight slacks otherwise nonunique.
        delta=np.maximum(0.,-(bv@sol[:3]+cv))
        return sol[:3],bool(good),int(result.info.iter),delta

@njit(cache=True)
def monitor_ma(x,u,tape,td,t,dt,fine,par):
    # Imports are module-level below so Numba can resolve state_at.
    minimum=np.full(3,1e20);length=0.
    for k in range(round(dt/fine)):
        before=x[:3].copy();x[:3]+=fine*x[3:]+.5*fine*fine*u;x[3:]+=fine*u
        length+=np.linalg.norm(x[:3]-before)
        obs=state_at(tape,t+(k+1)*fine,td,1)
        bv,cv,lv,bh,ch,lh,weight,ttc=terms(x,obs,par)
        for i in range(len(obs)):
            minimum[0]=min(minimum[0],lh[i,0]-par[21]-obs[i,9])
            minimum[1]=min(minimum[1],lh[i,7]);minimum[2]=min(minimum[2],bh[i]@u+ch[i])
    return x,minimum,length
from scene_tapes import state_at

def clarabel_qp(nom,prev,bv,cv,bh,ch,m,w,smooth):
    import scipy.sparse as sp
    import clarabel
    n=len(w);nv=3+n
    hard=np.hstack([-bh,np.zeros((n,n))])
    soft=np.hstack([-bv,-np.eye(n)])
    bounds=np.vstack([np.eye(nv),-np.eye(nv)])
    # Finite inequalities only: u<=2,-u<=2,delta>=0.
    boundA=np.vstack([bounds[:3],bounds[nv:nv+3],bounds[nv+3:]])
    A=sp.csc_matrix(np.vstack([hard,soft,boundA]));b=np.r_[ch-m,cv,[2.]*6,[0.]*n]
    P=sp.diags(np.r_[np.full(3,2*(1+smooth)),2*KVO*w],format='csc')
    linear=np.r_[-2*(nom+smooth*prev),np.zeros(n)]
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-11;settings.max_iter=400
    answer=clarabel.DefaultSolver(P,linear,A,b,[clarabel.NonnegativeConeT(len(b))],settings).solve()
    return np.array(answer.x) if str(answer.status) in ('Solved','AlmostSolved') else None
