"""Fixed-budget QP solvers for MPC; host numerical checks, not FPGA timing."""
import ctypes
from pathlib import Path
import subprocess
import numpy as np
from .fpga_pid import FixedFormat

MAX_N = 36
ITERATIONS = 256


def validate_qp(h, g, warm, beta, tolerance):
    """Validate the convex problem before entering either solver backend."""
    h, g, warm = (np.asarray(x, dtype=float) for x in (h, g, warm))
    if g.ndim != 1 or not 1 <= g.size <= MAX_N or h.shape != (g.size,g.size) or warm.shape != g.shape:
        raise ValueError('QP shape')
    if not all(np.all(np.isfinite(x)) for x in (h,g,warm,beta,tolerance)):
        raise ValueError('Nonfinite QP')
    if not np.isscalar(beta) or not np.isscalar(tolerance) or not 0 <= beta < 1 or not 0 < tolerance <= 1:
        raise ValueError('Invalid acceleration or residual tolerance')
    if np.max(np.abs(warm)) > 1 or np.max(np.sum(abs(h),axis=1)) > 1+1e-12:
        raise ValueError('QP scaling or warm-start range')
    if not np.allclose(h,h.T,rtol=0,atol=1e-12) or np.linalg.eigvalsh(h)[0] <= 0:
        raise ValueError('Symmetric positive definite QP required')
    return h,g,warm


def scaled_qp(matrix, target):
    """Least squares -> strictly convex, unit-row-bound QP, same minimizer."""
    matrix=np.asarray(matrix,dtype=float);target=np.asarray(target,dtype=float)
    if matrix.ndim!=2 or target.shape!=(len(matrix),) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError('Finite matrix and target required')
    n=matrix.shape[1]
    if not 1<=n<=MAX_N:raise ValueError('QP exceeds fixed workspace')
    with np.errstate(over='ignore',invalid='ignore'):
        h=matrix.T@matrix;g=-matrix.T@target
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(g)):
        raise ValueError('QP formation overflow')
    eigen=np.linalg.eigvalsh(h)
    if eigen[0]<=0:raise ValueError('Positive definite QP required')
    # Slack protects the integer row bound after rounding at 24/16 or 32/24.
    lipschitz=1.01*np.max(np.sum(abs(h),axis=1))
    beta=(np.sqrt(lipschitz)-np.sqrt(eigen[0]))/(np.sqrt(lipschitz)+np.sqrt(eigen[0]))
    return h/lipschitz,g/lipschitz,float(beta)


def float_solve(h,g,warm,beta,tolerance=2e-5):
    try:h,g,x=validate_qp(h,g,warm,beta,tolerance)
    except (ValueError,TypeError,np.linalg.LinAlgError):
        return 1,np.zeros(np.size(g)),float('inf')
    x=x.copy();y=x.copy()
    for _ in range(ITERATIONS):
        updated=np.clip(y-(h@y+g),-1,1)
        y=updated+beta*(updated-x);x=updated
    residual=float(np.max(abs(x-np.clip(x-(h@x+g),-1,1))))
    return (0,x,residual) if residual<=tolerance else (16,np.zeros_like(x),residual)


def compile_core(output,bits=32,fraction=24):
    source=Path(__file__).resolve().parents[1]/'hardware'/'orbit_mpc_core.cpp'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['clang++','-std=c++11','-O2','-shared','-fPIC','-Wno-unknown-pragmas',
        f'-DQRC_MPC_BITS={bits}',f'-DQRC_MPC_FRAC={fraction}',f'-DQRC_MPC_ITERATIONS={ITERATIONS}',str(source),'-o',str(output)],check=True)
    return output


class FixedQP:
    def __init__(self,library,bits=32,fraction=24):
        self.format=FixedFormat(bits,fraction)
        self.library=ctypes.CDLL(str(Path(library).resolve()))
        try:
            metadata=self.library.orbit_mpc_metadata
        except AttributeError as exc:
            raise ValueError('Unversioned MPC library; rebuild the core') from exc
        metadata.argtypes=[ctypes.c_int];metadata.restype=ctypes.c_int
        if tuple(metadata(i) for i in range(5)) != (1,bits,fraction,MAX_N,ITERATIONS):
            raise ValueError('MPC library ABI, precision or iteration mismatch')
        self.function=self.library.orbit_mpc_solve
        p=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS')
        self.function.argtypes=[p,p,p,ctypes.c_int,ctypes.c_int32,ctypes.c_int32,ctypes.c_int,p,p]
        self.function.restype=ctypes.c_int

    def encode(self,h,g,warm,beta,tolerance):
        h,g,warm=validate_qp(h,g,warm,beta,tolerance)
        n=len(g)
        if not 1<=n<=MAX_N or np.shape(h)!=(n,n) or np.shape(warm)!=(n,):raise ValueError('QP shape')
        H=np.zeros((MAX_N,MAX_N));H[:n,:n]=h
        G=np.zeros(MAX_N);G[:n]=g
        W=np.zeros(MAX_N);W[:n]=warm
        encoded=self.format.encode(H.ravel())
        # A convex float problem can lose convexity when quantized.
        quantized=self.format.decode(encoded.reshape(MAX_N,MAX_N)[:n,:n])
        if np.linalg.eigvalsh(quantized)[0] <= 0:
            raise ValueError('Quantization destroyed positive definiteness')
        return (encoded,self.format.encode(G),self.format.encode(W),
                self.format.encode(beta).item(),max(1,self.format.encode(tolerance).item()))

    def solve(self,h,g,warm,beta,tolerance=2e-5,valid=True):
        n=np.size(g);out=np.zeros(MAX_N,dtype=np.int32);res=np.zeros(1,dtype=np.int32)
        if not valid:return 2,np.zeros(n),0.
        try:H,G,W,b,tol=self.encode(h,g,warm,beta,tolerance)
        except (ValueError,TypeError,np.linalg.LinAlgError):return 1,np.zeros(n),float('inf')
        status=int(self.function(H,G,W,n,b,tol,int(valid),out,res))
        return status,self.format.decode(out[:n]),float(self.format.decode(res[0]))


def integer_oracle(h,g,warm,n,beta,tolerance,fmt):
    """Unlimited Python integers, independent of native signed arithmetic."""
    H=[[int(v) for v in row] for row in np.asarray(h).reshape(MAX_N,MAX_N)]
    g=[int(v) for v in g];x=[int(v) for v in warm];y=x.copy();one=fmt.scale;fault=False
    def bound(a):
        nonlocal fault
        if not fmt.lo<=a<=fmt.hi:fault=True
        return max(fmt.lo,min(fmt.hi,a))
    def round_product(a):return bound((-1 if a<0 else 1)*((abs(a)+(one>>1))>>fmt.fraction))
    def clip(a):return max(-one,min(one,a))
    for _ in range(ITERATIONS):
        new=[clip(y[i]-bound(round_product(sum(H[i][j]*y[j] for j in range(MAX_N)))+g[i])) if i<n else 0 for i in range(MAX_N)]
        y=[bound(new[i]+round_product(beta*(new[i]-x[i]))) for i in range(MAX_N)];x=new
    residual=max(abs(x[i]-clip(x[i]-bound(round_product(sum(H[i][j]*x[j] for j in range(MAX_N)))+g[i]))) for i in range(MAX_N))
    status=4 if fault else 16 if residual>tolerance else 0
    return status,np.array(x if status==0 else [0]*MAX_N,dtype=np.int32),residual
