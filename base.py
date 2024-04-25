import numpy as np
from function import RBF_kernel
import scipy
import time
from sklearn.metrics.pairwise import laplacian_kernel

def _pinv_1d(v, eps=1e-5):
    return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)


class Kernel:

    def __init__(self,spatial,ss_loc,cov=None,dependency=None,d=5,kernel='laplacian',l=0.01):
        """
        number of superspots: M
        cov: full pre-determined covariance matrix
        dependency format: M-length list of lists, the i-th element indicates the superspots that the i-th superspot is dependent on
        ss_loc: M-length list of lists, the i-th element indicates the spots (ordinal) that the i-th superspots contain
        spatial: spatial coordinates of all spots
        l: hyperparameter for kernel
        """
        super().__init__()
        # if mean.ndim == 1:
        #     self.all_mean = np.array(mean)[:,np.newaxis]
        # else:
        #     self.all_mean = np.array(mean)       
        self.all_cov = cov
        self.dependency = dependency   # temporarily assuming dependency[i] > i 
        self.M = len(dependency)
        self.N = len(spatial)
        self.ss_loc = ss_loc
        self.spatial = spatial
        self.cond_mean = None
        self.l = l
        self.d = d
        self.A = []
        self.cond_cov = []
        self._initialize()
    
    def _initialize(self):
        self._init_ds_loc()
        self._init_ds_eig()
        self._init_base_cond_cov()        
        #self._init_cond_cov()

    def _init_ds_loc(self):
        self.ds_loc = []        
        for i,ds in enumerate(self.dependency):
           ind = []
           for ss in ds:
               ind += self.ss_loc[ss]
           self.ds_loc.append(ind)
    
    def _init_ds_eig(self):
        #C_m,C_m
        self.ds_eig = []
        for i,loc in enumerate(self.ds_loc):
            ds_cov = self.all_cov[np.ix_(loc,loc)]
            if len(ds_cov) == 0:
                self.ds_eig.append(())
                self.A.append(())
            else:
                s,u = np.linalg.eigh(ds_cov)
                #s_inv = _pinv_1d(s)
                self.ds_eig.append((s,u))
                self.A.append(self.all_cov[np.ix_(self.ss_loc[i],loc)] @ u)
    
    def _init_base_cond_cov(self):
        for i,loc in enumerate(self.ds_loc):
            if len(loc) == 0:
                self.cond_cov.append(self.all_cov[np.ix_(self.ss_loc[i],self.ss_loc[i])])
            else:
                self.cond_cov.append(self.all_cov[np.ix_(self.ss_loc[i],self.ss_loc[i])] - np.multiply(1/self.ds_eig[i][0],self.A[i])@self.A[i].T)

    def update_cond_cov(self,delta):
        self.cond_cov_eig = []
        for i in range(self.M):
            if len(self.dependency[i]) == 0:
                #self.cond_cov.append(self.kernel.base_cond_cov[i]+self.delta*np.eye(len(self.kernel.ss_loc[i])))
                s,u = np.linalg.eigh(self.cond_cov[i]+delta*np.eye(len(self.ss_loc[i])))
            else:
                s,u = np.linalg.eigh(self.cond_cov[i]+delta*np.eye(len(self.ss_loc[i]))+delta*np.multiply(1/((self.ds_eig[i][0]+delta)*self.ds_eig[i][0]),self.A[i])@self.A[i].T)
            self.cond_cov_eig.append((s,u))



class Gaussian:
    def __init__(self,kernel,mean=0,sigma_sq=1,delta=1) -> None:
        assert isinstance(kernel,Kernel)
        self.kernel = kernel
        # mean should be N*1
        if mean.ndim == 1:
            self.mean = np.array(mean)[:,np.newaxis]
        elif mean.ndim == 0:
            self.mean = np.repeats(mean,kernel.N)[:,np.newaxis]
        else:
            self.mean = np.array(mean)    
        self.sigma_sq = sigma_sq
        self.delta = delta
        self.cond_eig = []
        self.temp = []

    def update_cond_mean(self,Y):
        self.cond_dev = Y - self.mean
        dev = Y - self.mean
        for i in range(self.kernel.M):
            if len(self.kernel.dependency[i]) > 0:
                self.cond_dev[self.kernel.ss_loc[i],:] -= np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta),self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[self.kernel.ds_loc[i],:]


    def ll_sep(self,Y):
        """
        Y: observation of dimension N*G
        """
        N,G = Y.shape
        ll = np.log(2 * np.pi)*N + 2*np.log(self.sigma_sq)*N
        result = np.array([-0.5*ll]*G)
        self.kernel.update_cond_cov(self.delta)
        self.update_cond_mean(Y)
        for i in range(self.M):
            det = np.prod(self.kernel.cond_cov_eig[i][0])
            result += np.log(det)
            temp = self.cond_dev[self.kernel.ss_loc[i],:].T @ self.kernel.A[i]
            result += np.sum(np.multiply(self.kernel.cond_cov_eig[i][0],np.square(temp)),axis=1)/self.sigma_sq
        return result*-0.5
        
        