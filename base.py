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
        self.base_cond_cov = []
        for i,loc in enumerate(self.ds_loc):
            if len(loc) == 0:
                self.base_cond_cov.append(self.all_cov[np.ix_(self.ss_loc[i],self.ss_loc[i])])
            else:
                self.base_cond_cov.append(self.all_cov[np.ix_(self.ss_loc[i],self.ss_loc[i])] - np.multiply(1/self.ds_eig[i][0],self.A[i])@self.A[i].T)



class Gaussian:
    def __init__(self,kernel,mean=0,sigma_sq=1,delta=1) -> None:
        assert isinstance(kernel,Kernel)
        self.kernel = kernel
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
        self.cond_cov = []
    
    def update_cond_conv(self):
        for i in range(self.kernel.M):
            if len(self.kernel.dependency[i]) == 0:
                #self.cond_cov.append(self.kernel.base_cond_cov[i]+self.delta*np.eye(len(self.kernel.ss_loc[i])))
                s,u = np.linalg.eigh(self.kernel.base_cond_cov[i]+self.delta*np.eye(len(self.kernel.ss_loc[i])))
            else:
                s,u = np.linalg.eigh(self.kernel.base_cond_cov[i]+self.delta*np.eye(len(self.kernel.ss_loc[i]))+self.delta*np.multiply(1/((self.ds_eig[i][0]+self.delta)*self.ds_eig[i][0]),self.A[i])@self.A[i].T)
            self.cond_conv.append((s,u))

    def ll_sep(self,Y):
        """
        Y: observation of dimension N*G
        """
        N,G = Y.shape
        ll = np.log(2 * np.pi)*N + 2*np.log(self.sigma_sq)*N
        result = np.array([-0.5*ll]*G)
        dev = Y - self.all_mean    #N*G
        self.update_cond_conv()
        for i in range(self.M):
            det = np.prod(self.cond_cov[i][0])
            result += np.log(det)

            
            if len(self.dependency[i]) == 0:
            
                cond_dev = dev[self.ss_loc[i]]
            else:
                cond_dev = self.get_cond_dev(dev,i)
            for g in range(G):
                result[g] += -0.5* (cond_dev[:,g].T @ inv @ cond_dev[:,g])
        return np.array(result)
    
    def _init_cond_cov(self):
        eps = 1e-5

        #start = time.time()
        for i in range(self.kernel.M):
            in_cov = self.kernel.all_cov[np.ix_(self.kernel.ss_loc[i],self.kernel.ss_loc[i])]    #m,m
            if len(self.dependency[i]) == 0:
                cond_cov = in_cov
                self.base_cond_cov.append([])
                self.A.append([])
            else:
                
                self.A.append(self.all_cov[np.ix_(self.ss_loc[i],self.ds_loc[i])] @ self.kernel.ds_eig[i][1])
                cond_cov = in_cov - np.multiply(self.A[i],self.kernel.ds_eig[i][0]) @ self.A[i].T              #Sigma_(m)(m)^\prime
            cond_cov_inv = np.linalg.inv(cond_cov)
            L = np.linalg.cholesky(cond_cov_inv)
                # print(np.average((cond_cov_1-cond_cov)/cond_cov))
                # np.savetxt("cond_cov_1_"+str(i),np.ones_like(in_cov)*self.ds_inv_sums[i]*np.average(self.ss_cov[i,self.dependency[i]])**2)
                # np.savetxt("cond_cov_"+str(i),self.all_cov[np.ix_(self.ss_loc[i],self.ds_loc[i])] @ self.ds_inv[i] @ self.all_cov[np.ix_(self.ds_loc[i],self.ss_loc[i])])
                # np.savetxt("in_cov_"+str(i),in_cov)
            #treat cond_cov as positive definite 
            self.cond_inv.append(np.linalg.inv(cond_cov))
            #print(np.linalg.det(cond_cov))
            #self.log_pdet.append(np.log(np.linalg.det(cond_cov)))
            s, u = scipy.linalg.eigh(cond_cov)
            #print(len(s[s<=0]))
            d = s[s > eps]
            s_pinv = _pinv_1d(s,eps)
            # print(len(s_pinv[s_pinv<eps]),len(s_pinv[s_pinv<-eps]))
            # print(s_pinv[s_pinv<eps])
            # print("det:",np.sum(np.log(d)))
            # self.cond_U.append(np.multiply(u, np.sqrt(s_pinv)))
            # #print(np.isnan(self.cond_U).any())
            
            self.log_pdet.append(np.sum(np.log(d)))
        #print("%.3f seconds" %(time.time()-start))
        
        