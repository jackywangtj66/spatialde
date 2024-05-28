import numpy as np
from function import RBF_kernel
import scipy
import time
from sklearn.metrics.pairwise import laplacian_kernel,rbf_kernel
from scipy.sparse import csr_matrix
from operator import itemgetter

def _pinv_1d(v, eps=1e-5):
    return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)


class Kernel:

    def __init__(self,spatial,ss_loc,cov=None,dependency=None,d=5,kernel='laplacian',l=0.01,k=1):
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
        self.dependency = dependency   # temporarily assuming dependency[i] < i 
        self.M = len(dependency)
        self.N = len(spatial)
        self.all_cov = [[0 for _ in range(self.M)] for _ in range(self.M)]
        self.ss_loc = ss_loc
        self.spatial = spatial
        self.cond_mean = None
        self.l = l
        self.d = d
        self.A = []
        self.k = k
        self.cond_cov = []
        self._initialize(cov)
    
    def get_mat(self,rows,cols):
        row = []
        for r in rows:
            col = []
            for c in cols:
                if r >= c:
                    col.append(self.all_cov[r][c])
                else:
                    col.append(self.all_cov[c][r].T)
            row.append(np.hstack(col))
        return np.vstack(row)

    
    def _initialize(self,cov):
        self._init_ds_loc()
        self._init_all_cov(cov)
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
    
    def _init_all_cov(self,cov):
        for i in range(self.M):
            if cov is not None:
                self.all_cov[i][i] = cov[np.ix_(self.ss_loc[i],self.ss_loc[i])]
            else:
                self.all_cov[i][i] = rbf_kernel(self.spatial[self.ss_loc[i]])
            for j in self.dependency[i]:
                if cov is not None:
                    self.all_cov[i][j] = cov[np.ix_(self.ss_loc[i],self.ss_loc[j])]
                else:
                    self.all_cov[i][j] = rbf_kernel(self.spatial[self.ss_loc[i]],self.spatial[self.ss_loc[j]])
        for k in range(self.M):
            for i in self.dependency[k]:
                for j in self.dependency[k]:
                    if j<i and isinstance(self.all_cov[i][j],int):
                        if cov is not None:
                            self.all_cov[i][j] = cov[np.ix_(self.ss_loc[i],self.ss_loc[j])]
                        else:
                            self.all_cov[i][j] = rbf_kernel(self.spatial[self.ss_loc[i]],self.spatial[self.ss_loc[j]])
    
    # def _init_ds_eig(self):
    #     #C_m,C_m
    #     self.ds_eig = []
    #     for i,loc in enumerate(self.ds_loc):
    #         ds_cov = self.all_cov[np.ix_(loc,loc)]
    #         if len(ds_cov) == 0:
    #             self.ds_eig.append(())
    #             self.A.append(())
    #         else:
    #             s,u = np.linalg.eigh(ds_cov)
    #             #s_inv = _pinv_1d(s)
    #             self.ds_eig.append((s,u))
    #             self.A.append(self.all_cov[np.ix_(self.ss_loc[i],loc)] @ u)

    def _init_ds_eig(self):
        #C_m,C_m
        self.ds_eig = []
        for i in range(self.M):
            if len(self.dependency[i]) == 0:
                self.ds_eig.append(())
                self.A.append(())            
            else:            
                ds_cov = self.get_mat(self.dependency[i],self.dependency[i])
                s,u = np.linalg.eigh(ds_cov)
                #s_inv = _pinv_1d(s)
                self.ds_eig.append((s,u))
                self.A.append(self.get_mat([i],self.dependency[i]) @ u)
    
    # def _init_base_cond_cov(self):
    #     for i,loc in enumerate(self.ds_loc):
    #         if len(loc) == 0:
    #             self.cond_cov.append(self.all_cov[np.ix_(self.ss_loc[i],self.ss_loc[i])])
    #         else:
    #             self.cond_cov.append(self.all_cov[np.ix_(self.ss_loc[i],self.ss_loc[i])] - np.multiply(1/self.ds_eig[i][0],self.A[i])@self.A[i].T)

    def _init_base_cond_cov(self):
        for i in range(self.M):
            if len(self.dependency[i]) == 0:
                self.cond_cov.append(self.get_mat([i],[i]))
            else:
                self.cond_cov.append(self.get_mat([i],[i]) - np.multiply(1/self.ds_eig[i][0],self.A[i])@self.A[i].T)
    
    # def _calc_B(self):
    #     self.B = []
    #     for i in range(self.M):
    #         if len(self.dependency[i]) == 0:
    #             self.B.append(())            
    #         else:            
    #             ds_cov = self.get_mat(self.dependency[i],self.dependency[i])
    #             s,u = np.linalg.eigh(ds_cov)
    #             #s_inv = _pinv_1d(s)
    #             self.ds_eig.append((s,u))
    #             self.A.append(self.get_mat([i],self.dependency[i]) @ u)
    #     for i,loc in enumerate(self.ds_loc):
    #         ds_cov = self.all_cov[np.ix_(loc,loc)]
    #         if len(ds_cov) == 0:
    #             self.B.append(())
    #         else:
    #             self.B.append(self.all_cov[np.ix_(self.ss_loc[i],loc)] @ np.linalg.inv(ds_cov))

    def update_cond_cov(self,Delta):
        self.cond_cov_eig = [[] for _ in range(self.k)]
        for eig, delta in zip(self.cond_cov_eig,Delta):
            for i in range(self.M):
                if len(self.dependency[i]) == 0:
                #self.cond_cov.append(self.kernel.base_cond_cov[i]+self.delta*np.eye(len(self.kernel.ss_loc[i])))
                    s,u = np.linalg.eigh(self.cond_cov[i]+delta*np.eye(len(self.ss_loc[i])))
                else:
                    s,u = np.linalg.eigh(self.cond_cov[i]+delta*np.eye(len(self.ss_loc[i]))+delta*np.multiply(1/((self.ds_eig[i][0]+delta)*self.ds_eig[i][0]),self.A[i])@self.A[i].T)
                eig.append((s,u))



class Gaussian:
    def __init__(self,kernel,mean=0,sigma_sq=1,delta=0) -> None:
        assert isinstance(kernel,Kernel)        
        mean = np.array(mean)
        self.kernel = kernel
        # mean should be N*1
        if mean.ndim == 1:
            self.mean = np.array(mean)[:,np.newaxis]
        if mean.ndim == 0:
            self.mean = np.repeat(mean,kernel.N)[:,np.newaxis]
        self.sigma_sq = sigma_sq
        self.delta = delta
        self.temp = []

    def update_cond_mean(self,Y):
        self.cond_dev = Y - self.mean
        dev = Y - self.mean
        for i in range(self.kernel.M):
            if len(self.kernel.dependency[i]) > 0:
                # print(self.delta)
                # print(np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta),self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[self.kernel.ds_loc[i],:])
                self.cond_dev[self.kernel.ss_loc[i],:] = self.cond_dev[self.kernel.ss_loc[i],:] - np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta),self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[self.kernel.ds_loc[i],:]


    def ll_sep(self,Y):
        """
        Y: observation of dimension N*G
        """
        N,G = Y.shape
        ll = np.log(2 * np.pi)*N + 2*np.log(self.sigma_sq)*N
        result = np.array([ll]*G)
        self.kernel.update_cond_cov(self.delta)
        self.update_cond_mean(Y)
        for i in range(self.kernel.M):
            det = np.prod(self.kernel.cond_cov_eig[i][0])
            result += np.log(det)
            # print(self.kernel.ss_loc[i])
            # print(self.kernel.A[i])
            temp = self.cond_dev[self.kernel.ss_loc[i],:].T @ self.kernel.cond_cov_eig[i][1]
            result += np.sum(np.multiply(1/self.kernel.cond_cov_eig[i][0],np.square(temp)),axis=1)/self.sigma_sq
        return result*-0.5


class MixedGaussian:
    def __init__(self,K,spatial,ss_loc,cov=None,dependency=None,d=5,kernel='laplacian',l=0.01):
        self.K = K    #number of cluster
        self.kernel = Kernel(spatial,ss_loc,cov,dependency,d,kernel,l,k=self.K)
        self.pi = [1/self.K]*K
        self.mean = [np.repeat(0,self.kernel.N)[:,np.newaxis]]*K
        self.delta = [1]*K
        self.sigma_sq = [1]*K

    def update_cond_mean(self):
        self.cond_dev = [0 for _ in range(self.K)]
        for k in range(self.K):
            self.cond_dev[k] = self.Y - self.mean[k]
            dev = self.Y - self.mean[k]
            for i in range(self.kernel.M):
                if len(self.kernel.dependency[i]) > 0:
                # print(self.delta)
                # print(np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta),self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[self.kernel.ds_loc[i],:])
                    self.cond_dev[k][self.kernel.ss_loc[i],:] = self.cond_dev[k][self.kernel.ss_loc[i],:] - np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta),self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[self.kernel.ds_loc[i],:]

    def ll(self):
        N,G = self.Y.shape
        ll = np.zeros((G,self.K))
        for k in range(self.K):
            ll[:,k] = np.log(2 * np.pi)*N + 2*np.log(self.sigma_sq[k])*N
            self.kernel.update_cond_cov(self.delta)
            self.update_cond_mean()
            for i in range(self.kernel.M):
                det = np.prod(self.kernel.cond_cov_eig[k][i][0])
                ll[:,k] += np.log(det)
            # print(self.kernel.ss_loc[i])
            # print(self.kernel.A[i])
                temp = self.cond_dev[k][self.kernel.ss_loc[i],:].T @ self.kernel.cond_cov_eig[k][i][1]
                ll[:,k] += np.sum(np.multiply(1/self.kernel.cond_cov_eig[k][i][0],np.square(temp)),axis=1)/self.sigma_sq[k]
        return ll*-0.5

    def update_param():
        return 0

    def run_cluster(self,Y,pi=None,mean=None,sigma_sq=None,delta=0,iter=500):
        self.Y = Y
        if pi is not None:
            self.pi = pi    
        if mean is not None:
            self.mean = mean    
        if sigma_sq is not None:
            self.sigma_sq = sigma_sq   
            
        self.delta = delta
        self.kernel.update_cond_cov(0)
        self.update_cond_mean()
        N,G = self.Y.shape
        # power = np.zeros((G,self.K))
        omega = np.zeros((G,self.K))
        converge = False
        count = 0
        while not converge:
            #expectation
            ll = self.ll()
            for k in range(self.kernel.K):
                for g in range(G):
                    omega[g,k] = 1/np.sum(self.pi / self.pi[k] * np.exp(ll[g]-ll[g][k]))
            for k in range(self.kernel.K):
                mean[k] = self.Y @ omega[:,k:(k+1)] / np.sum(omega[:,k])
            self.update_param()
            self.pi = np.average(omega,axis=0)
            count += 1
            if count > iter:
                converge = True
        return self.pi,self.mean,self.sigma_sq

    # def no_delta_cond_mean(self,Y,mean):
    #     self.kernel._calc_B()
    #     for k in range(self.K):
    #         cond_dev = Y - mean[k]
    #         dev = Y - mean[k]
    #         for i in range(self.kernel.M):
    #             if len(self.kernel.dependency[i]) > 0:
    #             # print(self.delta)
    #             # print(np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta),self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[self.kernel.ds_loc[i],:])
    #                 cond_dev[self.kernel.ss_loc[i],:] = cond_dev[self.kernel.ss_loc[i],:] - self.kernel.B[i] @ dev[self.kernel.ds_loc[i],:]

    # def update_cond_mean(self):
    #     self.cond_dev = [0] * self.K
    #     for k in range(self.K):
    #         self.cond_dev[k] = self.Y - self.mean[k]
    #         dev = self.Y - self.mean[k]
    #         for i in range(self.kernel.M):
    #             if len(self.kernel.dependency[i]) > 0:
    #             # print(self.delta)
    #             # print(np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta),self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[self.kernel.ds_loc[i],:])
    #                 self.cond_dev[k][self.kernel.ss_loc[i],:] = self.cond_dev[k][self.kernel.ss_loc[i],:] - np.multiply(1/(self.kernel.ds_eig[i][0]+self.delta),self.kernel.A[i]) @ self.kernel.ds_eig[i][1].T @ dev[self.kernel.ds_loc[i],:]

    # def calc_power(self,cond_dev):   #cond_dev: K * np.array((N,G)) 
    #     N,G = self.Y.shape
    #     power = np.zeros((G,self.K))
    #     for k in range(self.K):
    #         for i in range(self.kernel.M):
    #             temp = cond_dev[k][self.kernel.ss_loc[i],:].T @ self.kernel.cond_cov_eig[i][1]
    #             sum = np.sum(np.multiply(1/self.kernel.cond_cov_eig[i][0],np.square(temp)),axis=1)/self.sigma_sq   #shape G*1
    #             power[:,k:(k+1)] = power[:,k:(k+1)] + sum 
    #     return power
  
  
    # def run_cluster(self,pi=None,mean=None,sigma_sq=None,delta=0,iter=500):
    #     if pi is not None:
    #         self.pi = pi    
    #     if mean is not None:
    #         self.mean = mean    
    #     if sigma_sq is not None:
    #         self.sigma_sq = sigma_sq   
            
    #     self.delta = delta
    #     self.kernel.update_cond_cov(0)
    #     self.update_cond_mean()
    #     N,G = self.Y.shape
    #     # power = np.zeros((G,self.K))
    #     omega = np.zeros((G,self.K))
    #     converge = False
    #     count = 0
    #     while not converge:
    #         #expectation
    #         power = self.calc_power(self.cond_dev)     #not yet divided by sigma
    #         for k in range(self.kernel.K):
    #             for g in range(G):
    #                 omega[g][k] = 1/np.sum(self.pi / self.pi[k] * np.exp(power[g]-power[g][k]))
    #         for k in range(self.kernel.K):
    #             mean[k] = self.Y @ omega[:,k:(k+1)] / np.sum(omega[:,k])
    #         self.update_param()
    #         self.pi = np.average(omega,axis=0)
    #         count += 1
    #         if count > iter:
    #             converge = True
    #     return self.pi,self.mean,self.sigma_sq

        



        
        