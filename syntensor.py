import numpy as np

class SynTensor:
    def __init__(self, n, mlist, Plist):
        '''
        :param self:
        :param n: shape的个数
        :param mlist: [m1,...,mn]的list
        :param Plist: P[i][j]是从j到i的映射，并且需要满足P[i][j]^T = P[j][i]
        '''
        assert len(mlist) == n,'Length of mlist must equal to n'
        assert np.shape(Plist) == (n,n), 'Size of Plist must be (n,n)'
        for i in range(n):
            for j in range(i+1,n):
                if (Plist[i][j] != np.transpose(Plist[j][i])).any():
                    print(i,j)
        Plist = np.transpose(Plist)
        self.n = n
        self.mList = mlist.copy()
        self.N = sum(mlist)

        # 构造索引序列,即前缀和
        self.indBegin = mlist.copy()
        s = 0
        for i in range(n):
            self.indBegin[i] = 0 if i==0 else self.indBegin[i-1]+mlist[i-1]
        self.indEnd = [ self.indBegin[i+1] if i!= n-1 else self.N for i in range(n)  ]
        # 处理Plist的型状，让他变成mi*mj的
        self.Plist = np.zeros((n,n),np.ndarray)
        for i in range(n):
            for j in range(n):
                tmp = np.array(Plist[i][j])
                self.Plist[i][j] = tmp[ 0:mlist[j],0:mlist[i] ]
        print('building P')
        self.buildP()
        print('building C')
        self.buildC()
        print('building R')
        self.buildR()



    def buildP(self):
        self.P = np.zeros([self.N,self.N],np.double)
        n = self.n
        for i in range(n):
            for j in range(n):
                print(i,j)
                self.P[self.indBegin[j]:self.indEnd[j],self.indBegin[i]:self.indEnd[i]] = self.Plist[i][j]


    def buildC(self):
        '''
        计算 Cijk =  Pki * Pjk * Pij 并舍弃对角线以外的值
        '''
        n = self.n
        self.Clist = np.zeros([n,n,n],np.ndarray)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    tmp = np.dot(self.Plist[j][k],self.Plist[i][j])
                    tmp = np.dot(self.Plist[k][i],tmp)
                    d = np.diag(tmp)
                    d = np.diag(d)
                    self.Clist[i][j][k] = np.where(d > 0, 1,0)

    def getRijk(self,i,j,k):
        '''
        :param i:
        :param j:
        :param k:
        :return: 计算Rijk = P[i][j] * C[i][j][k]
        '''
        return np.dot(self.Plist[i][j],self.Clist[i][j][k])

    def buildR(self):
        '''
        :return:
        Rlist 是分块的tensor
        R是整个的tensor
        '''
        n = self.n
        self.Rlist = np.zeros([n,n,n],np.ndarray)
        self.R = np.zeros([self.N, self.N, n], np.double)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    self.Rlist[i][j][k] = self.getRijk(i,j,k)
                    self.R[self.indBegin[j]:self.indEnd[j], self.indBegin[i]:self.indEnd[i],k] = self.Rlist[i][j][k]


    def build_Wrst(self):

        '''
        获取R中每个点都和哪个C相乘，方法是先把他们都设成全为1的数组，模拟一次与C相乘的结果就是与之相乘的Cijk
        :return:
        '''
        self.Wrst = np.ones([self.N,self.N,self.n],np.double)
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n): # 取出来一小片
                    tmp = self.Wrst[ self.indBegin[j]:self.indEnd[j], self.indBegin[i]:self.indEnd[i], k]
                    tmp = np.dot(tmp,self.Clist[i][j][k])
                    self.Wrst[self.indBegin[j]:self.indEnd[j], self.indBegin[i]:self.indEnd[i], k] = self.Rlist[i][j][k] = tmp


    def solution(self):

        self.build_Wrst()

        self.m_bar,self.Ax,self.Bx,self.Cx = self.get_init()
        print(self.m_bar)
        self.Ax = np.array(self.Ax,np.double)
        self.Bx = np.array(self.Bx, np.double)
        self.Cx = np.array(self.Cx, np.double)

        cont  = 0

        while(True):
            tmp1, tmp2,tmp3 = self.Ax.copy(),self.Bx.copy(),self.Cx.copy()
            cont +=1
            if cont == 100:
                break

            self.Ax = self.optA()
            self.Bx = self.optB()
            self.Cx = self.optC()
            #print(self.Ax, self.Bx, self.Cx)
            d = self.dist(tmp1,self.Ax) + self.dist(tmp2,self.Bx) + self.dist(tmp3,self.Cx)
            print(d)
            if d <= 1e-3 :
                break

        tmp = self.Bx.dot(np.transpose(self.Ax)) + self.Ax.dot(np.transpose(self.Bx))
        print(np.shape(tmp))
        tmp = tmp /2
        assert (tmp == np.transpose(tmp)).all(), '???'
        val,vec = np.linalg.eig(tmp)
        ind = val.argsort()
        ind = ind [::-1]
        vec = vec[:,ind]

        Q = vec[:,0:self.m_bar]
        print(np.shape(Q))
        return np.transpose(Q)

    def dist(self, A, B):
        return np.linalg.norm(A-B)

    def get_init(self):

        eigvalue,eigvector = np.linalg.eig(self.P)
        idx = eigvalue.argsort()
        idx = idx[::-1] # 数组调过来改为降序

        # 获得排序后的特征值和特征向量
        eigvalue = eigvalue[idx]
        print("eigvalue:",eigvalue)
        eigvector = eigvector[:,idx]

        # 求最大的gap来确定m_bar
        l = len(eigvalue)
        dv = eigvalue[0:l-1] - eigvalue[1:l]
        m_bar = dv.argmax()+1 # 因为python是从零开始标的

        #m_bar = max(self.mList) # 要不要这样？

        # 获得AB初值
        tmp  = np.dot( eigvector[:,0:m_bar] , np.diag(eigvalue[0:m_bar])**0.5 )

        Ctmp = np.ones([self.n,m_bar])

        return m_bar, tmp, tmp, Ctmp

    ### 注意这里用的是伪逆
    def optA(self):

        # 公式(17)
        H = np.zeros([self.N,self.m_bar,self.m_bar],np.double)
        for r in range(self.N):
            for i in range(self.m_bar):
                for j in range(self.m_bar):
                    for s in range(self.N):
                        for t in range(self.n):
                            H[r][i][j] += self.Wrst[r][s][t]*self.Bx[s][i]*self.Bx[s][j]*self.Cx[t][i] * self.Cx[t][j]


        g = np.zeros([self.N,self.m_bar],np.double)

        for r in range(self.N):
            for l in range(self.m_bar):
                for s in range(self.N):
                    for t in range(self.n):
                        g[r][l] += self.Wrst[r][s][t] * self.Bx[s][l] * self.Cx[t][l] * self.R[r][s][t]

        ans = np.zeros([self.N,self.m_bar])

        for r in range(self.N):
            ans[r] = np.linalg.pinv(H[r]).dot(g[r])
        #ans = self.normalize(ans)
        return ans

    def optB(self):

        # 公式(19) ctrl c ctrl v 不规范，亲人两行泪
        # k = self.m_bar
        H = np.zeros([self.N,self.m_bar,self.m_bar],np.double)
        for r in range(self.N):
            for i in range(self.m_bar):
                for j in range(self.m_bar):
                    for s in range(self.N):
                        for t in range(self.n):
                            H[s][i][j] += self.Wrst[r][s][t]*self.Ax[r][i]*self.Ax[r][j]*self.Cx[t][i] * self.Cx[t][j]


        g = np.zeros([self.N,self.m_bar],np.double)

        for r in range(self.N):
            for l in range(self.m_bar):
                for s in range(self.N):
                    for t in range(self.n):
                        g[s][l] += self.Wrst[r][s][t] * self.Ax[r][l] * self.Cx[t][l] * self.R[r][s][t]

        ans = np.zeros([self.N,self.m_bar])

        for s in range(self.N):
            ans[s] = np.linalg.pinv(H[s]).dot(g[s])

        return ans

    def optC(self):

        # 公式(20) 这个公式大小又双叒叕写错了
        H = np.zeros([self.n,self.m_bar,self.m_bar],np.double)
        for r  in range(self.N):
            for i in range(self.m_bar):
                for j in range(self.m_bar):
                    for s in range(self.N):
                        for t in range(self.n):
                            H[t][i][j] += self.Wrst[r][s][t]*self.Ax[r][i]*self.Ax[r][j]*self.Bx[s][i] * self.Bx[s][j]


        g = np.zeros([self.n,self.m_bar],np.double)

        for r in range(self.N):
            for l in range(self.m_bar):
                for s in range(self.N):
                    for t in range(self.n):
                        g[t][l] += self.Wrst[r][s][t] * self.Ax[r][l] * self.Bx[s][l] * self.R[r][s][t]

        ans = np.zeros([self.n,self.m_bar])

        for t in range(self.n):
            ans[t] = np.linalg.pinv(H[t]).dot(g[t])
        #ans = self.normalize(ans)
        return ans

    def normalize(self,vectors):
        _,m =np.shape(vectors)
        tmp = vectors.copy()
        for i in range(m):
            tmp[:,i] = tmp[:,i] / sum(tmp[:,i]**2)**0.5
        return tmp