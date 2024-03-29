import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import os
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
        '''
        for i in range(n):
            for j in range(i+1,n):
                if (Plist[i][j] != np.transpose(Plist[j][i])).any():
                    print(i,j)
        '''
        #Plist = np.transpose(Plist)
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
                '''
                if(np.shape(Plist[i][j]) != (mlist[j],mlist[i]) ):
                    print(Plist[i][j],mlist[j],mlist[i])
                    return
                '''
                self.Plist[i][j] = tmp[ 0:mlist[j],0:mlist[i] ]

        print('building P')
        self.buildP()
        print('building C')
        self.buildC()
        print('building R')
        self.buildR()
        self.sol = None
        self.rounded_sol = None



    def buildP(self):
        self.P = np.zeros([self.N,self.N],np.double)
        n = self.n
        for i in range(n):
            for j in range(i,n):

                k = ( sum(sum(self.Plist[i][j])) >= sum(sum(self.Plist[j][i])))
                if k==True:
                    self.P[self.indBegin[j]:self.indEnd[j],self.indBegin[i]:self.indEnd[i]] = self.Plist[i][j]
                    self.P[self.indBegin[i]:self.indEnd[i], self.indBegin[j]:self.indEnd[j]] = np.transpose( self.Plist[i][j] )
                else:
                    self.P[self.indBegin[j]:self.indEnd[j], self.indBegin[i]:self.indEnd[i]] = np.transpose( self.Plist[j][i] )
                    self.P[self.indBegin[i]:self.indEnd[i], self.indBegin[j]:self.indEnd[j]] = self.Plist[j][i]

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
        print('m_bar = ',self.m_bar)
        self.Ax = np.array(self.Ax,np.double)
        self.Bx = np.array(self.Bx, np.double)
        self.Cx = np.array(self.Cx, np.double)


        cont  = 0

        while(True):

            tmp1, tmp2,tmp3 = self.Ax.copy(),self.Bx.copy(),self.Cx.copy()

            cont +=1
            print("iter%d"%cont)
            if cont == 100:
                break
            print('OPT A')
            self.Ax = self.optA()
            print('OPT B')
            self.Bx = self.optB()
            print('OPT C')
            self.Cx = self.optC()

            d = self.dist(tmp1,self.Ax) + self.dist(tmp2,self.Bx) + self.dist(tmp3,self.Cx)
            print('fraction rate = ',d)
            if d <= 1e-3 :
                break

        tmp = self.Bx.dot(np.transpose(self.Ax)) + self.Ax.dot(np.transpose(self.Bx))
        tmp = tmp /2
        assert (tmp == np.transpose(tmp)).all(), '???'
        val,vec = np.linalg.eig(tmp)
        ind = val.argsort()
        ind = ind [::-1]
        vec = vec[:,ind]

        Q = vec[:,0:self.m_bar]
        #print(np.shape(Q))
        self.sol = np.transpose(Q)
        return np.transpose(Q)

    def dist(self, A, B):
        return np.max(np.abs(A-B) )

    def get_init(self):

        eigvalue,eigvector = np.linalg.eig(self.P)
        idx = eigvalue.argsort()
        idx = idx[::-1] # 数组调过来改为降序

        # 获得排序后的特征值和特征向量
        eigvalue = eigvalue[idx]
        eigvector = eigvector[:,idx]

        # 求最大的gap来确定m_bar
        l = len(eigvalue)
        dv = eigvalue[0:l-1] - eigvalue[1:l]
        m_bar = dv.argmax()+1 # 因为python是从零开始标的

        m_bar = max(2,m_bar) # 要不要这样？

        # 获得AB初值
        tmp  = np.dot( eigvector[:,0:m_bar] , np.diag(eigvalue[0:m_bar])**0.5 )

        Ctmp = np.ones([self.n,m_bar])

        return m_bar, tmp, tmp, Ctmp

    ### 注意这里用的是伪逆


    def optA(self):

        # 公式(17)

        H = np.zeros([self.N,self.m_bar,self.m_bar],np.double)
        for r in range(self.N):
            for t in range(self.n):
                B_tmp = self.Wrst[r,:,t]
                B_tmp = self.Bx.transpose() * B_tmp
                B_tmp = B_tmp.dot(self.Bx)
                C = self.Cx[t]
                for i in range(self.m_bar):
                    for j in range(self.m_bar):
                        H[r][i][j] += B_tmp[i][j]*C[i]*C[j]

        g = np.zeros([self.N,self.m_bar],np.double)

        for r in range(self.N):
            for s in range(self.N):
                C_tmp = np.multiply( self.Wrst[r][s], self.R[r][s] )
                C_tmp = np.dot(C_tmp,self.Cx)
                g[r] += np.multiply( self.Bx[s],C_tmp )

        ans = np.zeros([self.N,self.m_bar])

        for r in range(self.N):
            ans[r] = np.linalg.pinv(H[r]).dot(g[r])
        #ans = self.normalize(ans)
        return ans

    def optB(self):

        # 公式(19) ctrl c ctrl v 不规范，亲人两行泪

        H = np.zeros([self.N,self.m_bar,self.m_bar],np.double)

        for s in range(self.N):
            for t in range(self.n):
                A_tmp = self.Wrst[:,s,t]
                A_tmp = self.Ax.transpose() * A_tmp
                A_tmp = A_tmp.dot(self.Ax)
                C = self.Cx[t]
                for i in range(self.m_bar):
                    for j in range(self.m_bar):
                        H[s][i][j] += A_tmp[i][j]*C[i]*C[j]

        g = np.zeros([self.N,self.m_bar],np.double)


        for s in range(self.N):
            for r in range(self.N):
                C_tmp = np.multiply(self.Wrst[r][s], self.R[r][s])
                C_tmp = np.dot(C_tmp, self.Cx)
                g[s] += np.multiply(self.Ax[r], C_tmp)

        ans = np.zeros([self.N,self.m_bar])

        for s in range(self.N):
            ans[s] = np.linalg.pinv(H[s]).dot(g[s])

        return ans

    def optC(self):

        # 公式(20) 这个公式大小又双叒叕写错了
        H = np.zeros([self.n,self.m_bar,self.m_bar],np.double)

        for t in range(self.n):
            for r in range(self.N):
                B_tmp = self.Wrst[r, :, t]
                B_tmp = self.Bx.transpose() * B_tmp
                B_tmp = B_tmp.dot(self.Bx)
                A = self.Ax[r]
                for i in range(self.m_bar):
                    for j in range(self.m_bar):
                        H[t][i][j] += B_tmp[i][j] * A[i] * A[j]


        g = np.zeros([self.n,self.m_bar],np.double)


        for t in range(self.n):
            for s in range(self.N):
                A_tmp = np.multiply( self.Wrst[:,s,t], self.R[:,s,t] )
                A_tmp = np.dot(A_tmp,self.Ax)
                g[t] += np.multiply( self.Bx[s],A_tmp )
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

    def test_opt(self):
        g = np.zeros([self.n, self.m_bar], np.double)
        for r in range(self.N):
            for l in range(self.m_bar):
                for s in range(self.N):
                    for t in range(self.n):
                        g[t][l] += self.Wrst[r][s][t] * self.Ax[r][l] * self.Bx[s][l] * self.R[r][s][t]

        g1 = np.zeros([self.n, self.m_bar], np.double)
        for t in range(self.n):
            for s in range(self.N):
                A_tmp = np.multiply(self.Wrst[:, s, t], self.R[:, s, t])
                A_tmp = np.dot(A_tmp, self.Ax)
                g1[t] += np.multiply(self.Bx[s], A_tmp)
        print(g-g1)

    def rounded_solution(self, th=0.5, k=2, t=2, sol=None):

        '''
        :param th: 两个向量大于th算是一类
        :param k: 允许一个universal part 对应到某个物体的k个part
        :param t: 允许一个part对应到t个universal part
        :param sol: degbug用的，输入一个现成的小数解他就不用自己再求一个
        :return:
        '''

        if sol is None:
            sol = self.solution()

        sol = np.transpose(sol)

        n, m = np.shape(sol)

        print('n,m',n,m)

        for r in range(n):
            sol[r] = sol[r] / sum(sol[r]**2)**0.5

        N = np.cumsum(self.mList)
        flag = np.zeros([n], np.int)
        ans = np.zeros([n, 0], np.int)
        for r in range(n):
            if (flag[r] != 0):
                continue

            cur_ans = np.zeros([n, 1], np.int)
            cur_ans[r] = 1
            flag[r] = flag[r] + 1

            ob = np.where(N > r)
            ob = np.min(ob)

            if ob < self.n:
                for ob1 in range(ob + 1, self.n):

                    cc = np.dot(sol[r, :], np.transpose(sol[N[ob1 - 1] : N[ob1], :]))
                    q = flag[ N[ob1 - 1]:N[ob1] ]
                    mx = cc[ np.where( q <= t-1 ) ].copy()

                    mx = np.sort(mx)
                    mx = mx[::-1]
                    if np.size(mx) >= k:
                        mx = mx[0:k]

                    for j in mx:
                        z = np.zeros([0,0],np.int)
                        if j > th:
                            z = np.where(cc == j) + N[ob1 - 1]
                        if np.size(z) > 0:
                            cur_ans[z] = 1
                            flag[z] +=  1

            ans = np.hstack([ans, cur_ans])
            self.rounded_sol = ans.copy()
        return ans

    def print_rounded_sol(self,filename = ''):
        if filename != '':
            f = open(filename,'r')
            sys.stdout = f
        for i in self.rounded_sol:
            print(i[0],end = '')
            for j in i[1::]:
                print(',%d'%j,end = '')
            print()
        sys.__stdout__ = sys.__stdout__

    def read_data(self,file_name):
        Q_index = []
        tensor_result_file = open(file_name, 'r')
        while True:
            this_line = tensor_result_file.readline().strip()
            # print(this_line)
            if this_line == '':
                break
            this_line = this_line.split(',')

            this_line_inQ = []
            for i in this_line:
                this_line_inQ.append(int(i))

            Q_index.append(this_line_inQ)
        return np.array(Q_index,np.int)

    def visualize(self,data_num,data_path,save_path,start_num = 0, sol_path='' ):

        w, h = 256, 256

        # 读入
        if sol_path !='':
            #Q_index = self.read_data(sol_path)
            Q_index = np.load(sol_path)
        else :
            if self.rounded_sol is None:
                sol = self.rounded_solution()
            Q_index = self.rounded_sol

        assert data_num <= self.n,'data num is bigged than the n syntensor got when it is established'
        image_list_old = np.load(data_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        color_num = np.size(Q_index[0])
        cmap = plt.get_cmap('gist_ncar')
        colors_set = cmap(np.linspace(0, 1,color_num+1))
        image_label_index = np.zeros([data_num,color_num],np.int)

        new_image_label_list = []
        now_n = 0

        for image_index in range(data_num):
            this_image_old_label = image_list_old[image_index]
            this_image_part_nums = int(this_image_old_label.max())
            #print(this_image_part_nums)
            temp_part_index = []
            new_this_image = np.zeros([w, h],np.ndarray)
            for part_index in range(1, this_image_part_nums + 1):
                this_part_map = Q_index[now_n]
                now_n += 1
                this_part_new_label = np.where( this_part_map==1 )
                if(np.size(this_part_new_label) >1 ):
                    print('>1:', this_part_map)
                else:
                    print('<=1:',this_part_map)


                this_part_new_label = np.array( [cc for cc in this_part_new_label] )
                #print('????',  np.ndarray( [cc[0] for cc in this_part_new_label] ) )
                temp_part_index.append(this_part_new_label)
                #print('part',this_part_new_label)

                tmp_ind = np.where(this_image_old_label == part_index)
                l = np.shape(tmp_ind)[1]
                for ii in range(l):
                    new_this_image[tmp_ind[0][ii]][tmp_ind[1][ii]] = this_part_new_label + 1
                print('this_part',this_part_new_label+1)
            new_image_label_list.append(new_this_image)
        #print('$',type(new_image_label_list[0][0][0]) )

        for image_index in range(data_num):
            final_im = new_image_label_list[image_index]
            draw_image = np.zeros([w, h, 3])
            #max_color = int(final_im.max())
            myrand = [np.random.randint(0, i+1) for i in range(10)]
            for i in range(w):
                if i%10==0:
                    myrand = [ np.random.randint(0,i+1) for i in range(10)]
                for j in range(h):
                    c = final_im[i][j]
                    if type(c)!=np.ndarray:
                        continue
                    #print('c',c)
                    #if(np.size(c) >1):
                        #print('c>1:',c)
                    t = myrand[np.size(c[0])-1]
                    t = c[0][t]
                    #print(t)
                    draw_image[i][j] = colors_set[t][0:3] * 255
            draw_image = draw_image.astype(np.uint8)
            cv2.imwrite('%s/%04d.png' % (save_path, image_index+start_num), draw_image)