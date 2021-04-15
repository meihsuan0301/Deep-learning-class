import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import cupy as cp
# filename = [
# 	["training_images","train-images-idx3-ubyte.gz"],
# 	["test_images","t10k-images-idx3-ubyte.gz"],
# 	["training_labels","train-labels-idx1-ubyte.gz"],
# 	["test_labels","t10k-labels-idx1-ubyte.gz"]
# ]

# def download_mnist():
#     base_url = "http://yann.lecun.com/exdb/mnist/"
#     for name in filename:
#         print("Downloading "+name[1]+"...")
#         request.urlretrieve(base_url+name[1], name[1])
#     print("Download complete.")

# def save_mnist():
#     mnist = {}
#     for name in filename[:2]:
#         with gzip.open(name[1], 'rb') as f:
#             mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
#     for name in filename[-2:]:
#         with gzip.open(name[1], 'rb') as f:
#             mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
#     with open("mnist.pkl", 'wb') as f:
#         pickle.dump(mnist,f)
#     print("Save complete.")

# def init():
#     download_mnist()
#     save_mnist()

# def load():
#     with open("mnist.pkl",'rb') as f:
#         mnist = pickle.load(f)
#     return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = cp.zeros((N, D_out))
    Z[cp.arange(N), Y] = 1
    return Z

def draw_losses(losses):
    t = cp.arange(len(losses)).get()
    plt.plot(t, losses)
    plt.show()

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]

class FC():
    """
    Fully connected layer
    """
    def __init__(self, D_in, D_out):
        #print("Build FC")
        self.cache = None
        #self.W = {'val': np.random.randn(D_in, D_out), 'grad': 0}
        self.W = {'val': cp.random.normal(0.0, cp.sqrt(2/D_in), (D_in,D_out)), 'grad': 0}
        self.b = {'val': cp.random.randn(D_out), 'grad': 0}

    def _forward(self, X):
        #print("FC: _forward")
        out = cp.dot(X, self.W['val']) + self.b['val']
        self.cache = X
        return out

    def _backward(self, dout):
        #print("FC: _backward")
        X = self.cache
        dX = cp.dot(dout, self.W['val'].T).reshape(X.shape)
        self.W['grad'] = cp.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)  
        self.b['grad'] = cp.sum(dout, axis=0)
        #self._update_params()
        return dX

    def _update_params(self, lr=0.001):
        # Update the parameters
        self.W['val'] -= lr*self.W['grad']
        self.b['val'] -= lr*self.b['grad']

class ReLU():
    """
    ReLU activation layer
    """
    def __init__(self):
        #print("Build ReLU")
        self.cache = None

    def _forward(self, X):
        #print("ReLU: _forward")
        out = cp.maximum(0, X)
        self.cache = X
        return out

    def _backward(self, dout):
        #print("ReLU: _backward")
        X = self.cache
        dX = cp.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX

class Sigmoid():
    """
    Sigmoid activation layer
    """
    def __init__(self):
        self.cache = None

    def _forward(self, X):
        self.cache = X
        x = cp.clip(x, -500, 500)
        return 1 / (1 + cp.exp(-X))

    def _backward(self, dout):
        X = self.cache
        dX = dout*X*(1-X)
        return dX
# def Sigmoid(x):
#     x = cp.clip(x, -500, 500)
#     return 1./(1 + cp.exp(-x))
                
class tanh():
    """
    tanh activation layer
    """
    def __init__(self):
        self.cache = X

    def _forward(self, X):
        self.cache = X
        return cp.tanh(X)

    def _backward(self, X):
        X = self.cache
        dX = dout*(1 - cp.tanh(X)**2)
        return dX

class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        #print("Build Softmax")
        self.cache = None

    def _forward(self, X):
        #print("Softmax: _forward")
        maxes = cp.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = cp.exp(X - maxes)
        Z = Y / cp.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)
        return Z # distribution

    def _backward(self, dout):
        X, Y, Z = self.cache
        dZ = cp.zeros(X.shape)
        dY = cp.zeros(X.shape)
        dX = cp.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = cp.argmax(Z[n])
            dZ[n,:] = cp.diag(Z[n]) - cp.outer(Z[n],Z[n])
            M = cp.zeros((N,N))
            M[:,i] = 1
            dY[n,:] = cp.eye(N) - M
        dX = cp.dot(dout,dZ)
        dX = cp.dot(dX,dY)
        return dX

class Dropout():
    """
    Dropout layer
    """
    def __init__(self, p=1):
        self.cache = None
        self.p = p

    def _forward(self, X):
        M = (cp.random.rand(*X.shape) < self.p) / self.p
        self.cache = X, M
        return X*M

    def _backward(self, dout):
        X, M = self.cache
        dX = dout*M/self.p
        return dX

# class Conv():
#     """
#     Conv layer
#     """
#     def __init__(self, Cin, Cout, F, stride=1, padding=0, bias=True):
#         self.Cin = Cin
#         self.Cout = Cout
#         self.F = F
#         self.S = stride
#         #self.W = {'val': np.random.randn(Cout, Cin, F, F), 'grad': 0}
#         self.W = {'val': cp.random.normal(0.0,cp.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0} # Xavier Initialization
#         self.b = {'val': cp.random.randn(Cout), 'grad': 0}
#         self.cache = None
#         self.pad = padding

#     def _forward(self, X):
#         X = cp.pad(X, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant')
#         (N, Cin, H, W) = X.shape
#         H_ = H - self.F + 1
#         W_ = W - self.F + 1
#         Y = cp.zeros((N, self.Cout, H_, W_))

#         for n in range(N):
#             for c in range(self.Cout):
#                 for h in range(H_):
#                     for w in range(W_):
#                         Y[n, c, h, w] = cp.sum(X[n, :, h:h+self.F, w:w+self.F] * self.W['val'][c, :, :, :]) + self.b['val'][c]

#         self.cache = X
#         return Y

#     def _backward(self, dout):
#         # dout (N,Cout,H_,W_)
#         # W (Cout, Cin, F, F)
#         X = self.cache
#         (N, Cin, H, W) = X.shape
#         H_ = H - self.F + 1
#         W_ = W - self.F + 1
#         W_rot = cp.rot90(cp.rot90(self.W['val']))

#         dX = cp.zeros(X.shape)
#         dW = cp.zeros(self.W['val'].shape)
#         db = cp.zeros(self.b['val'].shape)

#         # dW
#         for co in range(self.Cout):
#             for ci in range(Cin):
#                 for h in range(self.F):
#                     for w in range(self.F):
#                         dW[co, ci, h, w] = cp.sum(X[:,ci,h:h+H_,w:w+W_] * dout[:,co,:,:])

#         # db
#         for co in range(self.Cout):
#             db[co] = cp.sum(dout[:,co,:,:])

#         dout_pad = cp.pad(dout, ((0,0),(0,0),(self.F,self.F),(self.F,self.F)), 'constant')
#         #print("dout_pad.shape: " + str(dout_pad.shape))
#         # dX
#         for n in range(N):
#             for ci in range(Cin):
#                 for h in range(H):
#                     for w in range(W):
#                         #print("self.F.shape: %s", self.F)
#                         #print("%s, W_rot[:,ci,:,:].shape: %s, dout_pad[n,:,h:h+self.F,w:w+self.F].shape: %s" % ((n,ci,h,w),W_rot[:,ci,:,:].shape, dout_pad[n,:,h:h+self.F,w:w+self.F].shape))
#                         dX[n, ci, h, w] = cp.sum(W_rot[:,ci,:,:] * dout_pad[n, :, h:h+self.F,w:w+self.F])

#         return dX

# class MaxPool():
#     def __init__(self, F, stride):
#         self.F = F
#         self.S = stride
#         self.cache = None

#     def _forward(self, X):
#         # X: (N, Cin, H, W): maxpool along 3rd, 4th dim
#         (N,Cin,H,W) = X.shape
#         F = self.F
#         W_ = int(float(W)/F)
#         H_ = int(float(H)/F)
#         Y = cp.zeros((N,Cin,W_,H_))
#         M = cp.zeros(X.shape) # mask
#         for n in range(N):
#             for cin in range(Cin):
#                 for w_ in range(W_):
#                     for h_ in range(H_):
#                         Y[n,cin,w_,h_] = cp.max(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)])
#                         i,j = cp.unravel_index(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)].argmax(), (F,F))
#                         M[n,cin,F*w_+i,F*h_+j] = 1
#         self.cache = M
#         return Y

#     def _backward(self, dout):
#         M = self.cache
#         (N,Cin,H,W) = M.shape
#         dout = cp.array(dout)
#         #print("dout.shape: %s, M.shape: %s" % (dout.shape, M.shape))
#         dX = cp.zeros(M.shape)
#         for n in range(N):
#             for c in range(Cin):
#                 #print("(n,c): (%s,%s)" % (n,c))
#                 dX[n,c,:,:] = dout[n,c,:,:].repeat(2, axis=0).repeat(2, axis=1)
#         return dX*M
class Conv:
    	# 初始化权重（卷积核4维）、偏置、步幅、填充
    def __init__(self, Cin, Cout, X_H, X_W, F, stride, pad):

        self.stride = stride
        self.pad = pad
        # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_W = None
        self.W = {'val': cp.random.normal(0.0,cp.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0} # Xavier Initialization
        self.b = {'val': cp.random.randn(Cout), 'grad': 0}
        self.X_H = X_H
        self.X_W = X_W
        self.Cin = Cin

    def _forward(self, x):
        x = x.reshape(x.shape[0], self.Cin, self.X_H, self.X_W)
        # 卷积核大小
        FN, C, FH, FW = self.W['val'].shape
        # 数据数据大小
        N, C, H, W = x.shape
        # 计算输出数据大小
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        # 利用im2col转换为行
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 卷积核转换为列，展开为2维数组
        col_W = self.W['val'].reshape(FN, -1).T
        # 计算正向传播
        out = cp.dot(col, col_W) + self.b['val']
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def _backward(self, dout):
        # 卷积核大小
        FN, C, FH, FW = self.W['val'].shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        db = cp.sum(dout, axis=0)
        dW = cp.dot(self.col.T, dout)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)

        self.W['grad'] = dW
        self.b['grad'] = db

        dcol = cp.dot(dout, self.col_W.T)
        # 逆转换
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    input_data = cp.asarray(input_data)
    # 输入数据的形状
    # N：批数目，C：通道数，H：输入数据高，W：输入数据长
    N, C, H, W = input_data.shape  
    out_h = (H + 2*pad - filter_h)//stride + 1  # 输出数据的高
    out_w = (W + 2*pad - filter_w)//stride + 1  # 输出数据的长
    # 填充 H,W
#     print(type(input_data))
#     print(input_data)
    img = cp.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # (N, C, filter_h, filter_w, out_h, out_w)的0矩阵
    col = cp.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # 按(0, 4, 5, 1, 2, 3)顺序，交换col的列，然后改变形状
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = cp.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class MaxPool:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def _forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
		# 展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
		# 最大值
        arg_max = cp.argmax(col, axis=1)
        out = cp.max(col, axis=1)
        # 转换
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def _backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = cp.zeros((dout.size, pool_size))
        dmax[cp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

def NLLLoss(Y_pred, Y_true):
    """
    Negative log likelihood loss
    """
    loss = 0.0
    N = Y_pred.shape[0]
    M = cp.sum(Y_pred*Y_true, axis=1)
    for e in M:
        #print(e)
        if e == 0:
            loss += 500
        else:
            loss += -cp.log(e)
    return loss/N

class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax._forward(Y_pred)
        loss = NLLLoss(prob, Y_true)
        Y_serial = cp.argmax(Y_true, axis=1)
        dout = prob.copy()
        dout[cp.arange(N), Y_serial] -= 1
        return loss, dout

class SoftmaxLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        loss = NLLLoss(Y_pred, Y_true)
        Y_serial = cp.argmax(Y_true, axis=1)
        dout = Y_pred.copy()
        dout[cp.arange(N), Y_serial] -= 1
        return loss, dout

class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass


class TwoLayerNet(Net):

    #Simple 2 layer NN

    def __init__(self, N, D_in, H, D_out, weights=''):
        self.FC1 = FC(D_in, H)
        self.ReLU1 = ReLU()
        self.FC2 = FC(H, D_out)

        if weights == '':
            pass
        else:
            with open(weights,'rb') as f:
                params = pickle.load(f)
                self.set_params(params)

    def forward(self, X):
        h1 = self.FC1._forward(X)
        a1 = self.ReLU1._forward(h1)
        h2 = self.FC2._forward(a1)
        return h2

    def backward(self, dout):
        dout = self.FC2._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.FC1._backward(dout)

    def get_params(self):
        return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

    def set_params(self, params):
        [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params


class ThreeLayerNet(Net):

    #Simple 3 layer NN

    def __init__(self, N, D_in, H1, H2, D_out, weights=''):
        self.FC1 = FC(D_in, H1)
        self.ReLU1 = ReLU()
        self.FC2 = FC(H1, H2)
        self.ReLU2 = ReLU()
        self.FC3 = FC(H2, D_out)

        if weights == '':
            pass
        else:
            with open(weights,'rb') as f:
                params = pickle.load(f)
                self.set_params(params)

    def forward(self, X):
        h1 = self.FC1._forward(X)
        a1 = self.ReLU1._forward(h1)
        h2 = self.FC2._forward(a1)
        a2 = self.ReLU2._forward(h2)
        h3 = self.FC3._forward(a2)
        return h3

    def backward(self, dout):
        dout = self.FC3._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.FC2._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.FC1._backward(dout)

    def get_params(self):
        return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params


class LeNet5(Net):
    # LeNet5

    def __init__(self):
        self.conv1 = Conv(1, 6, 5)
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2,2)
        self.conv2 = Conv(6, 16, 5)
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool(2,2)
        self.FC1 = FC(16*4*4, 120)
        self.ReLU3 = ReLU()
        self.FC2 = FC(120, 84)
        self.ReLU4 = ReLU()
        self.FC3 = FC(84, 10)
        self.Softmax = Softmax()

        self.p2_shape = None

    def forward(self, X):
        h1 = self.conv1._forward(X)
        a1 = self.ReLU1._forward(h1)
        p1 = self.pool1._forward(a1)
        h2 = self.conv2._forward(p1)
        a2 = self.ReLU2._forward(h2)
        p2 = self.pool2._forward(a2)
        self.p2_shape = p2.shape
        fl = p2.reshape(X.shape[0],-1) # Flatten
        h3 = self.FC1._forward(fl)
        a3 = self.ReLU3._forward(h3)
        h4 = self.FC2._forward(a3)
        a5 = self.ReLU4._forward(h4)
        h5 = self.FC3._forward(a5)
        a5 = self.Softmax._forward(h5)
        return a5

    def backward(self, dout):
        #dout = self.Softmax._backward(dout)
        dout = self.FC3._backward(dout)
        dout = self.ReLU4._backward(dout)
        dout = self.FC2._backward(dout)
        dout = self.ReLU3._backward(dout)
        dout = self.FC1._backward(dout)
        dout = dout.reshape(self.p2_shape) # reshape
        dout = self.pool2._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.conv2._backward(dout)
        dout = self.pool1._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params

class SGD():
    def __init__(self, params, lr=0.001, reg=0):
        self.parameters = params
        self.lr = lr
        self.reg = reg

    def step(self):
        for param in self.parameters:
            param['val'] -= (self.lr*param['grad'] + self.reg*param['val'])

class SGDMomentum():
    def __init__(self, params, lr=0.001, momentum=0.99, reg=0):
        self.l = len(params)
        self.parameters = params
        self.velocities = []
        for param in self.parameters:
            self.velocities.append(cp.zeros(param['val'].shape))
        self.lr = lr
        self.rho = momentum
        self.reg = reg

    def step(self):
        for i in range(self.l):
            self.velocities[i] = self.rho*self.velocities[i] + (1-self.rho)*self.parameters[i]['grad']
            self.parameters[i]['val'] -= (self.lr*self.velocities[i] + self.reg*self.parameters[i]['val'])


"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""

# #mnist.init()
# X_train, Y_train, X_test, Y_test = load()
# X_train, X_test = X_train/float(255), X_test/float(255)
# X_train -= np.mean(X_train)
# X_test -= np.mean(X_test)

# batch_size = 64
# D_in = 784
# D_out = 10

# print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

# ### TWO LAYER NET FORWARD TEST ###
# #H=400
# #model = nn.TwoLayerNet(batch_size, D_in, H, D_out)
# H1=300
# H2=100
# model = ThreeLayerNet(batch_size, D_in, H1, H2, D_out)


# losses = []
# #optim = optimizer.SGD(model.get_params(), lr=0.0001, reg=0)
# optim = SGDMomentum(model.get_params(), lr=0.0001, momentum=0.80, reg=0.00003)
# criterion = CrossEntropyLoss()

# # TRAIN
# ITER = 25000
# for i in range(ITER):
# 	# get batch, make onehot
# 	X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
# 	Y_batch = MakeOneHot(Y_batch, D_out)

# 	# forward, loss, backward, step
# 	Y_pred = model.forward(X_batch)
# 	loss, dout = criterion.get(Y_pred, Y_batch)
# 	model.backward(dout)
# 	optim.step()

# 	if i % 100 == 0:
# 		print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
# 		losses.append(loss)


# # save params
# weights = model.get_params()
# with open("weights.pkl","wb") as f:
# 	pickle.dump(weights, f)

# draw_losses(losses)



# # TRAIN SET ACC
# Y_pred = model.forward(X_train)
# result = np.argmax(Y_pred, axis=1) - Y_train
# result = list(result)
# print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))

# # TEST SET ACC
# Y_pred = model.forward(X_test)
# result = np.argmax(Y_pred, axis=1) - Y_test
# result = list(result)
# print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(result.count(0)/X_test.shape[0]))
