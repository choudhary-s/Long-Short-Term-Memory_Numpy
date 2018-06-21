import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-1*x))
def tanh(x):
    return np.tanh(x)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def lstm_cell(xt,a_prev, c_prev, parameters):
    wf = parameters["wf"]
    wu = parameters["wu"]
    wc = parameters["wc"]
    wo = parameters["wo"]
    wy = parameters["wy"]
    bf = parameters["bf"]
    bu = parameters["bu"]
    bc = parameters["bc"]
    bo = parameters["bo"]
    by = parameters["by"]
    
    nx, m = xt.shape
    na, m = a_prev.shape
    
    concat = np.zeros(((na+nx),m))
    concat[: na, :] = a_prev
    concat[na :, :] = xt
    
    forget = sigmoid(np.dot(wf, concat)+bf)
    update = sigmoid(np.dot(wu, concat)+bu)
    cell_up1 = tanh(np.dot(wc, concat)+bc)
    cell_up2 = forget*c_prev + update*cell_up1
    output = sigmoid(np.dot(wo,concat)+bo)
    a_next = output*tanh(cell_up2)
    y_next = softmax(np.dot(wy, a_next)+by)
    
    cache = (a_next, cell_up2, a_prev, c_prev, xt, parameters)
    
    return a_next, cell_up2, y_next, cache
    


def lstm_forward(X, a0, c0, parameters):
    nx, m, Tx = X.shape
    ny, na = parameters["wy"].shape
    
    a = np.zeros((na, m, Tx))
    c = np.zeros((na, m, Tx))
    y = np.zeros((ny, m, Tx))
    caches = []
    
    a_next = a0
    c_next = c0
    
    for i in range(Tx):
        a_next, c_next, y_next, cache = lstm_cell(X[:,:,i], a_next, c_next, parameters)
        a[:,:,i] = a_next
        c[:,:,i] = c_next
        y[:,:,i] = y_next
        caches.append(cache)
    
    caches = (caches, X)
    return a, y, c, caches
        

nx = 3
na = 5
m  = 10
ny = 2
Tx = 7
np.random.seed(1)
wf = np.random.randn(na, na+nx)
wu = np.random.randn(na, na+nx)
wc = np.random.randn(na, na+nx)
wo = np.random.randn(na, na+nx)
wy = np.random.randn(ny, na)
bf = np.random.randn(na, 1)
bu = np.random.randn(na, 1)
bc = np.random.randn(na, 1)
bo = np.random.randn(na, 1)
by = np.random.randn(ny, 1)
a0 = np.random.randn(na, m)
c0 = np.zeros((na, m))
X  = np.random.randn(nx, m, Tx)

parameters = {"wf":wf, "wu":wu, "wc":wc, "wy":wy, "bf":bf, "bu":bu, "bc":bc, "by":by, "wo":wo, "bo":bo}

a, y, c, caches = lstm_forward(X, a0, c0, parameters)

print("-------------------------------------------I m checking------------------------------------------------")
print("a.shape", a.shape)
print("y.shape", y.shape)
print("c.shape", c.shape)
print("caches length", len(caches))
print("a[4][3][6]", a[4][3][6])
print("y[1][4][3]", y[1][4][3])
print("caches[1][2][1]", caches[1][2][1])