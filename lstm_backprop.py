import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-1*x))
def tanh(x):
    return np.tanh(x)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def backprop_cell(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, forget, update, cct, output, xt, parameters) = cache
    wf = parameters["wf"]
    wu = parameters["wu"]
    wc = parameters["wc"]
    wo = parameters["wo"]
    
    dot = da_next*output*(1-output)*tanh(c_next)
    dcct = (da_next*output*(1-tanh(c_next)**2)+dc_next)*update*(1-tanh(cct)**2)
    dut = (da_next*output*(1-tanh(c_next)**2)+dc_next)*update*(1-update)*cct
    dft = (da_next*output*(1-tanh(c_next)**2)+dc_next)*c_prev*forget*(1-forget)
    
    dwo = np.dot(dot, np.hstack([a_prev.T, xt.T]))
    dwc = np.dot(dcct, np.hstack([a_prev.T, xt.T]))
    dwu = np.dot(dut, np.hstack([a_prev.T, xt.T]))
    dwf = np.dot(dft, np.hstack([a_prev.T, xt.T]))
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbu = np.sum(dut, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)
    
    da_prev = np.dot(wo[:,: na].T, dot) + np.dot(wf[:,: na].T, dft) + np.dot(wu[:,: na].T, dut) + np.dot(wc[:,: na].T, dcct)
    dxt = np.dot(wo[:,na :].T, dot) + np.dot(wf[:,na :].T, dft) + np.dot(wu[:,na :].T, dut) + np.dot(wc[:,na :].T, dcct)
    dc_prev = (da_next*output*(1-tanh(c_next)**2)+dc_next)*forget
    
    gradients = {"dwo":dwo, "dwc":dwc, "dwu":dwu, "dwf":dwf, "dbf":dbf, "dbu":dbu, "dbc":dbc, "dbo":dbo, "da_prev":da_prev, "dxt":dxt, "dc_prev":dc_prev}
    return gradients

def lstm_backwards(da, caches):
    (cache, x) = caches
    (a_next, c_next, a_prev, c_prev, forget, update, cct, output, xt, parameters) = cache[0]
    
    nx, m, Tx = x.shape
    na, m = a_next.shape
    
    dx = np.zeros((nx, m, Tx))
    da0 = np.zeros((na, m))
    dwf = np.zeros((na, na+nx))
    dwu = np.zeros((na, na+nx))
    dwc = np.zeros((na, na+nx))
    dwo = np.zeros((na, na+nx))
    dc_next = np.zeros((na, m))
    da_next = np.zeros((na, m))
    dbf = np.zeros((na, 1))
    dbu = np.zeros((na, 1))
    dbc = np.zeros((na, 1))
    dbo = np.zeros((na, 1))
    
    for i in reversed(range(Tx)):
        gradients = backprop_cell(da[:,:,i]+da_next, dc_next, cache[i])
        dx[:,:,i] = gradients["dxt"]
        dwf += gradients["dwf"]
        dwu += gradients["dwu"]
        dwc += gradients["dwc"]
        dwo += gradients["dwo"]
        
        dbf += gradients["dbf"]
        dbu += gradients["dbu"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]
    da0 = gradients["da_prev"]
    
    gradients = {"dx":dx, "da0":da0, "dwf":dwf, "dbf":dbf, "dwu":dwu, "dbu":dbu, "dwc":dwc, "dbc":dbc, "dwo":dwo, "dbo":dbo}
    return gradients
    

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
    
    cache = (a_next, cell_up2, a_prev, c_prev, forget, update, cell_up1, output, xt, parameters)
    
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

da = np.random.randn(na, m, Tx)
gradients = lstm_backwards(da, caches)

print("-------------------------------------------I m checking------------------------------------------------")
print("a.shape", a.shape)
print("y.shape", y.shape)
print("c.shape", c.shape)
print("caches length", len(caches))
print("a[4][3][6]", a[4][3][6])
print("y[1][4][3]", y[1][4][3])
print("caches[1][2][1]", caches[1][2][1])
print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dwf\"][3][1] =", gradients["dwf"][3][1])
print("gradients[\"dwf\"].shape =", gradients["dwf"].shape)
print("gradients[\"dwu\"][1][2] =", gradients["dwu"][1][2])
print("gradients[\"dwu\"].shape =", gradients["dwu"].shape)
print("gradients[\"dwc\"][3][1] =", gradients["dwc"][3][1])
print("gradients[\"dwc\"].shape =", gradients["dwc"].shape)
print("gradients[\"dwo\"][1][2] =", gradients["dwo"][1][2])
print("gradients[\"dwo\"].shape =", gradients["dwo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients["dbu"][4])
print("gradients[\"dbi\"].shape =", gradients["dbu"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)