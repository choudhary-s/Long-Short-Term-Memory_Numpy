import numpy as np

def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1/(1+np.exp(-1*x))
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

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

def lstm_cell(xt, a0, c_prev, parameters):
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
    na, m = a0.shape
    
    concat = np.zeros((na+nx, m))
    concat[: na, :] = a0
    concat[na :, :] = xt
    
    forget = sigmoid(np.dot(wf, concat)+bf)
    update = sigmoid(np.dot(wu, concat)+bu)
    cell_up = tanh(np.dot(wc, concat)+bc)
    c_next = forget*c_prev + update*cell_up
    output = sigmoid(np.dot(wo, concat)+bo)
    a_next = output*tanh(c_next)
    y_predict = softmax(np.dot(wy, a_next)+by)
    
    cache = (a_next, c_next, a0, c_prev, forget, update, cell_up, output, xt, parameters)
    return a_next, c_next, y_predict, cache

    
    

nx = 3
na = 5
m = 10
ny = 2

np.random.seed(100)
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
xt = np.random.randn(nx, m)
c_prev = np.random.randn(na, m)

da_next = np.random.randn(na, m)
dc_next = np.random.randn(na, m)

parameters = {"wf":wf, "wu":wu, "wc":wc, "wo":wo, "wy":wy, "bf":bf, "bu":bu, "bc":bc, "bo":bo, "by":by}

a_next, c_next, y_predict, cache = lstm_cell(xt, a0, c_prev, parameters)
gradients = backprop_cell(da_next, dc_next, cache)
print("------------------------------I am just checking------------------------------")
print("a_next.shape", a_next.shape)
print("c_next.shape", c_next.shape)
print("y_predict.shape", y_predict.shape)
print("cache length", len(cache))
print("gradients length", len(gradients))
print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
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
print("gradients[\"dbu\"][4] =", gradients["dbu"][4])
print("gradients[\"dbu\"].shape =", gradients["dbu"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)