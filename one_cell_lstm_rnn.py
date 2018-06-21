import numpy as np
def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1/(1+np.exp(-1*x))
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def lstm_cell(xt,at_prev,c_prev,parameters):
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
    
    n_x,m = xt.shape
    n_a,m = at_prev.shape
    
    concat = np.zeros((n_a+n_x,m))
    concat[: n_a,:] = at_prev
    concat[n_a :,:] = xt
    
    forget = sigmoid(np.dot(wf,concat)+bf)
    update = sigmoid(np.dot(wu,concat)+bu)
    cell_up1 = tanh(np.dot(wc,concat)+bc)
    cell_up2 = forget*c_prev + update*cell_up1
    output = sigmoid(np.dot(wo,concat)+bo)
    at_next = output*tanh(cell_up2)
    yt_predict = softmax(np.dot(wy,at_next)+by)
    
    cache = (at_next,cell_up2,at_prev,c_prev,forget,update,cell_up1,output,xt,parameters)
    return at_next,cell_up2,yt_predict,cache
    

n_a = 5
n_x = 3
m = 10
n_y = 2

np.random.seed(1)
wf = np.random.randn(n_a,n_a+n_x)
wu = np.random.randn(n_a,n_a+n_x)
wc = np.random.randn(n_a,n_a+n_x)
wo = np.random.randn(n_a,n_a+n_x)
wy = np.random.randn(n_y,n_a)
bf = np.random.randn(n_a,1)
bu = np.random.randn(n_a,1)
bc = np.random.randn(n_a,1)
bo = np.random.randn(n_a,1)
by = np.random.randn(n_y,1)
at_prev = np.random.randn(n_a,m)
xt = np.random.randn(n_x,m)
c_prev = np.random.randn(n_a,m)

parameters = {"wf":wf, "wu":wu, "wc":wc, "wy":wy, "bf":bf, "bu":bu, "bc":bc, "by":by, "wo":wo, "bo":bo}

at_next,c_next, yt_predict,cache = lstm_cell(xt, at_prev, c_prev, parameters)

print("-----------------------------Just checking--------------------------------------")
print("at_next.shape ",at_next.shape)
print("c_next.shape ", c_next.shape)
print("yt_predict.shape ", yt_predict.shape)
print("len(cache) ", len(cache))
print("at_next[4] ",at_next[4])
print("c_next[2] ", c_next[2])
print("yt_predict[1] ", yt_predict[1])