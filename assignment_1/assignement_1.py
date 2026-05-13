import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import copy


# ========================================================================
# STEP 1
# ========================================================================

def loadBatch(filename):
    cifar_dir = 'cifar-10-batches-py/'
    with open(cifar_dir + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    # Extract the image data and cast to float from the dict dictionary
    X = dict[b'data'].astype(np.float64) / 255.0
    X = X.transpose()
    # one-hot representation of the label for each image
    y = list(dict[b'labels'])
    Y = np.eye(10, dtype=X.dtype)[y].T

    return X, Y, y

X,Y,y = loadBatch("data_batch_1")
print(" ==============================================")
print("Verification STEP 1")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("len(y):", len(y))
print(" ==============================================")




# ========================================================================
# STEP 2
# ========================================================================


trainX, trainY, train_y = loadBatch("data_batch_1")
valX, valY, val_y = loadBatch("data_batch_2")
testX, testY, test_y = loadBatch("test_batch")

d, n = trainX.shape
mean_trainX = np.mean(trainX, axis=1).reshape(d, 1)
std_trainX = np.std(trainX, axis=1).reshape(d, 1)

trainX = (trainX - mean_trainX) / std_trainX
valX = (valX - mean_trainX) / std_trainX
testX = (testX - mean_trainX) / std_trainX


# ========================================================================
# STEP 3
# ========================================================================

## initialize the W and b

W = np.empty((trainY.shape[0],trainX.shape[0]),dtype=float)
b = np.empty((trainY.shape[0],1),dtype=float)
#print(W.shape) -> verify dimensions
#print(b.shape)
init_net= {'W': W, 'b': b}
def random_assignment(network):
    rng = np.random.default_rng()
    # get the BitGenerator used by default_rng
    BitGen = type(rng.bit_generator)
    # use the state from a fresh bit generator
    seed = 42
    rng.bit_generator.state = BitGen(seed).state
    network['W'] = .01*rng.standard_normal(size = network['W'].shape)
    network['b'] = np.zeros(network['b'].shape)
    

random_assignment(init_net)


# ========================================================================
# STEP 4
# ========================================================================

def softMax(s):
    exp_s = np.exp(s)
    return exp_s/ np.sum(exp_s, axis=0, keepdims=True)

def ApplyNetwork(X,network):
    s = network['W'] @ X + network['b']
    return softMax(s)

P = ApplyNetwork(trainX,init_net)

# ========================================================================
# STEP 5
# ========================================================================
def ComputeLoss(P, y):
    y = np.array(y)
    N = P.shape[1]
    return -np.mean(np.log(P[y, np.arange(N)]))

def ComputeCost(P, y, network, lam):
    loss = ComputeLoss(P, y)
    return loss + lam * np.sum(network['W'] ** 2)

L = ComputeLoss(P,train_y)

# ========================================================================
# STEP 6
# ========================================================================

def ComputeAccuracy(P, y):
    pred = np.argmax(P, axis=0)
    y = np.array(y)
    return np.mean(pred == y)

acc = ComputeAccuracy(P,train_y)


# ========================================================================
# STEP 7
# ========================================================================

def BackwardPass(X, Y, P, network, lam):
    G = P - Y ## erreur de sortie du modèle P --> ces probabilité et Y les one hot fix ==> c est le gradient de la loss
    N = P.shape[1]
    grads = {}
    grads['W'] = 1/N * G @ X.T + 2*lam*network['W']
    grads['b'] = (1/N) * np.sum(G, axis=1, keepdims=True) ## axis=1 pour sommer sur les colonnes
    return grads

def ComputeGradsWithTorch(X, y, network_params):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)    
    
    N = X.shape[1]
    
    scores = torch.matmul(W, Xt)  + b;

    ## give an informative name to this torch class
    apply_softmax = torch.nn.Softmax(dim=0)

    # apply softmax to each column of scores
    P = apply_softmax(scores)
    
    ## compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))    

    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()

    return grads

### TEST ###
rng = np.random.default_rng(seed=0)
small_net = {}
d_small = 10
n_small = 3
lam = 0
small_net['W'] = .01* rng.standard_normal(size = (10, d_small))
small_net['b'] = np.zeros((10, 1))
X_small = trainX[0:d_small, 0:n_small]
Y_small = trainY[:, 0:n_small]
P = ApplyNetwork(X_small, small_net)
my_grads = BackwardPass(X_small, Y_small, P, small_net, lam)
torch_grads = ComputeGradsWithTorch(X_small, train_y[0:n_small], small_net)

eps = 1e-6
res_W = np.abs(my_grads['W'] - torch_grads['W']) / np.maximum(
    eps, np.abs(my_grads['W']) + np.abs(torch_grads['W'])
)

res_b = np.abs(my_grads['b'] - torch_grads['b']) / np.maximum(
    eps, np.abs(my_grads['b']) + np.abs(torch_grads['b'])
)
print("TEST step 7 without L2 regularization")
print("max relative error W =", np.max(res_W))
print("max relative error b =", np.max(res_b))
print(" ==============================================")

def ComputeGradsWithTorch_L2(X, y, network_params,lam):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)    
    
    N = X.shape[1]
    
    scores = torch.matmul(W, Xt)  + b;

    ## give an informative name to this torch class
    apply_softmax = torch.nn.Softmax(dim=0)

    # apply softmax to each column of scores
    P = apply_softmax(scores)
    
    ## compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))    

    # compute the backward by adding the cost
    cost = loss + lam * torch.sum(torch.multiply(W, W))
    cost.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()

    return grads


lam = 0.9

my_grads = BackwardPass(X_small, Y_small, P, small_net, lam)
torch_grads = ComputeGradsWithTorch_L2(X_small, train_y[0:n_small], small_net,lam)

res_W = np.abs(my_grads['W'] - torch_grads['W']) / np.maximum(
    eps, np.abs(my_grads['W']) + np.abs(torch_grads['W'])
)

res_b = np.abs(my_grads['b'] - torch_grads['b']) / np.maximum(
    eps, np.abs(my_grads['b']) + np.abs(torch_grads['b'])
)


print("TEST step 7 with L2 regularization")
print("max relative error W =", np.max(res_W))
print("max relative error b =", np.max(res_b))
print(" ==============================================")

# ========================================================================
# STEP 8
# ========================================================================

def miniBatchGD(X, Y, y, valX, val_y, GDparams, init_net, seed):
    trained_net = copy.deepcopy(init_net)

    n_batch = GDparams['n_batch']
    eta = GDparams['eta']
    n_epochs = GDparams['n_epochs']
    lam = GDparams['lam']
    n = X.shape[1]

    rng = np.random.default_rng(seed)

    train_losses = []
    val_losses = []
    train_costs = []
    val_costs = []

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_shuffled = X[:, perm]
        Y_shuffled = Y[:, perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = slice(j_start, j_end)

            Xbatch = X_shuffled[:, inds]
            Ybatch = Y_shuffled[:, inds]

            Pbatch = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, Pbatch, trained_net, lam)
            
            trained_net['W'] = trained_net['W'] - eta * grads['W']
            trained_net['b'] = trained_net['b'] - eta * grads['b']

        ## sauvegarde de tout les calculs fait ##
        trainP = ApplyNetwork(X, trained_net)
        valP = ApplyNetwork(valX, trained_net)

        train_loss = ComputeLoss(trainP, y)
        val_loss = ComputeLoss(valP, val_y)
        train_cost = ComputeCost(trainP, y, trained_net, lam)
        val_cost = ComputeCost(valP, val_y, trained_net, lam)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_costs.append(train_cost)
        val_costs.append(val_cost)

        print(f"Epoch {epoch + 1}/{n_epochs}: train loss = {train_loss:.6f}, val loss = {val_loss:.6f}, train cost = {train_cost:.6f}, val cost = {val_cost:.6f}")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_cost': train_costs,
        'val_cost': val_costs,
    }

    return trained_net, history

# ========================================================================
# fonctions annexes
# ========================================================================

def plot_result(epochs, history, trained_net):
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='training loss')
    plt.plot(epochs, history['val_loss'], label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Training and validation loss')
    plt.show()

    plt.figure()
    plt.plot(epochs, history['train_cost'], label='training cost')
    plt.plot(epochs, history['val_cost'], label='validation cost')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend()
    plt.title('Training and validation cost')
    plt.show()
    Ws = trained_net['W'].transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))

    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax = axs[i // 5, i % 5]
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        ax.imshow(w_im_norm)
        ax.set_title(f'class {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ========================================================================
# TEST 1: lam=0, n_epochs=40, n_batch=100, eta=.1
# ========================================================================

## initalisation ##
GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = .1
GDparams['n_epochs'] = 40 ## nombre de fois que je vais traverser les n_batch
GDparams['lam'] = 0
seed = 42
W = np.empty((trainY.shape[0],trainX.shape[0]),dtype=float)
b = np.empty((trainY.shape[0],1),dtype=float)
init_net= {'W': W, 'b': b}
random_assignment(init_net)

## TRAINING ##

trained_net, history = miniBatchGD(trainX, trainY, train_y, valX, val_y, GDparams, init_net, seed)
 

## TEST ## 
testP = ApplyNetwork(testX, trained_net)
test_acc = ComputeAccuracy(testP, test_y)
print("TEST 1 accuracy:", test_acc)

## RESULT ##
epochs = np.arange(1, GDparams['n_epochs'] + 1)
plot_result(epochs,history,trained_net)

# ========================================================================
# TEST 2 lam=0, n_epochs=40, n_batch=100, eta=.001
# ========================================================================

## initalisation ##
GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = .001
GDparams['n_epochs'] = 40 ## nombre de fois que je vais traverser les n_batch
GDparams['lam'] = 0
seed = 42
W = np.empty((trainY.shape[0],trainX.shape[0]),dtype=float)
b = np.empty((trainY.shape[0],1),dtype=float)
init_net= {'W': W, 'b': b}
random_assignment(init_net)

## TRAINING ##

trained_net, history = miniBatchGD(trainX, trainY, train_y, valX, val_y, GDparams, init_net, seed)
 

## TEST ## 
testP = ApplyNetwork(testX, trained_net)
test_acc = ComputeAccuracy(testP, test_y)
print("TEST 2 accuracy:", test_acc)

## RESULT ##
epochs = np.arange(1, GDparams['n_epochs'] + 1)
plot_result(epochs,history,trained_net)

# ========================================================================
# TEST 3 lam=.1, n_epochs=40, n_batch=100, eta=.001
# ========================================================================

## initalisation ##
GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = .001
GDparams['n_epochs'] = 40 ## nombre de fois que je vais traverser les n_batch
GDparams['lam'] = 0.1
seed = 42
W = np.empty((trainY.shape[0],trainX.shape[0]),dtype=float)
b = np.empty((trainY.shape[0],1),dtype=float)
init_net= {'W': W, 'b': b}
random_assignment(init_net)

## TRAINING ##

trained_net, history = miniBatchGD(trainX, trainY, train_y, valX, val_y, GDparams, init_net, seed)
 

## TEST ## 
testP = ApplyNetwork(testX, trained_net)
test_acc = ComputeAccuracy(testP, test_y)
print("TEST 3 accuracy:", test_acc)

## RESULT ##
epochs = np.arange(1, GDparams['n_epochs'] + 1)
plot_result(epochs,history,trained_net)

# ========================================================================
# TEST 4 lam=1, n_epochs=40, n_batch=100, eta=.001
# ========================================================================

## initalisation ##
GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = .001
GDparams['n_epochs'] = 40 ## nombre de fois que je vais traverser les n_batch
GDparams['lam'] = 1
seed = 42
W = np.empty((trainY.shape[0],trainX.shape[0]),dtype=float)
b = np.empty((trainY.shape[0],1),dtype=float)
init_net= {'W': W, 'b': b}
random_assignment(init_net)

## TRAINING ##

trained_net, history = miniBatchGD(trainX, trainY, train_y, valX, val_y, GDparams, init_net, seed)
 

## TEST ## 
testP = ApplyNetwork(testX, trained_net)
test_acc = ComputeAccuracy(testP, test_y)
print("TEST 4 accuracy:", test_acc)

## RESULT ##
epochs = np.arange(1, GDparams['n_epochs'] + 1)
plot_result(epochs,history,trained_net)