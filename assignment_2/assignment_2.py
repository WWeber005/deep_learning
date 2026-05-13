######################################################################
# Assignement 2
######################################################################
#
# The objective of this assignment is to build a 2-layer neural network.

import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
import copy


######################################################################
# Exercice 1
######################################################################
#
# 1.1 Read in the data & initialize the parameters of the network

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

# Load the data and compute mean_X and std_X.

trainX, trainY, train_y = loadBatch("data_batch_1")
valX, valY, val_y = loadBatch("data_batch_2")
testX, testY, test_y = loadBatch("test_batch")

d, n = trainX.shape
mean_trainX = np.mean(trainX, axis=1).reshape(d, 1)
std_trainX = np.std(trainX, axis=1).reshape(d, 1)

## normalization ##

trainX = (trainX - mean_trainX) / std_trainX
valX = (valX - mean_trainX) / std_trainX
testX = (testX - mean_trainX) / std_trainX

# 1.2 Network initialization

def initalize_net(seed,m,d):
    rng = np.random.default_rng(seed)
    init_net = {}
    init_net['W'] = [None] * 2
    init_net['b'] = [None] * 2
    
    init_net['W'][0] = (1/np.sqrt(d)) * rng.standard_normal(size=(m, d))
    init_net['b'][0] = np.zeros((m, 1))
    init_net['W'][1] = (1/np.sqrt(m)) * rng.standard_normal(size=(10, m))
    init_net['b'][1] = np.zeros((10, 1))
    return init_net

######################################################################
# Exercice 2
######################################################################
#

## Same as assignment 1, but now for a 2-layer network

def softMax(s):
    s_shift = s - np.max(s, axis=0, keepdims=True)
    exp_s = np.exp(s_shift)
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)
    
def ApplyNetwork(X, network):
    fp_data = {}
    s1 = network['W'][0] @ X + network['b'][0]
    h = np.maximum(0, s1)
    s = network['W'][1] @ h + network['b'][1]
    p = softMax(s)

    fp_data['s1'] = s1
    fp_data['h'] = h
    fp_data['s'] = s
    fp_data['p'] = p
    
    return fp_data

######################################################################
# 2.2 Compute the cost function
######################################################################
## took it from assignement 1
## 2.2.1 compute loss
def ComputeLoss(P, y):
    y = np.array(y)
    N = P.shape[1]
    return -np.mean(np.log(P[y, np.arange(N)]))

def ComputeCost(P, y, network, lam):
    loss = ComputeLoss(P, y)
    reg = lam * sum(np.sum(W**2) for W in network['W'])
    return reg + loss

######################################################################
# 2.3 Compute the Gradient
######################################################################

def BackwardPass(X, Y, fp_data, network, lam):
    N = X.shape[1]
    grads = {'W': [None] * 2, 'b': [None] * 2}
    h = fp_data['h']
    s1 = fp_data['s1']
    P = fp_data['p']

    G = P - Y

    grads['W'][1] = (G @ h.T) / N + 2 * lam * network['W'][1]
    grads['b'][1] = np.sum(G, axis=1, keepdims=True) / N

    G = network['W'][1].T @ G
    G = G * (s1 > 0) # Chain rule through the ReLU activation function.

    grads['W'][0] = (G @ X.T) / N + 2 * lam * network['W'][0]
    grads['b'][0] = np.sum(G, axis=1, keepdims=True) / N

    return grads


######################################################################
# Verification of our implementation (using the torch code)
######################################################################

def ComputeGradsWithTorch(X, y, network_params):
    
    Xt = torch.from_numpy(X)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True)        

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    #### BEGIN your code ###########################
    
    # Apply the scoring function corresponding to equations (1-3) in assignment description 
    # If X is d x n then the final scores torch array should have size 10 x n
    
    s1 = W[0] @ Xt + b[0]
    h = apply_relu(s1)
    scores = W[1] @ h + b[1]

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    n = X.shape[1]
    loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    return grads

# Test to ensure that the results are close

### TEST ###
L = 2 ## this assignement is a network of 2 layers
seed=0
d_small = 5
n_small = 3
m = 6
lam = 0
small_net = initalize_net(seed,m,d_small)
X_small = trainX[0:d_small, 0:n_small]
Y_small = trainY[:, 0:n_small]
fp_data = ApplyNetwork(X_small, small_net)
my_grads = BackwardPass(X_small, Y_small, fp_data, small_net, lam)
torch_grads = ComputeGradsWithTorch(X_small, train_y[0:n_small], small_net)

eps = 1e-6

res_W = []
res_b = []

for l in range(len(my_grads['W'])):
    w_diff = np.abs(my_grads['W'][l] - torch_grads['W'][l]) / np.maximum(
        eps,
        np.abs(my_grads['W'][l]) + np.abs(torch_grads['W'][l])
    )
    res_W.append(np.max(w_diff))

    b_diff = np.abs(my_grads['b'][l] - torch_grads['b'][l]) / np.maximum(
        eps,
        np.abs(my_grads['b'][l]) + np.abs(torch_grads['b'][l])
    )
    res_b.append(np.max(b_diff))

print("TEST Gradient between my code and pytorch")
for l in range(len(res_W)):
    print(f"Layer {l}: max relative error W = {res_W[l]}")
    print(f"Layer {l}: max relative error b = {res_b[l]}")
print(" ==============================================")

######################################################################
# Check by training a network
######################################################################
## Verify that everything implemented so far works correctly :)

import numpy as np
import copy

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

            Xbatch = X_shuffled[:, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            # forward pass
            fp_data = ApplyNetwork(Xbatch, trained_net)

            # backward pass
            grads = BackwardPass(Xbatch, Ybatch, fp_data, trained_net, lam)

            # Update layer by layer
            for l in range(len(trained_net['W'])):
                trained_net['W'][l] -= eta * grads['W'][l]
                trained_net['b'][l] -= eta * grads['b'][l]

        # ----- Evaluation -----
        fp_data_train = {}
        fp_data_train = ApplyNetwork(X, trained_net)
        trainP = fp_data_train['p']

        fp_data_val = ApplyNetwork(valX, trained_net)
        valP = fp_data_val['p']

        train_loss = ComputeLoss(trainP, y)
        val_loss = ComputeLoss(valP, val_y)

        train_cost = ComputeCost(trainP, y, trained_net, lam)
        val_cost = ComputeCost(valP, val_y, trained_net, lam)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_costs.append(train_cost)
        val_costs.append(val_cost)

        print(f"Epoch {epoch+1}/{n_epochs}: train loss = {train_loss:.6f}, val loss = {val_loss:.6f}")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_cost': train_costs,
        'val_cost': val_costs,
    }

    return trained_net, history

## initalisation ##
GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = .001
GDparams['n_epochs'] = 40 ## number of times the model goes through all mini-batches
GDparams['lam'] = 0.0
L = 2 ## this assignement is a network of 2 layers
seed = 0
m = 50 ## demandé au début du document
lam = 0.0
d = trainX.shape[0]
init_net = initalize_net(seed,m,d)

## TRAINING ##

# Testing on 100 samples
trained_net, history = miniBatchGD(trainX[:, :100], trainY[:, :100], train_y[:100], valX, val_y, GDparams, init_net, seed)


######################################################################
# 3. Train your network with cyclical learning rates
######################################################################

import numpy as np
import copy

def BatchGD_cyclical_learning_rates(X, Y, y, valX, valY, val_y,
                                    init_net, lam, seed,n_s = 500, n_cycles = 1):
    trained_net = copy.deepcopy(init_net)

    eta_min = 1e-5
    eta_max = 1e-1
    n_batch = 100

    n = X.shape[1]
    rng = np.random.default_rng(seed)

    train_losses = []
    val_losses = []
    train_costs = []
    val_costs = []
    train_accs = []
    val_accs = []
    steps = []

    t = 1
    t_max = 2 * n_s * n_cycles

    # As in Figure 3: about 10 measurements per cycle
    eval_every = max(1, t_max // 10)

    while t < t_max:
        perm = rng.permutation(n)
        X_shuffled = X[:, perm]
        Y_shuffled = Y[:, perm]

        for j in range(n // n_batch):
            if t >= t_max:
                break

            j_start = j * n_batch
            j_end = (j + 1) * n_batch

            Xbatch = X_shuffled[:, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            fp_data = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, fp_data, trained_net, lam)

            l_cycle = t // (2 * n_s)

            if 2 * l_cycle * n_s <= t <= (2 * l_cycle + 1) * n_s:
                eta = eta_min + (t - 2 * l_cycle * n_s) / n_s * (eta_max - eta_min)
            else:
                eta = eta_max - (t - (2 * l_cycle + 1) * n_s) / n_s * (eta_max - eta_min)

            for layer in range(len(trained_net['W'])):
                trained_net['W'][layer] -= eta * grads['W'][layer]
                trained_net['b'][layer] -= eta * grads['b'][layer]

            t += 1
            # Do not evaluate at every update in order to keep smoother curves
            if (t % eval_every == 0) or (t == t_max):
                trainP = ApplyNetwork(X, trained_net)['p']
                valP = ApplyNetwork(valX, trained_net)['p']

                train_loss = ComputeLoss(trainP, y)
                val_loss = ComputeLoss(valP, val_y)

                train_cost = ComputeCost(trainP, y, trained_net, lam)
                val_cost = ComputeCost(valP, val_y, trained_net, lam)

                train_acc = ComputeAccuracy(trainP, y)
                val_acc = ComputeAccuracy(valP, val_y)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_costs.append(train_cost)
                val_costs.append(val_cost)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                steps.append(t)

                print(f"Update {t}/{t_max}: "
                      f"train loss = {train_loss:.6f}, val loss = {val_loss:.6f}, "
                      f"train acc = {train_acc:.4f}, val acc = {val_acc:.4f}")

    history = {
        'steps': steps,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_cost': train_costs,
        'val_cost': val_costs,
        'train_acc': train_accs,
        'val_acc' : val_accs
    }

    return trained_net, history

# Function to compute accuracy

def ComputeAccuracy(P, y):
    pred = np.argmax(P, axis=0)
    y = np.array(y)
    return np.mean(pred == y)

# Plot training curves

def PlotTrainingCurves(history):
    steps = history['steps']

    plt.figure(figsize=(18, 5))

    # Cost
    plt.subplot(1, 3, 1)
    plt.plot(steps, history['train_cost'], label='training')
    plt.plot(steps, history['val_cost'], label='validation')
    plt.xlabel('update step')
    plt.ylabel('cost')
    plt.title('Cost plot', fontsize=18, fontweight='bold')
    plt.legend()

    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(steps, history['train_loss'], label='training')
    plt.plot(steps, history['val_loss'], label='validation')
    plt.xlabel('update step')
    plt.ylabel('loss')
    plt.title('Loss plot', fontsize=18, fontweight='bold')
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(steps, history['train_acc'], label='training')
    plt.plot(steps, history['val_acc'], label='validation')
    plt.xlabel('update step')
    plt.ylabel('accuracy')
    plt.title('Accuracy plot', fontsize=18, fontweight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Final test for Exercise 3

lam = 0.01
seed = 0
rng = np.random.default_rng(seed=0)
m = 50
d = trainX.shape[0]

init_net = initalize_net(seed,m,d)

trained_net, history = BatchGD_cyclical_learning_rates(
    X=trainX,
    Y=trainY,
    y=train_y,
    valX=valX,
    valY=valY,
    val_y=val_y,
    init_net=init_net,
    lam=lam,
    seed=seed
)
fp_data = ApplyNetwork(testX,trained_net)
test_acc = ComputeAccuracy(fp_data['p'], test_y)

PlotTrainingCurves(history)

print(f"Final test accuracy: {100 * test_acc:.2f}%")

# The training curves obtained with cyclical learning rates match the expected behavior shown in Figure 3. The loss and cost decrease smoothly while the accuracy increases over time. The validation accuracy reaches approximately 46.47%, which is consistent with the reference result of 46.29%.

######################################################################
# Exercice 4:
######################################################################
#
# We will test with n_s = 800 and 3 cycles

lam = 0.01
seed = 0
rng = np.random.default_rng(seed=0)
m = 50
d = trainX.shape[0]

init_net=initalize_net(seed,m,d)

trained_net, history = BatchGD_cyclical_learning_rates(
    X=trainX,
    Y=trainY,
    y=train_y,
    valX=valX,
    valY=valY,
    val_y=val_y,
    init_net=init_net,
    lam=lam,
    seed=seed,
    n_s = 800, 
    n_cycles = 3
)

fp_data = ApplyNetwork(testX,trained_net)
test_acc = ComputeAccuracy(fp_data['p'], test_y)

PlotTrainingCurves(history)

print(f"Final test accuracy: {100 * test_acc:.2f}%")

######################################################################
# Exercice 4:
######################################################################
# First perform a coarse search over a very broad range of values for lam


# Load all 5 training batches
trainX1, trainY1, train_y1 = loadBatch("data_batch_1")
trainX2, trainY2, train_y2 = loadBatch("data_batch_2")
trainX3, trainY3, train_y3 = loadBatch("data_batch_3")
trainX4, trainY4, train_y4 = loadBatch("data_batch_4")
trainX5, trainY5, train_y5 = loadBatch("data_batch_5")

# Concatenate all training data (50,000 images)
trainX_all = np.hstack((trainX1, trainX2, trainX3, trainX4, trainX5))
trainY_all = np.hstack((trainY1, trainY2, trainY3, trainY4, trainY5))
train_y_all = np.array(train_y1 + train_y2 + train_y3 + train_y4 + train_y5)

# Shuffle before splitting into training / validation
seed_ex4 = 0
rng_ex4 = np.random.default_rng(seed_ex4)
perm = rng_ex4.permutation(trainX_all.shape[1])

trainX_all = trainX_all[:, perm]
trainY_all = trainY_all[:, perm]
train_y_all = train_y_all[perm]

# Use 5,000 for validation
trainX_full = trainX_all[:, :-5000]
trainY_full = trainY_all[:, :-5000]
train_y_full = train_y_all[:-5000]

valX_full = trainX_all[:, -5000:]
valY_full = trainY_all[:, -5000:]
val_y_full = train_y_all[-5000:]


# Normalize using the statistics of the 45,000-image training split
mean_trainX_full = np.mean(trainX_full, axis=1, keepdims=True)
std_trainX_full = np.std(trainX_full, axis=1, keepdims=True)

trainX_full = (trainX_full - mean_trainX_full) / std_trainX_full
valX_full = (valX_full - mean_trainX_full) / std_trainX_full



# Utility function to save results
def save_lambda_results(results, filename):
    with open(filename, "w") as f:
        f.write("l,lambda,best_val_acc\n")
        for l, lam, acc in results:
            f.write(f"{l:.8f},{lam:.8e},{acc:.6f}\n")



# Random search over lambda

def search_best_lam_training(l_max, l_min, n_samples=8):
    seed = 0
    rng = np.random.default_rng(seed)
    results = []

    n_batch = 100
    n = trainX_full.shape[1]
    n_s = int(2 * np.floor(n / n_batch))
    m = 50
    d_full = trainX_full.shape[0]

    for _ in range(n_samples):
        l = l_min + (l_max - l_min) * rng.random()
        lam = 10 ** l

        # Use the same initialization for each lambda in order to compare fairly
        init_net = initalize_net(seed, m, d_full)

        trained_net, history = BatchGD_cyclical_learning_rates(
            X=trainX_full,
            Y=trainY_full,
            y=train_y_full,
            valX=valX_full,
            valY=valY_full,
            val_y=val_y_full,
            init_net=init_net,
            lam=lam,
            seed=seed,
            n_s=n_s,
            n_cycles=2
        )

        # Use the best validation accuracy reached during training
        best_val_acc = np.max(history['val_acc'])
        results.append((l, lam, best_val_acc))

        print(f"coarse search -> l = {l:.4f}, lambda = {lam:.8f}, best val acc = {best_val_acc:.4f}")

    # Sort from best to worst
    results.sort(key=lambda x: x[2], reverse=True)
    return results



# Fine search around the best lambda found in the coarse search

def fine_search_lambda(coarse_results, delta=0.3, n_samples=8):
    best_l = coarse_results[0][0]
    l_min_fine = best_l - delta
    l_max_fine = best_l + delta

    print(f"fine search interval: [{l_min_fine:.4f}, {l_max_fine:.4f}]")
    return search_best_lam_training(l_max=l_max_fine, l_min=l_min_fine, n_samples=n_samples)


# Run the searches and save the results
coarse_results = search_best_lam_training(-1, -5, n_samples=8)
fine_results_1 = fine_search_lambda(coarse_results, delta=0.3, n_samples=8)
fine_results_2 = fine_search_lambda(fine_results_1, delta=0.1, n_samples=8)
fine_results_3 = fine_search_lambda(fine_results_2, delta=0.03, n_samples=8)

best_lambda = fine_results_3[0][1]
print("Best lambda:", best_lambda)
print("Best validation accuracy:", fine_results_3[0][2]*100,"%")

# FINAL RUN with 1000 examples for validation and about 3 training cycles
# New final split: 49,000 training examples / 1,000 validation examples
trainX_final = trainX_all[:, :-1000]
trainY_final = trainY_all[:, :-1000]
train_y_final = train_y_all[:-1000]

valX_final = trainX_all[:, -1000:]
valY_final = trainY_all[:, -1000:]
val_y_final = train_y_all[-1000:]

# Recompute normalization from the final training split
mean_trainX_final = np.mean(trainX_final, axis=1, keepdims=True)
std_trainX_final = np.std(trainX_final, axis=1, keepdims=True)

trainX_final = (trainX_final - mean_trainX_final) / std_trainX_final
valX_final = (valX_final - mean_trainX_final) / std_trainX_final

testX_final, testY_final, test_y_final = loadBatch("test_batch")
testX_final = (testX_final - mean_trainX_final) / std_trainX_final

# Recompute n_s for the final run
n_batch = 100
n_final = trainX_final.shape[1]
n_s_final = int(2 * np.floor(n_final / n_batch))

seed = 0
m = 50
d_final = trainX_final.shape[0]

init_net_ex4 = initalize_net(seed, m, d_final)

trained_net_final, history_final = BatchGD_cyclical_learning_rates(
    X=trainX_final,
    Y=trainY_final,
    y=train_y_final,
    valX=valX_final,
    valY=valY_final,
    val_y=val_y_final,
    init_net=init_net_ex4,
    lam=best_lambda,
    seed=seed,
    n_s=n_s_final,
    n_cycles=3
)

fp_data_test = ApplyNetwork(testX_final, trained_net_final)
test_acc_final = ComputeAccuracy(fp_data_test['p'], test_y_final)

print(f"Final test accuracy with best lambda: {100 * test_acc_final:.2f}%")
PlotTrainingCurves(history_final)