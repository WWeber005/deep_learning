import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
import copy
import time

#############################################################
#### Exercice 1: implement the convolution efficiently
#############################################################

import numpy as np
import torch

debug_file = 'debug_info.npz'
load_data = np.load(debug_file)
X = load_data['X']
Fs = load_data['Fs']

# for getting back the dimension of image 
n = X.shape[1]
X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
f = Fs.shape[0]
nf = Fs.shape[3]
num_images = X_ims.shape[3]
n_p_side = 32 // f
n_p = n_p_side * n_p_side
## 1.2 Compute convolution applied with stride f
## F.shape[0] == f est la largeur du filtre
## Un sub-patch (ou simplement un "patch") est un petit carré de pixels découpé dans cette grande image.
def compute_convolution(X_imgs, F):
    
    results = np.zeros((n_p_side, n_p_side, nf, num_images))
    
    for i in range(num_images):
        for k in range(nf):
            for row in range(n_p_side):
                for col in range(n_p_side):
                    patch = X_imgs[row*f : (row+1)*f, col*f : (col+1)*f, :, i]
                    results[row, col, k, i] = np.sum(np.multiply(patch, F[:, :, :, k]))
    
    return results     

## debug_part ##
print("DEBUG : COMPUTE_CONV")
conv_output = load_data['conv_outputs']
results = compute_convolution(X_ims,Fs)
print("COMPARE WITH NUMPY, IS IT THE SAME :",np.array_equal(results, conv_output))
print("MAX ERROR:", np.max(np.abs(results - conv_output)))
## I will directly use np.float32 for MX ( question about memory )

def build_MX(X_imgs, f):
    num_images = X_imgs.shape[3]
    n_p_side = 32 // f
    n_p = n_p_side * n_p_side

    MX = np.zeros((n_p, f*f*3, num_images), dtype=np.float32) ## use float32 for ensuring no problem of memory

    for i in range(num_images):
        l = 0
        for row in range(n_p_side):
            for col in range(n_p_side):

                X_patch = X_imgs[row*f : (row+1)*f,col*f : (col+1)*f,:,i]
                MX[l, :, i] = X_patch.reshape((f*f*3,), order='C')
                l += 1

    return MX
    
def efficient_conv_compute(X_imgs,F):
    f = F.shape[0]
    nf = F.shape[3]
    MX = build_MX(X_imgs, f)
    Fs_flat = F.reshape((f*f*3, nf), order='C')
    conv_outputs_mat = np.einsum('ijn,jl->iln', MX, Fs_flat, optimize=True)
    return conv_outputs_mat

## DEBUG_PART ##
print("DEBUG : COMPUTE_CONV_EFFICIENTLY")
conv_outputs = load_data['conv_outputs']
conv_outputs_flat = conv_outputs.reshape((n_p, nf, num_images), order='C')
results = efficient_conv_compute(X_ims,Fs)
print("COMPARE WITH NUMPY, IS IT THE SAME :", np.allclose(results, conv_outputs_flat))
print("MAX ERROR:", np.max(np.abs(results - conv_outputs_flat)))



#############################################################
#### Exercice 2: Compute gradients
#############################################################

def softMax(s):
    s_shift = s - np.max(s, axis=0, keepdims=True)
    exp_s = np.exp(s_shift)
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)

def forwardPass(MX, network):
    ## si jamais les dimensions changent ##
    F = network['F']
    f = F.shape[0]
    nf = F.shape[3]
    n = MX.shape[2]
    n_p = MX.shape[0]
    # new step compare to assignment 2 #
    Fs_flat = F.reshape((f*f*3, nf), order='C')
    conv_outputs_mat = np.einsum('ijn,jl->iln', MX, Fs_flat, optimize=True)
    conv_flat = np.maximum(conv_outputs_mat.reshape((n_p*nf, n), order='C'), 0)
    ######

    fp_data = {}
    s1 = network['W'][0] @ conv_flat + network['b'][0]
    x1 = np.maximum(0, s1)
    s = network['W'][1] @ x1 + network['b'][1]
    p = softMax(s)
    fp_data = {
        'conv_outputs_mat': conv_outputs_mat,
        'conv_flat': conv_flat,
        's1': s1,
        'x1': x1,
        's': s,
        'p': p
    }
    
    return fp_data



#### DEBUG : Forward pass ####

print("DEBUG : FORWARD_PASS")

network_debug = {
    'F': Fs,
    'W': [load_data['W1'], load_data['W2']],
    'b': [load_data['b1'], load_data['b2']]
}

MX_debug = build_MX(X_ims, f)
fp_debug = forwardPass(MX_debug, network_debug)

conv_flat_debug = load_data['conv_flat']
X1_debug = load_data['X1']
P_debug = load_data['P']

print("conv_flat shape:", fp_debug['conv_flat'].shape, "expected:", conv_flat_debug.shape)
print("X1 shape:", fp_debug['x1'].shape, "expected:", X1_debug.shape)
print("P shape:", fp_debug['p'].shape, "expected:", P_debug.shape)

print("conv_flat allclose:", np.allclose(fp_debug['conv_flat'], conv_flat_debug))
print("conv_flat max error:", np.max(np.abs(fp_debug['conv_flat'] - conv_flat_debug)))

print("X1 allclose:", np.allclose(fp_debug['x1'], X1_debug))
print("X1 max error:", np.max(np.abs(fp_debug['x1'] - X1_debug)))

print("P allclose:", np.allclose(fp_debug['p'], P_debug))
print("P max error:", np.max(np.abs(fp_debug['p'] - P_debug)))


def BackwardPass(MX, Y, fp_data, network, lam):
    N = MX.shape[2]
    grads = {'F': None, 'W': [None] * 2, 'b': [None] * 2}
    conv_flat = fp_data['conv_flat']
    x1 = fp_data['x1']
    s1 = fp_data['s1']
    P = fp_data['p']


    G = P - Y

    grads['W'][1] = (G @ x1.T) / N + 2 * lam * network['W'][1]
    grads['b'][1] = np.sum(G, axis=1, keepdims=True) / N

    G = network['W'][1].T @ G
    G = G * (s1 > 0) # Chain rule through the ReLU activation function.

    grads['W'][0] = (G @ conv_flat.T) / N + 2 * lam * network['W'][0]
    grads['b'][0] = np.sum(G, axis=1, keepdims=True) / N

    ## new step compare to assignment 2 ##
    G_batch = network['W'][0].T @ G
    G_batch = G_batch * (conv_flat > 0)

    n_p = MX.shape[0]
    nf = network['F'].shape[3]

    GG = G_batch.reshape((n_p, nf, N), order='C')

    MXt = np.transpose(MX, (1, 0, 2))
    F_flat = network['F'].reshape((network['F'].shape[0] * network['F'].shape[1] * network['F'].shape[2], network['F'].shape[3]), order='C')
    grads['F'] = np.einsum('ijn,jln->il', MXt, GG, optimize=True) / N + 2 * lam * F_flat

    return grads


#### DEBUG : Backward pass ####
print("DEBUG : BACKWARD_PASS")
Y_debug = load_data['Y']
grads_debug = BackwardPass(MX_debug, Y_debug, fp_debug, network_debug, lam=0)

grad_Fs_flat_debug = load_data['grad_Fs_flat']
grad_Fs_flat = grads_debug['F'].reshape((f*f*3, nf), order='C')

print("grad_F shape:", grad_Fs_flat.shape, "expected:", grad_Fs_flat_debug.shape)
print("grad_F allclose:", np.allclose(grad_Fs_flat, grad_Fs_flat_debug))
print("grad_F max error:", np.max(np.abs(grad_Fs_flat - grad_Fs_flat_debug)))

######################################################################
# Verification of our implementation (using the torch code )
######################################################################

def ComputeGradsWithTorch(MX, Y, network_params, lam=0):
    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    MX_t = torch.tensor(MX, dtype=torch.float64)
    Y_t = torch.tensor(Y, dtype=torch.float64)

    F_np = network_params['F']
    f = F_np.shape[0]
    nf = F_np.shape[3]
    F_flat_np = F_np.reshape((f*f*3, nf), order='C')

    F = torch.tensor(F_flat_np, dtype=torch.float64, requires_grad=True)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True)        


    n_p = MX.shape[0]
    n = MX.shape[2]

    conv_outputs = torch.zeros((n_p, nf, n), dtype=torch.float64)
    for i in range(n):
        conv_outputs[:, :, i] = MX_t[:, :, i] @ F

    conv_flat = torch.relu(conv_outputs.reshape((n_p*nf, n)))
    s1 = W[0] @ conv_flat + b[0]
    x1 = torch.relu(s1)
    scores = W[1] @ x1 + b[1]

    P = torch.softmax(scores, dim=0)
    loss = -torch.sum(Y_t * torch.log(P)) / n

    if lam != 0:
        loss = loss + lam * (torch.sum(F * F) + torch.sum(W[0] * W[0]) + torch.sum(W[1] * W[1]))

    loss.backward()

    # extract the computed gradients and make them numpy arrays
    L = len(network_params['W'])
    grads = {}
    grads['F'] = F.grad.numpy()
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    return grads

#### DEBUG : Compare analytic gradients with Torch ####
print("DEBUG : TORCH GRADIENT CHECK")

torch_grads_debug = ComputeGradsWithTorch(MX_debug, Y_debug, network_debug, lam=0)
analytic_grads_debug = BackwardPass(MX_debug, Y_debug, fp_debug, network_debug, lam=0)

print("F grad allclose:", np.allclose(analytic_grads_debug['F'], torch_grads_debug['F'], atol=1e-8))
print("F grad max error:", np.max(np.abs(analytic_grads_debug['F'] - torch_grads_debug['F'])))

for layer in range(2):
    print(f"W{layer+1} grad allclose:", np.allclose(analytic_grads_debug['W'][layer], torch_grads_debug['W'][layer], atol=1e-8))
    print(f"W{layer+1} grad max error:", np.max(np.abs(analytic_grads_debug['W'][layer] - torch_grads_debug['W'][layer])))
    print(f"b{layer+1} grad allclose:", np.allclose(analytic_grads_debug['b'][layer], torch_grads_debug['b'][layer], atol=1e-8))
    print(f"b{layer+1} grad max error:", np.max(np.abs(analytic_grads_debug['b'][layer] - torch_grads_debug['b'][layer])))


######################################################################
# Upgrade: convolutional bias vector bF
######################################################################

def forwardPass_upgrade(MX, network):
    F = network['F']
    f = F.shape[0]
    nf = F.shape[3]
    n = MX.shape[2]
    n_p = MX.shape[0]

    Fs_flat = F.reshape((f*f*3, nf), order='C')
    conv_outputs_mat = np.einsum('ijn,jl->iln', MX, Fs_flat, optimize=True)
    conv_outputs_mat = conv_outputs_mat + network['bF'].reshape((1, nf, 1))

    conv_flat = np.maximum(conv_outputs_mat.reshape((n_p*nf, n), order='C'), 0)

    s1 = network['W'][0] @ conv_flat + network['b'][0]
    x1 = np.maximum(0, s1)
    s = network['W'][1] @ x1 + network['b'][1]
    p = softMax(s)

    fp_data = {
        'conv_outputs_mat': conv_outputs_mat,
        'conv_flat': conv_flat,
        's1': s1,
        'x1': x1,
        's': s,
        'p': p
    }

    return fp_data


def BackwardPass_upgrade(MX, Y, fp_data, network, lam):
    N = MX.shape[2]
    grads = {'F': None, 'bF': None, 'W': [None] * 2, 'b': [None] * 2}

    conv_flat = fp_data['conv_flat']
    x1 = fp_data['x1']
    s1 = fp_data['s1']
    P = fp_data['p']

    G = P - Y

    grads['W'][1] = (G @ x1.T) / N + 2 * lam * network['W'][1]
    grads['b'][1] = np.sum(G, axis=1, keepdims=True) / N

    G = network['W'][1].T @ G
    G = G * (s1 > 0)

    grads['W'][0] = (G @ conv_flat.T) / N + 2 * lam * network['W'][0]
    grads['b'][0] = np.sum(G, axis=1, keepdims=True) / N

    G_batch = network['W'][0].T @ G
    G_batch = G_batch * (conv_flat > 0)

    n_p = MX.shape[0]
    nf = network['F'].shape[3]
    f = network['F'].shape[0]

    GG = G_batch.reshape((n_p, nf, N), order='C')

    MXt = np.transpose(MX, (1, 0, 2))
    F_flat = network['F'].reshape((f*f*3, nf), order='C')

    grads['F'] = np.einsum('ijn,jln->il', MXt, GG, optimize=True) / N + 2 * lam * F_flat
    grads['bF'] = np.sum(GG, axis=(0, 2)).reshape((nf, 1)) / N

    return grads


def ComputeGradsWithTorch_upgrade(MX, Y, network_params, lam=0):
    ## Biais conbolutionnel bF added ##
    bF = torch.tensor(network_params['bF'], dtype=torch.float64, requires_grad=True)
    
    MX_t = torch.tensor(MX, dtype=torch.float64)
    Y_t = torch.tensor(Y, dtype=torch.float64)

    F_np = network_params['F']
    f = F_np.shape[0]
    nf = F_np.shape[3]
    F_flat_np = F_np.reshape((f*f*3, nf), order='C')

    F = torch.tensor(F_flat_np, dtype=torch.float64, requires_grad=True)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True)        


    n_p = MX.shape[0]
    n = MX.shape[2]

    conv_outputs = torch.zeros((n_p, nf, n), dtype=torch.float64)
    for i in range(n):
        conv_outputs[:, :, i] = MX_t[:, :, i] @ F + bF.T

    conv_flat = torch.relu(conv_outputs.reshape((n_p*nf, n)))
    s1 = W[0] @ conv_flat + b[0]
    x1 = torch.relu(s1)
    scores = W[1] @ x1 + b[1]

    P = torch.softmax(scores, dim=0)
    loss = -torch.sum(Y_t * torch.log(P)) / n

    if lam != 0:
        loss = loss + lam * (torch.sum(F * F) + torch.sum(W[0] * W[0]) + torch.sum(W[1] * W[1]))

    loss.backward()

    # extract the computed gradients and make them numpy arrays
    grads = {}
    grads['F'] = F.grad.numpy()
    grads['bF'] = bF.grad.numpy()
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    return grads


#### DEBUG : Forward pass with convolutional bias ####
print("DEBUG : FORWARD_PASS UPGRADE")

network_debug_upgrade = {
    'F': Fs,
    'bF': np.zeros((nf, 1)),
    'W': [load_data['W1'], load_data['W2']],
    'b': [load_data['b1'], load_data['b2']]
}

fp_debug_upgrade = forwardPass_upgrade(MX_debug, network_debug_upgrade)

print("conv_flat allclose:", np.allclose(fp_debug_upgrade['conv_flat'], conv_flat_debug))
print("conv_flat max error:", np.max(np.abs(fp_debug_upgrade['conv_flat'] - conv_flat_debug)))
print("X1 allclose:", np.allclose(fp_debug_upgrade['x1'], X1_debug))
print("X1 max error:", np.max(np.abs(fp_debug_upgrade['x1'] - X1_debug)))
print("P allclose:", np.allclose(fp_debug_upgrade['p'], P_debug))
print("P max error:", np.max(np.abs(fp_debug_upgrade['p'] - P_debug)))


#### DEBUG : Compare upgraded analytic gradients with Torch ####
print("DEBUG : TORCH GRADIENT CHECK UPGRADE")

torch_grads_upgrade = ComputeGradsWithTorch_upgrade(MX_debug, Y_debug, network_debug_upgrade, lam=0)
analytic_grads_upgrade = BackwardPass_upgrade(MX_debug, Y_debug, fp_debug_upgrade, network_debug_upgrade, lam=0)

print("F grad allclose:", np.allclose(analytic_grads_upgrade['F'], torch_grads_upgrade['F'], atol=1e-8))
print("F grad max error:", np.max(np.abs(analytic_grads_upgrade['F'] - torch_grads_upgrade['F'])))
print("bF grad allclose:", np.allclose(analytic_grads_upgrade['bF'], torch_grads_upgrade['bF'], atol=1e-8))
print("bF grad max error:", np.max(np.abs(analytic_grads_upgrade['bF'] - torch_grads_upgrade['bF'])))

for layer in range(2):
    print(f"W{layer+1} grad allclose:", np.allclose(analytic_grads_upgrade['W'][layer], torch_grads_upgrade['W'][layer], atol=1e-8))
    print(f"W{layer+1} grad max error:", np.max(np.abs(analytic_grads_upgrade['W'][layer] - torch_grads_upgrade['W'][layer])))
    print(f"b{layer+1} grad allclose:", np.allclose(analytic_grads_upgrade['b'][layer], torch_grads_upgrade['b'][layer], atol=1e-8))
    print(f"b{layer+1} grad max error:", np.max(np.abs(analytic_grads_upgrade['b'][layer] - torch_grads_upgrade['b'][layer])))


######################################################################
# Exercice 3: Train small networks with cyclic learning rates
######################################################################

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

def initialize_convnet(f, nf, nh, seed):
    rng = np.random.default_rng(seed)

    n_p_side = 32 // f
    n_p = n_p_side * n_p_side
    d0 = n_p * nf

    init_net = {}

    # Convolutional patchify layer
    # F has shape (f, f, 3, nf), consistent with forwardPass_upgrade.
    # He initialization: std = sqrt(2 / number_of_inputs_to_one_filter)
    init_net['F'] = (np.sqrt(2 / (f * f * 3))* rng.standard_normal(size=(f, f, 3, nf))).astype(np.float32)

    # Bias for the upgraded convolutional layer
    init_net['bF'] = np.zeros((nf, 1), dtype=np.float32)

    # Fully connected layers
    init_net['W'] = [None] * 2
    init_net['b'] = [None] * 2

    init_net['W'][0] = (np.sqrt(2 / d0)* rng.standard_normal(size=(nh, d0))).astype(np.float32)
    init_net['b'][0] = np.zeros((nh, 1), dtype=np.float32)
    init_net['W'][1] = (np.sqrt(2 / nh)* rng.standard_normal(size=(10, nh))).astype(np.float32)
    init_net['b'][1] = np.zeros((10, 1), dtype=np.float32)

    return init_net

def ComputeAccuracy(P, y):
    pred = np.argmax(P, axis=0)
    y = np.array(y)
    return np.mean(pred == y)

def ComputeLoss(P, y):
    y = np.array(y)
    N = P.shape[1]
    return -np.mean(np.log(P[y, np.arange(N)]))

def ComputeCost(P, y, network, lam):
    loss = ComputeLoss(P, y)
    reg = lam * (np.sum(network['F']**2) + sum(np.sum(W**2) for W in network['W']))
    return reg + loss

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

ApplyNetwork = forwardPass_upgrade
BackwardPass = BackwardPass_upgrade

import numpy as np
import copy

def BatchGD_cyclical_learning_rates(MX, Y, y, valX, val_y,
                                    init_net, lam, seed, n_s=500, n_cycles=1):
    trained_net = copy.deepcopy(init_net)

    eta_min = 1e-5
    eta_max = 1e-1
    n_batch = 100

    n = MX.shape[2]
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
        MX_shuffled = MX[:, :, perm]
        Y_shuffled = Y[:, perm]

        for j in range(n // n_batch):
            if t >= t_max:
                break

            j_start = j * n_batch
            j_end = (j + 1) * n_batch

            Xbatch = MX_shuffled[:, :, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            fp_data = ApplyNetwork(Xbatch, trained_net)
            grads = BackwardPass(Xbatch, Ybatch, fp_data, trained_net, lam)

            l_cycle = t // (2 * n_s)

            if 2 * l_cycle * n_s <= t <= (2 * l_cycle + 1) * n_s:
                eta = eta_min + (t - 2 * l_cycle * n_s) / n_s * (eta_max - eta_min)
            else:
                eta = eta_max - (t - (2 * l_cycle + 1) * n_s) / n_s * (eta_max - eta_min)

            trained_net['F'] -= eta * grads['F'].reshape(trained_net['F'].shape, order='C')
            trained_net['bF'] -= eta * grads['bF']

            for layer in range(len(trained_net['W'])):
                trained_net['W'][layer] -= eta * grads['W'][layer]
                trained_net['b'][layer] -= eta * grads['b'][layer]

            t += 1
            # Do not evaluate at every update in order to keep smoother curves
            if (t % eval_every == 0) or (t == t_max):
                trainP = ApplyNetwork(MX, trained_net)['p']
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


# New function: BatchGD_cyclical_learning_rates_increasing_steps
def BatchGD_cyclical_learning_rates_increasing_steps(MX, Y, y, valX, val_y,
                                                     init_net, lam, seed,
                                                     n_s=800,
                                                     n_cycles=3):
    trained_net = copy.deepcopy(init_net)

    eta_min = 1e-5
    eta_max = 1e-1
    n_batch = 100

    n = MX.shape[2]
    rng = np.random.default_rng(seed)

    train_losses = []
    val_losses = []
    train_costs = []
    val_costs = []
    train_accs = []
    val_accs = []
    steps = []

    total_updates = sum(2 * n_s * (2 ** i) for i in range(n_cycles))
    eval_every = max(1, total_updates // 20)

    t_global = 0

    for cycle in range(n_cycles):
        current_n_s = n_s * (2 ** cycle)
        cycle_updates = 2 * current_n_s
        t_cycle = 0

        while t_cycle < cycle_updates:
            perm = rng.permutation(n)
            MX_shuffled = MX[:, :, perm]
            Y_shuffled = Y[:, perm]

            for j in range(n // n_batch):
                if t_cycle >= cycle_updates:
                    break

                j_start = j * n_batch
                j_end = (j + 1) * n_batch

                Xbatch = MX_shuffled[:, :, j_start:j_end]
                Ybatch = Y_shuffled[:, j_start:j_end]

                fp_data = ApplyNetwork(Xbatch, trained_net)
                grads = BackwardPass(Xbatch, Ybatch, fp_data, trained_net, lam)

                if t_cycle <= current_n_s:
                    eta = eta_min + (t_cycle / current_n_s) * (eta_max - eta_min)
                else:
                    eta = eta_max - ((t_cycle - current_n_s) / current_n_s) * (eta_max - eta_min)

                trained_net['F'] -= eta * grads['F'].reshape(trained_net['F'].shape, order='C')
                trained_net['bF'] -= eta * grads['bF']

                for layer in range(len(trained_net['W'])):
                    trained_net['W'][layer] -= eta * grads['W'][layer]
                    trained_net['b'][layer] -= eta * grads['b'][layer]

                t_cycle += 1
                t_global += 1

                if (t_global % eval_every == 0) or (t_global == total_updates):
                    trainP = ApplyNetwork(MX, trained_net)['p']
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
                    steps.append(t_global)

                    print(f"Update {t_global}/{total_updates}: "
                          f"cycle = {cycle + 1}/{n_cycles}, n_s = {current_n_s}, "
                          f"train loss = {train_loss:.6f}, val loss = {val_loss:.6f}, "
                          f"train acc = {train_acc:.4f}, val acc = {val_acc:.4f}")

    history = {
        'steps': steps,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_cost': train_costs,
        'val_cost': val_costs,
        'train_acc': train_accs,
        'val_acc': val_accs
    }

    return trained_net, history




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

trainX = (trainX_final - mean_trainX_final) / std_trainX_final
valX = (valX_final - mean_trainX_final) / std_trainX_final

testX_final, testY_final, test_y_final = loadBatch("test_batch")
testX = (testX_final - mean_trainX_final) / std_trainX_final


## remise en forme d image ##

train_ims = np.transpose(trainX.reshape((32, 32, 3, trainX.shape[1]), order='F'), (1, 0, 2, 3))

val_ims = np.transpose(valX.reshape((32, 32, 3, valX.shape[1]), order='F'), (1, 0, 2, 3))

test_ims = np.transpose(testX.reshape((32, 32, 3, testX.shape[1]), order='F'), (1, 0, 2, 3))

short_architectures = ['Arch1', 'Arch2', 'Arch3', 'Arch4']
short_accuracies = []
short_times = []


######################################################################
# Network architecture 1: f=2, nf=3, nh=50
######################################################################

f= 2
nf = 3
nh = 50
lam = 0.003

seed = 0
init_net = initialize_convnet(f, nf, nh, seed)

MX_train = build_MX(train_ims, f)
MX_val = build_MX(val_ims, f)
MX_test = build_MX(test_ims, f)

start = time.time()
trained_net, history = BatchGD_cyclical_learning_rates(
    MX=MX_train,
    Y=trainY_final,
    y=train_y_final,
    valX=MX_val,
    val_y=val_y_final,
    init_net=init_net,
    lam=0.003,
    seed=0,
    n_s=800,
    n_cycles=3
)
end = time.time()

training_time = end-start
short_times.append(training_time)
testP = ApplyNetwork(MX_test, trained_net)['p']
test_acc = ComputeAccuracy(testP, test_y_final)
short_accuracies.append(test_acc*100)

PlotTrainingCurves(history)
print(f"Final test accuracy: {100 * test_acc:.2f}%")

######################################################################
# Network architecture 2: f=4, nf=10, nh=50
######################################################################

f= 4
nf = 10
nh = 50
lam = 0.003

seed = 0
init_net = initialize_convnet(f, nf, nh, seed)

MX_train = build_MX(train_ims, f)
MX_val = build_MX(val_ims, f)
MX_test = build_MX(test_ims, f)

start = time.time()
trained_net, history = BatchGD_cyclical_learning_rates(
    MX=MX_train,
    Y=trainY_final,
    y=train_y_final,
    valX=MX_val,
    val_y=val_y_final,
    init_net=init_net,
    lam=0.003,
    seed=0,
    n_s=800,
    n_cycles=3
)
end = time.time()

training_time = end-start


testP = ApplyNetwork(MX_test, trained_net)['p']
test_acc = ComputeAccuracy(testP, test_y_final)

short_times.append(training_time)
short_accuracies.append(test_acc * 100)

PlotTrainingCurves(history)
print(f"Final test accuracy: {100 * test_acc:.2f}%")

######################################################################
# Network architecture 3: f=8, nf=40, nh=50
######################################################################

f= 8
nf = 40
nh = 50
lam = 0.003

seed = 0
init_net = initialize_convnet(f, nf, nh, seed)

MX_train = build_MX(train_ims, f)
MX_val = build_MX(val_ims, f)
MX_test = build_MX(test_ims, f)
start = time.time()
trained_net, history = BatchGD_cyclical_learning_rates(
    MX=MX_train,
    Y=trainY_final,
    y=train_y_final,
    valX=MX_val,
    val_y=val_y_final,
    init_net=init_net,
    lam=0.003,
    seed=0,
    n_s=800,
    n_cycles=3
)

end = time.time()

training_time = end-start


testP = ApplyNetwork(MX_test, trained_net)['p']
test_acc = ComputeAccuracy(testP, test_y_final)

short_times.append(training_time)
short_accuracies.append(test_acc * 100)

PlotTrainingCurves(history)
print(f"Final test accuracy: {100 * test_acc:.2f}%")

######################################################################
# Network architecture 4: f=16, nf=160, nh=50
######################################################################

f= 16
nf = 160
nh = 50
lam = 0.003

seed = 0
init_net = initialize_convnet(f, nf, nh, seed)

MX_train = build_MX(train_ims, f)
MX_val = build_MX(val_ims, f)
MX_test = build_MX(test_ims, f)
start = time.time()
trained_net, history = BatchGD_cyclical_learning_rates(
    MX=MX_train,
    Y=trainY_final,
    y=train_y_final,
    valX=MX_val,
    val_y=val_y_final,
    init_net=init_net,
    lam=0.003,
    seed=0,
    n_s=800,
    n_cycles=3
)

end = time.time()

training_time = end-start


testP = ApplyNetwork(MX_test, trained_net)['p']
test_acc = ComputeAccuracy(testP, test_y_final)

short_times.append(training_time)
short_accuracies.append(test_acc * 100)

PlotTrainingCurves(history)
print(f"Final test accuracy: {100 * test_acc:.2f}%")


######################################################################
# Train for longer Architecture 2
######################################################################

# Architecture 2 longer training: f=4, nf=10, nh=50
f = 4
nf = 10
nh = 50
lam = 0.003
seed = 0

init_net = initialize_convnet(f, nf, nh, seed)

MX_train = build_MX(train_ims, f)
MX_val = build_MX(val_ims, f)
MX_test = build_MX(test_ims, f)


trained_net_long_arch2, history_long_arch2 = BatchGD_cyclical_learning_rates_increasing_steps(
    MX=MX_train,
    Y=trainY_final,
    y=train_y_final,
    valX=MX_val,
    val_y=val_y_final,
    init_net=init_net,
    lam=lam,
    seed=seed,
    n_s=800,
    n_cycles=3
)

testP = ApplyNetwork(MX_test, trained_net_long_arch2)['p']
test_acc_long_arch2 = ComputeAccuracy(testP, test_y_final)


PlotTrainingCurves(history_long_arch2)
print(f"Final test accuracy long architecture 2: {100 * test_acc_long_arch2:.2f}%")


######################################################################
# Train for longer Architecture 3
######################################################################

# Architecture 3 longer training: f=8, nf=40, nh=50
f = 8
nf = 40
nh = 50
lam = 0.003
seed = 0

init_net = initialize_convnet(f, nf, nh, seed)

MX_train = build_MX(train_ims, f)
MX_val = build_MX(val_ims, f)
MX_test = build_MX(test_ims, f)

trained_net_long_arch3, history_long_arch3 = BatchGD_cyclical_learning_rates_increasing_steps(
    MX=MX_train,
    Y=trainY_final,
    y=train_y_final,
    valX=MX_val,
    val_y=val_y_final,
    init_net=init_net,
    lam=lam,
    seed=seed,
    n_s=800,
    n_cycles=3
)


testP = ApplyNetwork(MX_test, trained_net_long_arch3)['p']
test_acc_long_arch3 = ComputeAccuracy(testP, test_y_final)

PlotTrainingCurves(history_long_arch3)
print(f"Final test accuracy long architecture 3: {100 * test_acc_long_arch3:.2f}%")

######################################################################
# Bump number of filters for  Architecture 2 
######################################################################

# Architecture 2 longer training: f=4, nf=10, nh=50
f = 4
nf = 40
nh = 50
lam = 0.003
seed = 0

init_net = initialize_convnet(f, nf, nh, seed)

MX_train = build_MX(train_ims, f)
MX_val = build_MX(val_ims, f)
MX_test = build_MX(test_ims, f)
trained_net_long_arch2, history_long_arch2 = BatchGD_cyclical_learning_rates_increasing_steps(
    MX=MX_train,
    Y=trainY_final,
    y=train_y_final,
    valX=MX_val,
    val_y=val_y_final,
    init_net=init_net,
    lam=lam,
    seed=seed,
    n_s=800,
    n_cycles=3
)


testP = ApplyNetwork(MX_test, trained_net_long_arch2)['p']
test_acc_long_arch2 = ComputeAccuracy(testP, test_y_final)
PlotTrainingCurves(history_long_arch2)
print(f"Final test accuracy long architecture 2: {100 * test_acc_long_arch2:.2f}%")

## SAVE TIME TAKEN ##


# Accuracy bar chart

plt.figure()

plt.bar(short_architectures, short_accuracies)

plt.ylabel("Accuracy (%)")

plt.title("Final Test Accuracy")

plt.savefig("bar_accuracy.png")

plt.show()

# Training time bar chart

plt.figure()

plt.bar(short_architectures, short_times)

plt.ylabel("Time (seconds)")

plt.title("Training Time")

plt.savefig("bar_training_time.png")


######################################################################
# Exercice 4: Larger networks and regularization with label smoothing
######################################################################


# Network architecture 5: f=4, nf=40, nh=300

f = 4
nf = 40
nh = 300
lam = 0.0025
seed = 0

init_net = initialize_convnet(f, nf, nh, seed)

MX_train = build_MX(train_ims, f)
MX_val = build_MX(val_ims, f)
MX_test = build_MX(test_ims, f)

trained_net_long_arch2, history_long_arch2 = BatchGD_cyclical_learning_rates_increasing_steps(
    MX=MX_train,
    Y=trainY_final,
    y=train_y_final,
    valX=MX_val,
    val_y=val_y_final,
    init_net=init_net,
    lam=lam,
    seed=seed,
    n_s=800,
    n_cycles=4
)

testP = ApplyNetwork(MX_test, trained_net_long_arch2)['p']
test_acc_long_arch2 = ComputeAccuracy(testP, test_y_final)

PlotTrainingCurves(history_long_arch2)
print(f"Final test accuracy long architecture 2: {100 * test_acc_long_arch2:.2f}%")

#### Label smoothing ### 
def label_smoothing(Y, eps=0.1):
    K = Y.shape[0]
    return Y * (1 - eps) + (1 - Y) * eps / (K - 1)


trainY_smooth = label_smoothing(trainY_final, eps=0.1)


f = 4
nf = 40
nh = 300
lam = 0.0025
seed = 0

init_net = initialize_convnet(f, nf, nh, seed)

MX_train = build_MX(train_ims, f)
MX_val = build_MX(val_ims, f)
MX_test = build_MX(test_ims, f)

trained_net_long_arch2, history_long_arch2 = BatchGD_cyclical_learning_rates_increasing_steps(
    MX=MX_train,
    Y=trainY_smooth,
    y=train_y_final,
    valX=MX_val,
    val_y=val_y_final,
    init_net=init_net,
    lam=lam,
    seed=seed,
    n_s=800,
    n_cycles=4
)

testP = ApplyNetwork(MX_test, trained_net_long_arch2)['p']
test_acc_long_arch2 = ComputeAccuracy(testP, test_y_final)

PlotTrainingCurves(history_long_arch2)
print(f"Final test accuracy long architecture 2: {100 * test_acc_long_arch2:.2f}%")
