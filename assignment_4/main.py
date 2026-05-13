import numpy as np
#import torch
import matplotlib.pyplot as plt
import pickle
import copy
import time
import sys
from torch_gradient_computations_column_wise import ComputeGradsWithTorchCW
from torch_gradient_computations_row_wise import ComputeGradsWithTorchRW

np.set_printoptions(threshold=sys.maxsize)


################################################################
### Exercice 1 : Implement and train a vanilla RNN
################################################################

## 0.1 Read in the data ## 

book_fname = 'goblet_book.txt'
fid = open(book_fname, "r")
book_data = fid.read()
fid.close()

unique_chars = list(set(book_data))
K = len(unique_chars)
print(f"value of K : {K}")


char_to_ind = {}
for idx, val in enumerate(unique_chars):
    char_to_ind[val] = idx

ind_to_char = {}
for idx, val in enumerate(unique_chars):
    ind_to_char[idx] = val

## 0.2 Init hyper-parameters ##
m=100
eta = .001
seq_length = 25
seed = 0
rng = np.random.default_rng(seed)

RNN = {}
RNN['U'] = (1/np.sqrt(2*K))*rng.standard_normal(size = (m, K))
RNN['W'] = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m))
RNN['V'] = (1/np.sqrt(m))*rng.standard_normal(size = (K, m))
RNN['b'] = np.zeros((m, 1))
RNN['c'] = np.zeros((K, 1))

## 0.3 Synthesize text from your randomly initialized RNN ##
# n = length of the sequence you want to generate

def softMax(s):
    s_shift = s - np.max(s, axis=0, keepdims=True)
    exp_s = np.exp(s_shift)
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)



def synthesize(h0,x0,n):
    Y = np.zeros((K,n))
    x = np.copy(x0)
    h = np.copy(h0)

    for t in range(n):
        a_t = RNN['W'] @ h + RNN['U'] @ x + RNN['b']
        h_t = np.tanh(a_t)
        o_t = RNN['V'] @ h_t + RNN['c']
        p_t = softMax(o_t)
        
        cp = np.cumsum(p_t, axis=0)
        a = rng.uniform(size=1)
        ii = np.argmax(cp - a > 0)

        x_next = np.zeros((K, 1))
        x_next[ii] = 1
        Y[:,t] = x_next.flatten()
        x = x_next
        h = h_t

    return Y

## 0.4 Implement the forward & backward pass of back-prop ##

X_chars = book_data[0:seq_length]
Y_chars = book_data[1:seq_length+1]

X = np.zeros((K, seq_length))
Y = np.zeros((K, seq_length))


for i in range(seq_length):
    X[char_to_ind[X_chars[i]], i] = 1
    Y[char_to_ind[Y_chars[i]], i] = 1


h0 = np.zeros((m,1))



def forward_pass(X,Y,h_prev,RNN):
    xs, hs, ps = {}, {}, {}
    hs[-1] = np.copy(h_prev)
    loss = 0
    for t in range(seq_length):
        xs[t] = X[:, t:t+1]

        a_t = RNN['W'] @ hs[t-1] + RNN['U'] @ xs[t] + RNN['b']
        hs[t] = np.tanh(a_t)
        
        o_t = RNN['V'] @ hs[t] + RNN['c']
        p_t = softMax(o_t)
        ps[t] = p_t / np.sum(p_t)

        loss += -np.log(ps[t][np.argmax(Y[:, t]), 0])
    return loss, xs, hs, ps

loss, xs, hs, ps = forward_pass(X,Y,h0,RNN)

def backward(xs, hs, ps, Y, RNN):
    """
    xs: cache of inputs (X)
    hs: cache of hidden states (including hs[-1])
    ps: cache of probabilities
    Y:  target one-hot matrix
    RNN: dictionary of parameters {W, U, V, b, c}
    """
    grads = {key: np.zeros_like(val) for key, val in RNN.items()}
    
    # dh_next represents the gradient flowing back from time t+1
    # It starts at 0 for the very last time step
    dh_next = np.zeros_like(hs[0])
    seq_length = Y.shape[1]

    # Iterating backwards through the sequence
    for t in reversed(range(seq_length)):
        # 1. Gradient of the loss w.r.t. the output logits (o_t)
        # dy = p_t - y_t
        dy = np.copy(ps[t])
        dy[np.argmax(Y[:, t])] -= 1 
        
        # 2. Gradients for the output weights and biases
        grads['V'] += dy @ hs[t].T
        grads['c'] += dy
        
        # 3. Gradient w.r.t. the hidden state h_t
        # This is the sum of gradients from the output layer and the next hidden state
        dh = RNN['V'].T @ dy + dh_next
        
        # 4. Backprop through the tanh nonlinearity
        # da = dh * (1 - tanh^2(a_t)) => da = dh * (1 - h_t^2)
        da = (1 - hs[t]**2) * dh
        
        # 5. Gradients for hidden-to-hidden and input-to-hidden weights
        grads['b'] += da
        grads['W'] += da @ hs[t-1].T
        grads['U'] += da @ xs[t].T
        
        # 6. Compute the gradient for the previous hidden state (h_{t-1})
        # This becomes the dh_next for the next iteration (t-1)
        dh_next = RNN['W'].T @ da
    
    for key in grads:
        grads[key] /= seq_length
        
    return grads

## --- TEST DE GRADIENT (Exercice 0.4) --- ##

# 1. On réduit les dimensions pour le test (m=10) pour éviter les instabilités
m_test = 10
seq_length_test = 25

# 2. Initialisation d'un mini-RNN pour le test
RNN_test = {}
RNN_test['U'] = (1/np.sqrt(2*K)) * rng.standard_normal(size=(m_test, K))
RNN_test['W'] = (1/np.sqrt(2*m_test)) * rng.standard_normal(size=(m_test, m_test))
RNN_test['V'] = (1/np.sqrt(m_test)) * rng.standard_normal(size=(K, m_test))
RNN_test['b'] = np.zeros((m_test, 1))
RNN_test['c'] = np.zeros((K, 1))

h0_test = np.zeros((m_test, 1))

# 3. Calcul des gradients analytiques (ton code)
# Note : Ton forward_pass actuel calcule la SOMME des pertes (pas la moyenne)
# Ton backward calcule donc le gradient de cette somme.
print(f"size of X  is ({X.shape})")
loss_val, xs_cache, hs_cache, ps_cache = forward_pass(X, Y, h0_test,RNN_test)
grads_analytiques = backward(xs_cache, hs_cache, ps_cache, Y, RNN_test)

# 4. Calcul des gradients avec PyTorch (La référence)
# On passe RNN_test, h0, X et Y
# Crée un vecteur d'indices pour PyTorch (le format attendu par P[y, ...])
y_indices = np.array([char_to_ind[char] for char in Y_chars])

# Appelle la fonction PyTorch avec les indices, pas la matrice one-hot
grads_pytorch = ComputeGradsWithTorchCW(X, y_indices, h0_test, RNN_test)

# 5. Comparaison et affichage de l'erreur relative
print(f"{'Paramètre':<10} | {'Erreur Relative Max':<20}")
print("-" * 35)

for key in RNN_test.keys():
    g_a = grads_analytiques[key]
    g_p = grads_pytorch[key]
    
    # Calcul de l'erreur relative : |a - b| / max(eps, |a| + |b|)
    relative_error = np.abs(g_a - g_p) / np.maximum(1e-9, np.abs(g_a) + np.abs(g_p))
    max_error = np.max(relative_error)
    
    print(f"{key:<10} | {max_error:.2e}")


## 0.5 Read in the data ## 
def rnn_training(book_data, seq_length, RNN, K, char_to_ind, ind_to_char, eta=0.001, max_iters=100000):
    # Adam parameters
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_adam = {key: np.zeros_like(val) for key, val in RNN.items()}
    v_adam = {key: np.zeros_like(val) for key, val in RNN.items()}

    # Training state
    e = 0
    t = 0
    hprev = np.zeros((RNN['U'].shape[0], 1))
    smooth_loss = -np.log(1.0/K) * seq_length 
    
    while t < max_iters:
        # 1. Handle Epoch Reset
        if e > len(book_data) - seq_length - 1:
            e = 0
            hprev = np.zeros((RNN['U'].shape[0], 1))
            print(f"\n--- End of book reached. Resetting e and hprev ---")

        # 2. Prepare Data
        X_chars = book_data[e:e+seq_length]
        Y_chars = book_data[e+1:e+seq_length+1]

        X = np.zeros((K, seq_length))
        Y = np.zeros((K, seq_length))
        for j in range(seq_length):
            X[char_to_ind[X_chars[j]], j] = 1
            Y[char_to_ind[Y_chars[j]], j] = 1

        # 3. Forward and Backward
        loss_val, xs_cache, hs_cache, ps_cache = forward_pass(X, Y, hprev, RNN)
        grads = backward(xs_cache, hs_cache, ps_cache, Y, RNN)

        # 4. Gradient Clipping
        for key in grads:
            np.clip(grads[key], -5, 5, out=grads[key])

        # 5. Adam Update
        t += 1
        for key in RNN.keys():
            m_adam[key] = beta1 * m_adam[key] + (1 - beta1) * grads[key]
            v_adam[key] = beta2 * v_adam[key] + (1 - beta2) * (grads[key] ** 2)

            m_hat = m_adam[key] / (1 - beta1 ** t)
            v_hat = v_adam[key] / (1 - beta2 ** t)
            RNN[key] -= eta * m_hat / (np.sqrt(v_hat) + eps)

        # 6. Update Pointers
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss_val
        hprev = hs_cache[seq_length - 1]
        e += seq_length

        # 7. Periodic Synthesis and Progress Report
        if t % 1000 == 0:
            print(f"\nIteration {t}, Smooth Loss: {smooth_loss/25:.4f}")
            
            # Use current hprev and the first character of the current sequence
            x0 = X[:, 0:1] 
            sample_indices = synthesize(hprev, x0, 200) # Assuming synthesize returns (K, n) matrix
            
            # Convert one-hot matrix back to string
            sample_text = "".join([ind_to_char[np.argmax(sample_indices[:, i])] for i in range(200)])
            print("-" * 30)
            print(f"Synthesized text:\n{sample_text}")
            print("-" * 30 + "\n")

    return RNN




rnn_training(book_data, seq_length, RNN, K, char_to_ind, ind_to_char, eta=0.001, max_iters=100000)