import ctypes
import numpy as np

training_data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

# Load library
lib = ctypes.CDLL(r"/home/arranthegreat/Desktop/neural network from scratch/v2/logic.so")

# Define argument types
lib.train_model.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double
]

lib.init_random.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
]

lib.sigmoid.argtypes = [ctypes.c_double]  
lib.sigmoid.restype = ctypes.c_double   

lib.seed.restype = None

# Create arrays
w = (ctypes.c_double * 6)()
b = (ctypes.c_double * 3)()

# Initialize
lib.seed()
lib.init_random(w, b)

# Train
lib.train_model(w, b, 50_000, 0.1)

# Convert back to Python
weights = np.array([w[i] for i in range(6)])
biases = np.array([b[i] for i in range(3)])

print("Weights:", weights)
print("Biases:", biases)



# Test
for x1, x2, y in training_data: #looping over training

#Forward pass
    z1 = w[0]*x1 + w[1]*x2 + b[0]
    a1 = lib.sigmoid(z1)

    z2 = w[2]*x1 + w[3]*x2 + b[1]
    a2 = lib.sigmoid(z2)

    z3 = w[4]*a1 + w[5]*a2 + b[2]

#forward pass to output
    y_hat = lib.sigmoid(z3)

# Convert prediction to difinite answer
    answer = "1" if y_hat >= 0.5 else "0"
    print(f"For inputs ({x1}, {x2}), the network predicts: {answer} (raw: {y_hat:.3f})")