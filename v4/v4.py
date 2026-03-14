#imporvements:
""" 
Dyanamic training data in C code
Small ajustments to incoperate things more smoothly

"""
try:
    import pandas as pd
    import ctypes
    import numpy as np
except ImportError:
    exit("Error impoting Libs")

#defining a path varibae
data_path = r"/home/arranthegreat/Desktop/PYTHON/neural network from scratch/v4/training_data.csv"
logic_path = r"/home/arranthegreat/Desktop/PYTHON/neural network from scratch/v4/logic.so"

try:
#   trainign data import / clean
    training_data = pd.read_csv(data_path,header=None) #honeslt can not remeber if i even need to state that
except IOError:
    exit("Error Loading Training data")


#convert to np array for eaze of slicing
#MAKEING sure data type is float for all (helpfull later)
training_data = np.array(training_data, dtype=float)
target = training_data[-1]
values = training_data[:-1]

#print("Values:", values)
#print("Target:", target)

class network_xor_model:
    def __init__(self, lib = logic_path):
        try:
            self.lib = ctypes.CDLL(lib)

        #its own wierd OSError message pmo a bit
        except OSError as e:
            print(f"Failed to load library: {e}")
        
        #defineing arg types for C functions
        self.lib.train_model.argtypes=[
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            #expects file path as arg
            ctypes.c_char_p
        ]
        
        self.lib.init_random.argtypes=[
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]

        self.lib.sigmoid.argtypes=[ctypes.c_double]
        self.lib.sigmoid.restype=ctypes.c_double

        #define results type as NONE since we dont expect one
        self.lib.seed.restype= None
        
        #allocating w (6)and b (3)
        self.w=(ctypes.c_double*6)()
        self.b=(ctypes.c_double *3)()
        
        #initing rand w
        self.lib.seed()
        self.lib.init_random(self.w, self.b)
    
    def train(self, epochs=50_000, lr=0.1):
        #training the XORnet 
        self.lib.train_model(self.w, self.b, epochs, lr, data_path.encode()) #we have to ecode the path so C can actually use it
    
    def pred_raw(self, x1, x2):
        #Getting the Raw output of the network
        a1 =self.lib.sigmoid(self.w[0]*x1 + self.w[1]*x2 + self.b[0])
        a2 = self.lib.sigmoid(self.w[2]*x1 +self.w[3]* x2 + self.b[1])
        y_hat= self.lib.sigmoid(self.w[4]*a1 +self.w[5]*a2+ self.b[2])
        return y_hat
    
    def pred(self, x1, x2):
        #retrun full prediction (1 or 0)
        y_hat = self.pred_raw(x1,x2)
        return 1 if y_hat >= 0.5 else 0, y_hat
    
    def review(self, dataset):
        #getting stats of the predictions
        correct = 0
        mse = 0
        results = []
        for x1, x2, y in dataset:
            #getting the return of the predict function
            pred, raw = self.pred(x1, x2)

            #appending that data to the results list
            results.append((x1, x2, y, pred, raw))

            #compares the predicted output (pred) with the correct(y)
            #if they are same it returns / sets correct to true else false
            correct += (pred == y)

            #mean square error (used in ML alot with linear models) (math i dont get underneath me)
            mse += ((raw - y)**2) / len(dataset)
            #math explained:
                #takes away the raw value from the correct value and finds the difference and sqaures it
                #then divedes by len of dataset to get the mean
        
        #gets the accuracy percentage (as a decimal)
        accuracy = correct / len(dataset)

        #returning values
        return results, accuracy, mse
    


 #######################   
#SHOWCASE
 ######################
#make a new XOR network
network=network_xor_model()

#train network
print("Training XOR network...")
network.train(epochs=50_000, lr=0.1)

# evaulate results
results,accuracy,mse=network.review(training_data)
print("Predictions:")
for x1, x2, y,pred, raw in results:
    #raw results
    print(f"Input: ({x1},{x2}) --- True: {y} --- Pred: {pred} --- Raw: {raw:.3f}")

#final
print(f"\nAccuracy: {accuracy*100:.1f}% --- MSE: {mse:.4f}")