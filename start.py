import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from cvxopt import matrix, solvers
import time


def discriminate(x):
    if x>0:
        return 1
    else:
        return 0
discriminate_vec = np.vectorize(discriminate)


def spectrum_kernel(X, Y, **kwargs):
    k = kwargs["k"]
    m1 = X.shape[0]
    m2 = Y.shape[0]
    list_of_dictionaries_X = []
    for chaine in X["seq"]:
        dico = {}
        for i in range(len(chaine)-k+1):
            motif = chaine[i:i+k]
            if not(motif in dico.keys()):
                dico[motif]=1
            else:
                dico[motif]+=1
        list_of_dictionaries_X.append(dico)
    list_of_dictionaries_Y = []
    for chaine in Y["seq"]:
        dico = {}
        for i in range(len(chaine)-k+1):
            motif = chaine[i:i+k]
            if not(motif in dico.keys()):
                dico[motif]=1
            else:
                dico[motif]+=1
        list_of_dictionaries_Y.append(dico)
    mat = np.zeros((m1,m2))
    for i in range(m1):
        for j in range(m2):
            for key in set(list_of_dictionaries_X[i].keys())&set(list_of_dictionaries_Y[j].keys()):
                mat[i][j]+= list_of_dictionaries_X[i][key]*list_of_dictionaries_Y[j][key]
    return mat


def build_1neighborhood(chain):
    chain = list(chain)
    lili = []
    for i in range(len(chain)):
        temp = chain.copy()
        for char in ['A', 'G', 'C', 'T']:
            temp[i]=char
            lili.append("".join(temp))
    return lili


def mismatch_kernel(X, Y, **kwargs):
    k = kwargs["k"]
    m1 = X.shape[0]
    m2 = Y.shape[0]
    list_of_dictionaries_X = []
    for chaine in X["seq"]:
        dico = {}
        for i in range(len(chaine)-k+1):
            motif = chaine[i:i+k]
            for neighbor in build_1neighborhood(motif):
                if not(neighbor in dico.keys()):
                    dico[neighbor]=1
                else:
                    dico[neighbor]+=1
        list_of_dictionaries_X.append(dico)
    list_of_dictionaries_Y = []
    for chaine in Y["seq"]:
        dico = {}
        for i in range(len(chaine)-k+1):
            motif = chaine[i:i+k]
            for neighbor in build_1neighborhood(motif):
                if not(neighbor in dico.keys()):
                    dico[neighbor]=1
                else:
                    dico[neighbor]+=1
        list_of_dictionaries_Y.append(dico)
    mat = np.zeros((m1,m2))
    for i in range(m1):
        for j in range(m2):
            for key in set(list_of_dictionaries_X[i].keys())&set(list_of_dictionaries_Y[j].keys()):
                mat[i][j]+= list_of_dictionaries_X[i][key]*list_of_dictionaries_Y[j][key]
    return mat

def run_SVM(K, name, lambd_SVM = 0.001, **kwargs):
    start_time = time.time()
    # Main code:
    export_header = True
    for dataset in range(3):
        start_step = time.time()
        # Load dataset:
        Xtr = pd.read_csv("data/Xtr"+str(dataset)+".csv", sep=",")
        Xte = pd.read_csv("data/Xte"+str(dataset)+".csv", sep=",")
        Ytr = pd.read_csv("data/Ytr"+str(dataset)+".csv").to_numpy()[:,1]
        Ytr = Ytr.reshape((Ytr.shape[0],1))
        Ytr_reshaped = 2*Ytr.reshape(Ytr.shape[0])-1
        # Build Gram matrix:
        print('Building Kernel')
        K_train = K(Xtr, Xtr, **kwargs)
        print("K_train successfully built")
        # Set problem for QP solver:
        ## Number of samples:
        n  = K_train.shape[0]
        ## Initial guess for mu:
        x0 = np.zeros(2*n)
        ## Matrix P:
        P = matrix((1/(2*lambd_SVM))*np.diag(Ytr_reshaped)@K_train@np.diag(Ytr_reshaped))
        ## Vector Q:
        q = -np.ones(n)
        q = matrix(q)
        ## Matrix G:
        G = np.zeros((2*n,n))
        G[:n] = -np.eye(n)
        G[n:] = np.eye(n)
        G = matrix(G)
        ## Vector h:
        h = np.zeros(2*n)
        h[n:]+= 1/n
        h = matrix(h)
        ## Call QP solver on dual SVM problem:
        resu_SVM = solvers.qp(P, q, G, h)
        mu = np.array(resu_SVM["x"])
        ## Recover primal solution:
        alpha = np.diag(Ytr_reshaped)@mu/(2*lambd_SVM)
        ## Build train prediction
        y_pred_SVM_raw = K_train@alpha
        y_pred_SVM = discriminate_vec(y_pred_SVM_raw)
        ## Show training accuracy:
        print('Training Accuracy for training set '+str(dataset)+' :',accuracy_score(y_pred_SVM, Ytr))
        # Test:
        ## Build Gram matrix:
        K_test = K(Xte, Xtr, **kwargs)
        print("K_test successfully built")
        ## Build test prediction:
        y_pred_SVM_test_raw = K_test@alpha
        y_pred_SVM_test = discriminate_vec(y_pred_SVM_test_raw)
        ## Build dataframe and save as .csv:
        start_index = dataset*1000
        dico = {"Id": range(start_index, start_index+Xte.shape[0]), "Bound": y_pred_SVM_test.reshape(Xte.shape[0]) }
        pred_SVM_test = pd.DataFrame(dico)
        pred_SVM_test.to_csv(name+".csv", index=False, sep=",", header=export_header, mode='a')
        export_header = False
        end_step = time.time()
        print("time on dataset {}: {}".format(dataset,np.round(end_step-start_step,2)))
    end_time = time.time()
    print("total time: {}".format(np.round(end_time-start_time,2)))


if __name__ == "__main__":
    run_SVM(mismatch_kernel, "mismatch_kernel_SVM", lambd_SVM = 0.01, **{"k":10})
