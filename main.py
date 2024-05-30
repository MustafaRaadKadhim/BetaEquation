import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pickle
from scipy.spatial.distance import cdist
from sklearn import metrics
import time
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
def T2D(Data_):
    return PCA(n_components=2, random_state=2).fit_transform(Data_)
def DataLoader(ds):
    if (ds == 0):
        ################ 'Iris' ################
        D = datasets.load_iris()
        Target = D.target
        Data = D.data
        gamma = np.array([45, 54, 115])
        alpha = 2
    elif (ds == 1):
        ################ 'Wine' ################
        D = datasets.load_wine()
        Target = D.target
        Data = D.data
        gamma = np.array([40, 115, 170])
        alpha = 2
    elif (ds == 2):
        ################ 'Depression' ################
        Data = load_object("./Datasets/Depression/Depression_Data.pickle")
        Target = load_object("./Datasets/Depression/Depression_Target.pickle")
        gamma = np.array([13658, 10])
        alpha = 0
    elif (ds == 3):
        ################ 'AdvDS' ################
        Data = load_object("./Datasets/Traffic/AdvDS_Data.pickle")
        Target = load_object("./Datasets/Traffic/AdvDS_Target.pickle")
        gamma = np.array([500, 3500])
        alpha = 1
    elif (ds == 4):
        ################ 'OLDS' ################
        Data = np.array([[-4.90927458, 3.13470696],
                         [0.79686721, 1.7768663],
                         [5.92779522, -1.3423817],
                         [1.26612626, 2.13956938],
                         [-0.37098296, 2.38916156],
                         [0.60225064, 1.02795307],
                         [-5.55336572, 8.58606835],
                         [8.44669983, 5.34653779],
                         [5.42254826, 12.71082177],
                         [-1.71084169, -3.56943013],
                         [1.43798457, -0.44301906],
                         [-12.29833245, 3.44992937],
                         [-0.45662389, 10.09710602],
                         [-8.80086533, -4.9028267],
                         [-10.40252187, -2.91553245],
                         [-1.05568039, -0.58601471],
                         [-8.20073569, -11.87886376],
                         [-8.02608697, -0.93360151],
                         [-9.30083092, -3.3484139],
                         [-5.96325902, -8.49915308],
                         [-3.38234975, 6.58705245],
                         [12.08810612, -1.37858587],
                         [4.87569088, 4.98288988],
                         [12.9248563, 12.18985098],
                         [4.84451357, -1.64649844],
                         [15.29665475, 2.33435836],
                         [3.4895276, 8.07443086],
                         [13.09967665, 14.59455088],
                         [13.39987659, 13.52281506],
                         [6.87186569, 16.79908694]])
        Target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        gamma = np.array([5, 14, 25])
        alpha = 2

    return Data, Target, gamma, alpha
def DRAWCIRCLEOBJECTS(Data_, Labels, Itrain, figsize_, IsDraw):
    global c
    if (IsDraw == True):
        Data_ = Data_.copy()
        plt.figure(figsize=figsize_)
        colors = ["red", "green", "black"]
        colors = np.array(colors)
        numbers = np.array(range(Data_.shape[0])) + 1
        cccccc = 0
        for i in np.unique(Labels):
            xx = Data_[np.where(Labels == i)[0], 0]
            yy = Data_[np.where(Labels == i)[0], 1]
            plt.scatter(xx, yy, s=2000, marker='.', facecolors='none', edgecolors=colors[cccccc])
            cccccc += 1
        plt.legend(["$K_1$", "$K_2$", "$K_3$"], ncol=3, loc="upper left")
        for xi, yi, num in zip(Data_[:, 0], Data_[:, 1], numbers):
            plt.text(xi, yi, str(num), ha='center', va='center', c='black')
        plt.grid(True)
        plt.tight_layout()
        fn = "./Outcomes/" + str(c) + ".png"
        plt.savefig(fn)
        c += 1
        return fn
    else:
        return ''


if __name__ == "__main__":
    """
    DataLoader(0): Load Iris 
    DataLoader(1): Load Wine 
    DataLoader(2): Load Depression 
    DataLoader(3): Load AdvDS 
    DataLoader(4): Load OLDS

    varepsilon: Number of times the data will be separated

    c: # Counter: Don't change

    IsDraw: If you want to display the results then change it to True

    """
    c = 0
    figsize_ = (8, 8)  # Prefered Figure size
    varepsilon = 30
    IsDraw = False

    for ds in range(5):
        TTTT = time.time()

        ################## Step 1 ##################
        # Load the data
        Data, y, gamma, alpha = DataLoader(ds)
        # Remove negative values
        X = 1 + abs(Data.min()) + Data.copy()
        # Use to draw and the outcomes will be shown in "Outcomes" file
        DRAWCIRCLEOBJECTS(X, y, gamma, figsize_, IsDraw)

        ################## Step 2 ##################
        for iter_ in range(1, varepsilon):
            tau_old = X.copy()
            rho = cdist(tau_old, tau_old)
            rho_sorted = np.argsort(rho, axis=1)
            psi = tau_old[rho_sorted[gamma[alpha]]]
            order = np.argsort(cdist(X[gamma], np.min(X[gamma], axis=0)[np.newaxis, :]), axis=0).reshape(-1, )

            ################## Step 3 ##################
            for i in order:
                f = (psi + X[gamma[i]]) / np.mean(np.array(X), axis=0)
                X = X + (f)
            X = X / tau_old

            ################## Step 4 ##################
            cl = np.argsort(cdist(X, X[gamma]), axis=1)[:, 0]
            print("Iteration: ", iter_, " -- SH: ", np.round(metrics.silhouette_score(X, y), 4), " -- ACC: ",
                  np.round(metrics.adjusted_rand_score(cl, y), 4))

        elapsed = time.time() - TTTT
        print("===================================================================================================")
        print(" $$$$$$$$$$$$$$$$ Elapsed time: %f seconds.\n" % elapsed)
        print("===================================================================================================")

        DRAWCIRCLEOBJECTS(X, y, gamma, figsize_, IsDraw)


