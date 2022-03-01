import pandas as pd
import numpy as np
from math import sqrt, erf
import matplotlib.pyplot as plt

def sigmaSquare(X, W, Y):
    n = len(Y)
    residue = X.dot(W) - Y
    return residue.dot(residue) / (n - len(W))

def StdE(sigma_square, X):
    return np.sqrt( (np.linalg.pinv( X.T.dot(X) ) * sigma_square).diagonal() )
        
def calT(se, W):
    return W / se

def GaussianCdf(x, mu, sigma):
    return (1.0 + erf((x - mu) / (sigma * sqrt(2.0)))) / 2.0

def calP(t_values):
    return [2 * GaussianCdf(-np.abs(t), 0, 1) for t in t_values]

def RSS(X, W, Y):
    Y_pred = X.dot(W)
    return (Y - Y_pred).dot(Y - Y_pred)
    
def Rsquare(rss, Y):
    Y_mean = Y.mean()
    return 1 -  rss / (Y - Y_mean).dot(Y - Y_mean)

def linear_regression(X, Y, _lambda):
    n = X.shape[1]
    if _lambda is None:
        W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y.T)
    else:
        W = np.linalg.pinv(_lambda * np.eye(n) + X.T.dot(X)).dot(X.T).dot(Y.T)
        
    # print("coefficient : ", W)
    sigma_square = sigmaSquare(X, W, Y)
    rse = np.sqrt(sigma_square)
    # print('RSE : ', rse)
    stde = StdE(sigma_square, X)
    # print('StdE : ', stde)
    ts = calT(stde, W)
    # print('Ts : ', ts)
    ps = calP(ts)
    # print('Ps : ', ps)
    rss = RSS(X, W, Y)
    # print('RSS : ', rss)
    rsquare = Rsquare(rss, Y)
    # print('R^2 : ', rsquare)
    
    return {'W' : W,
            'RSE' : rse,
            'StdE' : stde,
            'Ts' : ts,
            'Ps' : ps,
            'RSS' : rss,
            'R^2' : rsquare}
    
    
def CV5(X, Y):
    n = X.shape[1]
    _lambda = 10 ** (-5)
    avg_losses = []
    splits = np.linspace(0, X.shape[0], 6).astype(int)
    #print(splits)
    cnt = -5
    
    while cnt <= 3:
        losses = []
        for i in range(5):
            begin, end = splits[i], splits[i+1]
            
            X_train = []
            Y_train = []
            if begin == 0:
                X_train = X[end:]
                Y_train = Y[end:]
            elif end == X.shape[0]:
                X_train = X[:begin]
                Y_train = Y[:begin]
            else:
                X_train = np.vstack((X[:begin], X[end:]))
                Y_train = np.hstack((Y[:begin], Y[end:]))
            
            X_valid = X[begin:end]
            Y_valid = Y[begin:end]
            
            # print('training data dimension : [{}, {}]'.format(X_train.shape, Y_train.shape))
            # print('validation data dimension : [{}, {}]'.format(X_valid.shape, Y_valid.shape))
            W_reg = np.linalg.pinv(_lambda * np.eye(n) + X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train.T)
            rss = RSS(X_valid, W_reg, Y_valid)
            losses.append(rss)
        
        avg_losses.append(np.mean(losses))
        # print('avg loss of lambda : 10^{} in validation is : {}'.format(cnt, avg_losses[-1]))
        # print(avg_losses[-1])
        _lambda *= 10 ** 0.5
        cnt += 0.5
        
    res = np.argmin(avg_losses)*0.5 - 5
    print(f'best lambda is 10^{res}')
    return res, avg_losses

def featureSelect(_lambda=None):
    min_pair = [0, 1]
    min_rss = np.inf
    best_out = None
    cols = X.shape[1]
    
    for i in range(1, cols):
        for j in range(i + 1, cols):
            out = linear_regression(X[:, [i,j]], Y, _lambda)
            rss = out['RSS']
            # print(f'RSS of {i} {j} is {rss}')
            
            if rss < min_rss:
                # print(f'{i} and {j} are selected')
                min_pair = [i, j]
                min_rss = rss
                best_out = out
                
    return min_pair, best_out

def printInfo(out):
    for key, val in out.items():
        print(f'[{key} : [{val}]]')

def plotCV(losses):
    plt.figure()
    x_axis = np.arange(-5, 3.5, 0.5)
    plt.plot(x_axis, losses)
    plt.xlabel("lambda(log10)")
    plt.ylabel("loss")
    plt.title("loss curve of CV")
    
if __name__ == '__main__':
    filename='covid-19.csv'
    df = pd.read_csv(filename)
    X_cols = df.columns.values.tolist()[1:-1]
    Y_col = df.columns.values.tolist()[-1]
    
    X = [[1] * len(df[Y_col])]
    for col in X_cols:
        #print(df[col][:5])
        X.append(df[col])
    
    X = np.array(X).T
    Y = np.array(df[Y_col])
    
    print("For problem a ...")
    out_a  = linear_regression(X, Y, None)
    printInfo(out_a)
    print("----------end-----------")
    
    print("For problem b ...")
    pair_b, out_b = featureSelect()
    print("Feature of {} and {} have smallest RSS".format(*pair_b))
    printInfo(out_b)
    print("----------end-----------")
    
    print("For problem c ...")
    _lambda, losses = CV5(X, Y)
    plotCV(losses)
    pair_c, out_c = featureSelect(_lambda)
    print("Feature of {} and {} have smallest RSS".format(*pair_c))
    printInfo(out_c)
    print("----------end-----------")
    
    print('out of a : [ RSE : {:.3f} ] [R^2 : {:.3f}]'.format(out_a['RSE'], out_a['R^2']) )
    print('out of b : [ RSE : {:.3f} ] [R^2 : {:.3f}]'.format(out_b['RSE'], out_b['R^2']) )
    print('out of c : [ RSE : {:.3f} ] [R^2 : {:.3f}]'.format(out_c['RSE'], out_c['R^2']) )




