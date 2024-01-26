import numpy as np
import matplotlib.pyplot as plt

def plot(X,Y,n,label=None,color=None,linewidth=None,linestyle=None,alpha=1):
    """
    Plot a line graph with moving average.

    Parameters:
    - X: array-like, x-coordinates of the data points.
    - Y: array-like, y-coordinates of the data points.
    - n: int, width of the moving average window.
    Returns:
    None
    """
    xavg=np.average(X[0:n]); yavg=np.average(Y[0:n])
    XX=[xavg]; YY=[yavg]
    for i in range(n,len(X),1):
        xavg = xavg + (X[i]-X[i-n])/n
        yavg = yavg + (Y[i]-Y[i-n])/n
        XX.append(xavg)
        YY.append(yavg)
    plt.plot(XX,YY,label=label,color=color,linewidth=linewidth,linestyle=linestyle,alpha=alpha)
    
def axplot(axis,X,Y,n,label=None,color=None,linewidth=None,linestyle=None,alpha=1):
    '''same as above but add the plot to an axis'''
    xavg=np.average(X[0:n]); yavg=np.average(Y[0:n])
    XX=[xavg]; YY=[yavg]
    for i in range(n,len(X),1):
        xavg = xavg + (X[i]-X[i-n])/n
        yavg = yavg + (Y[i]-Y[i-n])/n
        XX.append(xavg)
        YY.append(yavg)
    axis.plot(XX,YY,label=label,color=color,linewidth=linewidth,linestyle=linestyle,alpha=alpha)