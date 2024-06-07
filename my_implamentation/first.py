import matplotlib.pyplot as plt
import math
import os
import jax.numpy as jnp #see supplementary notebook to see how to use this
from jax import jacfwd
import jax
# If you're `importing numpy as np` for debugging purposes, 
# while submitting, please remove 'import numpy' and replace all np's with jnp's.(more in supplementary notebook)

##############################################################################
# TODO: Code for Section 2.1                                                 #
def readVertex(fileName):
    f = open(fileName, 'r')
    A = f.readlines()
    f.close()
    
    x_arr = []
    y_arr = []
    theta_arr = []

    for line in A:
        if "VERTEX_SE2" in line:
            (ver, ind, x, y, theta) = line.split()
            x_arr.append(float(x))
            y_arr.append(float(y))
            theta_arr.append(float(theta.rstrip('\n')))

    return jnp.array([x_arr, y_arr, theta_arr])

def readEdge(fileName):
    f = open(fileName, 'r')
    A = f.readlines()
    f.close()

    ind1_arr = []
    ind2_arr = []
    del_x = []
    del_y = []
    del_theta = []

    for line in A:
        if "EDGE_SE2" in line:
            (edge, ind1, ind2, dx, dy, dtheta, _, _, _, _, _, _) = line.split()
            ind1_arr.append(int(ind1))
            ind2_arr.append(int(ind2))
            del_x.append(float(dx))
            del_y.append(float(dy))
            del_theta.append(float(dtheta))

    return (jnp.array( ind1_arr), jnp.array(ind2_arr), jnp.array(del_x), jnp.array(del_y), jnp.array(del_theta))

def draw(X, Y, THETA):
    ax = plt.subplot(111)
    ax.plot(X, Y, 'ro')
    plt.plot(X, Y, 'c-')

    for i in range(len(THETA)):
        x2 = 0.25*math.cos(THETA[i]) + X[i]
        y2 = 0.25*math.sin(THETA[i]) + Y[i]
        plt.plot([X[i], x2], [Y[i], y2], 'g->')

    plt.show()

def makeg2o(x, y, z, g2ofile):
    f = open(g2ofile, "w")
    for i in range(len(x)):
        f.write("VERTEX_SE2"+" " + str(i) + " " + str(x[i]) +" "+ str(y[i]) +" "+ str(z[i])+"\n")
    
    f2 = open("dataset/edges.txt", "r")
    lines = f2.readlines()
    for i in range(1, len(lines)):
        if lines[i][0:3] != "FIX":
            f.write(lines[i])

x, y, theta = readVertex('dataset/gt.txt')
_, _, delx, dely, delt = readEdge('dataset/edges.txt')
edges = readEdge('dataset/edges.txt')
X = []
Y = []
THETA = []
X.append(x[0])
Y.append(y[0])
THETA.append(theta[0])
for i in range(1, x.shape[0]):
    X.append(X[i-1] + delx[i-1]* jnp.cos(THETA[i-1]) - dely[i-1]*jnp.sin(THETA[i-1]))
    Y.append(Y[i-1] + dely[i-1]* jnp.cos(THETA[i-1]) + delx[i-1]*jnp.sin(THETA[i-1]))
    THETA.append(THETA[i-1] + delt[i-1])

X = jnp.array(X)
Y = jnp.array(Y)
THETA = jnp.array(THETA)
            
vertex = readVertex('./dataset/edges.txt')
edge = readEdge('./dataset/edges.txt')
    
draw(X,Y,THETA)
print(X.shape)
makeg2o(X,Y,THETA,g2ofile="edges-poses_backup.g2o")
makeg2o(X,Y,THETA,g2ofile="edges-poses.g2o")

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
