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

def getx(x,thetha,delx,dely):
    x=x.reshape((-1, 1))
    thetha = thetha.reshape((-1, 1))
    delx = delx.reshape((-1, 1))
    dely = dely.reshape((-1, 1))
    return x+delx*jnp.cos(thetha)-dely*jnp.sin(thetha) 

def gety(y,thetha,delx,dely):
    y=y.reshape((-1, 1))
    thetha = thetha.reshape((-1, 1))
    delx = delx.reshape((-1, 1))
    dely = dely.reshape((-1, 1))
    return y+dely*jnp.cos(thetha)+delx*jnp.sin(thetha) 

def getth(thetha,delt):
#     dt = dt.reshape((-1,1))
    thetha = thetha.reshape((-1, 1))
    delt = delt.reshape((-1, 1))
    return thetha+delt

# xx -> vertex
def get_fX(vertex,edges):
    x,y,theta=jnp.reshape(vertex,(3,vertex.shape[0]//3))
    x,y, theta = x.reshape((-1,1)), y.reshape((-1,1)), theta.reshape((-1,1))
    x, y, theta= jnp.array(x), jnp.array(y), jnp.array(theta)
    i, j, delx, dely, delthetha, ocl=edges[0], edges[1], edges[2], edges[3], edges[4], len(x)
    #odometry constraint
    odocx, odocy, odocth=getx(x[:-1],theta[:-1],delx[:ocl-1],dely[:ocl-1])-x[1:], gety(y[:-1],theta[:-1],delx[:ocl-1],dely[:ocl-1])-y[1:] , getth(theta[:-1],delthetha[:ocl-1])-theta[1:]
    #loop constrain
    ocl-=1
    iloop, jloop=i[ocl:], j[ocl:]
    loopcx=getx(x[iloop],theta[iloop],delx[ocl:],dely[ocl:])-x[jloop]
    loopcy=gety(y[iloop],theta[iloop],delx[ocl:],dely[ocl:])-y[jloop]
    loocpctx=getth(theta[iloop],delthetha[jloop])-theta[jloop]
    #initial constrains
    initcx, initcy, initth =x[0]-(-5), y[0]-(-8),theta[0]-(0)
    initcx, initcy, initth=initcx.reshape((-1,1)), initcy.reshape((-1,1)), initth.reshape((-1,1))
    final = jnp.concatenate((odocx,odocy,odocth,loopcx,loopcy,loocpctx,initcx,initcy,initth)).reshape(-1)
    return final

def CreateOmega(odo=1, loop=1,init=10):
    omega = jnp.zeros((119*3 + 20*3+3, 119*3 + 20*3+3))
    xodo = odo
    yodo = odo
    todo = odo
    xloop = loop
    yloop = loop
    tloop = loop
    
    for i in range(119):
        #omega = jax.ops.index_update(omega, jax.ops.index[i, i], xodo)
        #x.at[idx].set(y)
        omega = omega.at[i, i].set(xodo)
        
    for i in range(119, 2*119):
        #omega = jax.ops.index_update(omega, jax.ops.index[i, i], yodo)
        omega = omega.at[i, i].set(yodo)

    for i in range(2*119, 3*119):
        #omega = jax.ops.index_update(omega, jax.ops.index[i, i], todo)
        omega = omega.at[i, i].set(todo)

    for i in range(3*119, 3*119 + 20):
        #omega = jax.ops.index_update(omega, jax.ops.index[i, i], xloop)
        omega = omega.at[i, i].set(xloop)

    for i in range(3*119 + 20, 3*119 + 2*20):
        #omega = jax.ops.index_update(omega, jax.ops.index[i, i], yloop)
        omega = omega.at[i, i].set(yloop)

    for i in range(3*119 + 2*20, 3*119 + 3*20):
        #omega = jax.ops.index_update(omega, jax.ops.index[i, i], tloop)
        omega = omega.at[i, i].set(tloop)

    for i in range(3*119 + 3*20, 3*119 + 3*20+3):
        #omega = jax.ops.index_update(omega, jax.ops.index[i, i], init)
        omega = omega.at[i, i].set(init)

    return omega 

def jacobian(thetha,edges):
#     print("Check point 3")
    i, j, delx, dely, delth=edges[0], edges[1], edges[2], edges[3], edges[4]
    odocl=thetha.shape[0]
    loop1, loop2=i[odocl-1:], j[odocl-1:]
    jaco=[]
    x = 0 
    while x < odocl:
        Delft=jnp.zeros(420).tolist()
        if x!=0 and x!=odocl-1:
            Delft[x]=1
            Delft[x-1]=-1
        elif x==odocl-1:
            Delft[118]=-1
        elif x ==0:
            Delft[0]=1
            Delft[-3]=1
            
#         print("Check point 3")
        if x!=0 and x!=odocl-1: 
            Delft[x]=1
            Delft[x-1]=-1
        elif x==odocl-1:
            Delft[118]=-1
        elif x == 0:
            Delft[0]=1
            Delft[-3]=1
            
        i = 0 
        while i < 20 :
            if loop1[i]==x:
                Delft[357+i]=1    
            i+=1
        j = 0 
        while j < 20:
            if loop2[j]==x:
                Delft[357+j]=-1
            j+=1
        
        jaco.append(Delft)
        x+=1
        
    #Y
#     print("Check point 4")
    y = 0 
    while y < odocl :
#         print(y, odocl)
        delfy=jnp.zeros(420).tolist()
        if y!=0 and  y!=odocl-1:
            delfy[119+y]=1
            delfy[119+y-1]=-1
            
        elif y==odocl-1:
            delfy[237]=-1
        else:
            delfy[119+0]=1
            delfy[-2]=1
            
        i = 0 
        while i < 20:
            if loop1[i]==y:
                delfy[357+20+i]=1
            i+=1    
        j = 0
        while j < 20:
            if loop2[j]==y:
                delfy[357+20+j]=-1
            j+=1
        jaco.append(delfy)
        y+=1
        
    #thetha
    t=0
    while t < odocl :
        delft=jnp.zeros(420).tolist()
        
        if  t==119:
            delft[237+119]=-1
        elif t==0:
            delft[-1]=1
            delft[238+t]=1
        else:
            delft[237+t]=-1
            delft[238+t]=1

        if t<119:
#             print(delx.shape)
#             print(dely.shape)
#             print(thetha.shape)
            delft[t] = (-delx[t]*jnp.sin(thetha[t])-dely[t]*jnp.cos(thetha[t]))
            delft[119+t] = (-dely[t]*jnp.sin(thetha[t])+delx[t]*jnp.cos(thetha[t]))
        
        i = 0 
        while i < 20:
            if loop1[i]==t:
                delft[357+i] = (-delx[119+i]*jnp.sin(thetha[t])-dely[119+i]*jnp.cos(thetha[t]))
                delft[357+20+i] = (-dely[119+i]*jnp.sin(thetha[t])+delx[119+i]*jnp.cos(thetha[t]))
                delft[357+40+i] = 1
            i+=1
        
        j=0
        while j < 20 :
            if loop2[j]==t:
                delft[357+40+j] = -1
            j+=1
        
        jaco.append(delft)
        t+=1
        
    print("Check point 5")
    final = jnp.array(jaco).T
    return final

plt.style.use('fivethirtyeight')
def frobNorm(P1, P2, str1="mat1", str2="mat2"):
    jnp.set_printoptions(suppress=True)
    val = jnp.linalg.norm(P1 - P2, 'fro')
    print(f"Frobenius norm between {str1} and {str2} is: {val}")        
    
        
def levenMarq(x,y,theta,edges):
    cost_itr, x_itr =[], []
    it = 0
    lamda=0.1
    w=CreateOmega(2,500,1000)
    max_itr = 50
    tolerance = 0.000001
    while it < max_itr:
        vertices=jnp.concatenate((x,y,theta)).reshape(-1)
        f=get_fX(vertices,edges)
        lambf = lambda aa : get_fX(aa,edges)
        j=jacfwd(lambf)(vertices)
#         print(j.shape)
#         if it == 0 :
#             jj=analitical_jac(theta,edges)
        H, b = j.T@w@j + lamda*jnp.eye(360), j.T@w.T@f
        delx = -jnp.linalg.inv(H)@b
#         delx = jnp.reshape(delx,(3,delx.shape[0]//3))
        dX,dY,dT = jnp.reshape(delx,(3,len(delx)//3))
        x+=dX
        y+=dY
        theta+=dT
        costn = 0.5*f.T@w@f
        cost_itr.append(costn)
        l = len(cost_itr)
        print("err",it+1,"=",cost_itr[l-1])
        x_itr.append([x,y,theta])
        if it!=0:
            l = len(cost_itr)
            if cost_itr[l-1]>cost_itr[l-2]:
                lamda/=10
            else:
                lamda*=10
                
            if cost_itr[l-2] - cost_itr[l-1] < tolerance:
                break
        it+=1
        
    return cost_itr,x_itr

X = jnp.array(X)
Y = jnp.array(Y)
THETA = jnp.array(THETA)
cost,x=levenMarq(X, Y, THETA,edges) 
l = len(x)
makeg2o(x[l-1][0],x[l-1][1], x[l-1][2],g2ofile="opt.g2o")
makeg2o(x[l-1][0],x[l-1][1], x[l-1][2],g2ofile="opt-backup.g2o")




def final_output(vertex_gt, xn, yn, tn):  
    ax = plt.subplot()
    c = 0.25
    xgt, ygt=vertex_gt[0], vertex_gt[1]
    tg = vertex_gt[2]


    i = 0 
    while i < len(tg):
        pls, ply = c*math.cos(tg[i]) + xgt[i],  c*math.sin(tg[i]) + ygt[i]
        if i == 0:
            plt.plot([xgt[i], pls], [ygt[i], ply], 'r->', label="Ground Truth")
        else :
             plt.plot([xgt[i], pls], [ygt[i], ply], 'r->')
        i+=1

    
    i = 0
    while i < len(tn):
        pls, ply = c*math.cos(tn[i]) + xn[i],  c*math.sin(tn[i]) + yn[i]
        if i == 0 :
            plt.plot([xn[i], pls], [yn[i], ply], 'g->', label="Final Trajectory")
        else :
             plt.plot([xn[i], pls], [yn[i], ply], 'g->')
        i+=1
        

    i = 0
    while i < len(THETA):
        pls, ply = c*math.cos(THETA[i]) + X[i],  c*math.sin(THETA[i]) + Y[i]
        if i == 0 :
            plt.plot([X[i], pls], [Y[i], ply], 'b->', label="Initial Trajectory")
        else :
            plt.plot([X[i], pls], [Y[i], ply], 'b->')
        i+=1
    
    plt.legend()
    plt.show()
    

vertexgt = readVertex('./dataset/gt.txt')
for i in range(1, l+1):
    print("Iterations : ",str(i)," Loss : ",cost[i-1])
    xn,yn,tn=x[i-1][0],x[i-1][1],x[i-1][2]
    final_output(vertexgt, xn,yn,tn)