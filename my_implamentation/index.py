#import matplot to create position graph
import matplotlib.pyplot as plt
import math
import os

#import jax to use CPU for high levels of computaion
import jax.numpy as jnp 
from jax import jacfwd
import jax

##############################################################################
#function for reading nodes/vertex from dataset and creating jax arrays from them
def readVertex(fileName):
    #reading dataset
    f = open(fileName, 'r')
    A = f.readlines()
    f.close()
    
    #creating array variables
    x_arr = []
    y_arr = []
    theta_arr = []

    #if data in set is node/vertex then append potional and rotational data to relevent arrays
    for line in A:
        if "VERTEX_SE2" in line:
            (ver, ind, x, y, theta) = line.split()
            x_arr.append(float(x))
            y_arr.append(float(y))
            theta_arr.append(float(theta.rstrip('\n')))

    #return final array
    return jnp.array([x_arr, y_arr, theta_arr])

#function to read edges/constraints from data set
def readEdge(fileName):

    #readind data set
    f = open(fileName, 'r')
    A = f.readlines()
    f.close()

    #creating arrays variables
    ind1_arr = []
    ind2_arr = []
    del_x = []
    del_y = []
    del_theta = []

    #if data in set is edge/constraint then append potional and rotational data to relevent arrays
    for line in A:
        if "EDGE_SE2" in line:
            (edge, ind1, ind2, dx, dy, dtheta, _, _, _, _, _, _) = line.split()
            ind1_arr.append(int(ind1))
            ind2_arr.append(int(ind2))
            del_x.append(float(dx))
            del_y.append(float(dy))
            del_theta.append(float(dtheta))

     #return final arrays
    return (jnp.array( ind1_arr), jnp.array(ind2_arr), jnp.array(del_x), jnp.array(del_y), jnp.array(del_theta))

#function to use matpot to draw position graph
def draw(X, Y, THETA):

    #specifing graph to have one collum, one row, and to be the first index subplot
    ax = plt.subplot(111)

    #ploting red cirles of X and Y positions
    ax.plot(X, Y, 'ro')
    #ploting green lines of X and Y positions
    plt.plot(X, Y, 'c-')

    #reasigning positions on post graph using theta
    for i in range(len(THETA)):

        #creating new position variable, scalling them down to 0.25 and adding it to original coordinate
        x2 = 0.25*math.cos(THETA[i]) + X[i]
        y2 = 0.25*math.sin(THETA[i]) + Y[i]

        #reasigns position vectors using new positions and representing them with a green triangle
        plt.plot([X[i], x2], [Y[i], y2], 'g->')

    #creating a window to show the position graph
    plt.show()

#function for writing vertex file and reading edges file
def makeg2o(x, y, z, g2ofile):
    f = open(g2ofile, "w")

    #writing number of lines based on array input
    for i in range(len(x)):

        #writing lines containing index,x,y,z data
        f.write("VERTEX_SE2"+" " + str(i) + " " + str(x[i]) +" "+ str(y[i]) +" "+ str(z[i])+"\n")
    
    f2 = open("dataset/edges.txt", "r")
    lines = f2.readlines()
    for i in range(1, len(lines)):

        #if the first three characters are the string "FIX" write line to file f
        if lines[i][0:3] != "FIX":
            f.write(lines[i])

#defining position and rotation data based on input data
x, y, theta = readVertex('dataset/gt.txt')

#defining position and strength of edges based on input data
_, _, delx, dely, delt = readEdge('dataset/edges.txt')
edges = readEdge('dataset/edges.txt')

#creating global array variables
X = []
Y = []
THETA = []

#inputing first figures of array data into variables
X.append(x[0])
Y.append(y[0])
THETA.append(theta[0])

for i in range(1, x.shape[0]):

    #appending previous X figure + dirivitive of x in respect to theta
    X.append(X[i-1] + delx[i-1]* jnp.cos(THETA[i-1]) - dely[i-1]*jnp.sin(THETA[i-1]))

    #appending previous Y figure + dirivitive of y in respect to theta
    Y.append(Y[i-1] + dely[i-1]* jnp.cos(THETA[i-1]) + delx[i-1]*jnp.sin(THETA[i-1]))
    THETA.append(THETA[i-1] + delt[i-1])

#turning arrays into usable jax arrays
X = jnp.array(X)
Y = jnp.array(Y)
THETA = jnp.array(THETA)

#defining origin point from datasets
vertex = readVertex('./dataset/edges.txt')
edge = readEdge('./dataset/edges.txt')
    
#drawing position graph using matlabs
draw(X,Y,THETA)
print(X.shape)

#storing backup of edge potions
makeg2o(X,Y,THETA,g2ofile="edges-poses_backup.g2o")

#storing of edge potions
makeg2o(X,Y,THETA,g2ofile="edges-poses.g2o")

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

#reshaping the x position array using numpy (number python) to alter the array
def getx(x,thetha,delx,dely):

    #reshaping each array into a 2D array with one collum and as many rows as necessary
    x=x.reshape((-1, 1))
    thetha = thetha.reshape((-1, 1))
    delx = delx.reshape((-1, 1))
    dely = dely.reshape((-1, 1))

    #returning the dirivitive of x in respect to theta
    return x+delx*jnp.cos(thetha)-dely*jnp.sin(thetha) 

#reshaping the y position array using numpy (number python) to alter the array
def gety(y,thetha,delx,dely):

    #reshaping each array into a 2D array with one collum and as many rows as necessary
    y=y.reshape((-1, 1))
    thetha = thetha.reshape((-1, 1))
    delx = delx.reshape((-1, 1))
    dely = dely.reshape((-1, 1))

    #returning the dirivitive of y in respect to theta
    return y+dely*jnp.cos(thetha)+delx*jnp.sin(thetha) 

#reshaping the theta rotation array using numpy (number python) to alter the array
def getth(thetha,delt):

    #reshaping each array into a 2D array with one collum and as many rows as necessary
    thetha = thetha.reshape((-1, 1))
    delt = delt.reshape((-1, 1))

    #returning theta + the change in theta
    return thetha+delt

# xx -> vertex
def get_fX(vertex,edges):

    #reshapes the vertex array into three poition and rotational arrays
    x,y,theta=jnp.reshape(vertex,(3,vertex.shape[0]//3))

    #reshaping each array into a 2D array with one collum and as many rows as necessary
    x,y, theta = x.reshape((-1,1)), y.reshape((-1,1)), theta.reshape((-1,1))

    #ensures that arrays are jax arrays to be used by CPU
    x, y, theta= jnp.array(x), jnp.array(y), jnp.array(theta)

    #unpacks edges array into individual components
    i, j, delx, dely, delthetha, ocl=edges[0], edges[1], edges[2], edges[3], edges[4], len(x)
    
    #calculate odometry by comparing expected positions and orientations
    odocx, odocy, odocth=getx(x[:-1],theta[:-1],delx[:ocl-1],dely[:ocl-1])-x[1:], gety(y[:-1],theta[:-1],delx[:ocl-1],dely[:ocl-1])-y[1:] , getth(theta[:-1],delthetha[:ocl-1])-theta[1:]
    
    #Calucate the loop colsure constraints using get{array} function
    ocl-=1
    iloop, jloop=i[ocl:], j[ocl:]
    loopcx=getx(x[iloop],theta[iloop],delx[ocl:],dely[ocl:])-x[jloop]
    loopcy=gety(y[iloop],theta[iloop],delx[ocl:],dely[ocl:])-y[jloop]
    loocpctx=getth(theta[iloop],delthetha[jloop])-theta[jloop]
    
    #setting initial starting postion and orientation to a known value
    initcx, initcy, initth =x[0]-(-5), y[0]-(-8),theta[0]-(0)
    initcx, initcy, initth=initcx.reshape((-1,1)), initcy.reshape((-1,1)), initth.reshape((-1,1))
    
    #all constraints are concatenated into single array and reshaped into 1D array
    final = jnp.concatenate((odocx,odocy,odocth,loopcx,loopcy,loocpctx,initcx,initcy,initth)).reshape(-1)
    return final

#creating a matrix called Omega
def CreateOmega(odo=1, loop=1,init=10):

    #initialises a square matrix filled with zeros (119 odometry measurements, 20 loop closures, 3 initial conditions)
    omega = jnp.zeros((119*3 + 20*3+3, 119*3 + 20*3+3))

    #setting up local odometry and loop cosure mesurement variables
    xodo = odo
    yodo = odo
    todo = odo
    xloop = loop
    yloop = loop
    tloop = loop

    #populate omega matrix with x-component odometry measurements    
    for i in range(119):
        omega = omega.at[i, i].set(xodo)
    
    #populate omega matrix with y-component odometry measurements
    for i in range(119, 2*119):
        omega = omega.at[i, i].set(yodo)

    #populate omega matrix with theta-component odometry measurements
    for i in range(2*119, 3*119):
        omega = omega.at[i, i].set(todo)

    #populate omega matrix with x-component loop-closure measurements
    for i in range(3*119, 3*119 + 20):
        omega = omega.at[i, i].set(xloop)

    #populate omega matrix with y-component loop-closure measurements
    for i in range(3*119 + 20, 3*119 + 2*20):
        omega = omega.at[i, i].set(yloop)

    #populate omega matrix with theta-component loop-closure measurements
    for i in range(3*119 + 2*20, 3*119 + 3*20):
        omega = omega.at[i, i].set(tloop)

    #populate omega matrix with initial conditions
    for i in range(3*119 + 3*20, 3*119 + 3*20+3):
        omega = omega.at[i, i].set(init)

    return omega 

def jacobian(thetha,edges):

    #extrating indesices and changes in x,y from edges array
    i, j, delx, dely, delth=edges[0], edges[1], edges[2], edges[3], edges[4]
    
    #asignes odocl the size of the theta array for upcoming loops
    odocl=thetha.shape[0]

    #asinging local variables for loop closure
    loop1, loop2=i[odocl-1:], j[odocl-1:]

    #initialising jacobian matrix
    jaco=[]

    #loop for every number in matrix
    x = 0 
    while x < odocl:

        #creating blank row in change in x matrix
        Delft=jnp.zeros(420).tolist()

        #if not the first or last index set position in array to 1 and previous to -1
        if x!=0 and x!=odocl-1:
            Delft[x]=1
            Delft[x-1]=-1
        
        #if last index set 118th element to -1
        elif x==odocl-1:
            Delft[118]=-1
        
        #if first set first and third to last emements to 1
        elif x ==0:
            Delft[0]=1
            Delft[-3]=1
            
        #if not the first or last index set position in array to 1 and previous to -1
        if x!=0 and x!=odocl-1: 
            Delft[x]=1
            Delft[x-1]=-1

        #if last index set 118th element to -1
        elif x==odocl-1:
            Delft[118]=-1

        #if first set first and third to last emements to 1
        elif x == 0:
            Delft[0]=1
            Delft[-3]=1
        
        #sub-loop runs 20 times checking if current x matches any indices in loop1
        i = 0 
        while i < 20 :

            #if there is a match set element to 1
            if loop1[i]==x:
                Delft[357+i]=1    
            i+=1

        #sub-loop runs 20 times checking if current x matches any indices in loop2
        j = 0 
        while j < 20:

            #if there is a match set element to 1
            if loop2[j]==x:
                Delft[357+j]=-1
            j+=1
        
        #adding new row to jacobian matrix
        jaco.append(Delft)
        x+=1
        

    y = 0 
    while y < odocl :

        #creating blank row in change in y matrix
        delfy=jnp.zeros(420).tolist()

        #if not the first or last index set position in array to 1 and previous to -1
        if y!=0 and  y!=odocl-1:
            
            delfy[119+y]=1
            delfy[119+y-1]=-1
        #if last index set 237th element to -1
        elif y==odocl-1:
            delfy[237]=-1

        #if first set 199th and second to last emements to 1
        else:
            delfy[119+0]=1
            delfy[-2]=1
            
        #sub-loop runs 20 times checking if current y matches any indices in loop1
        i = 0 
        while i < 20:
            if loop1[i]==y:

                #if there is a match set element to -1
                delfy[357+20+i]=1
            i+=1    

        #sub-loop runs 20 times checking if current y matches any indices in loop2
        j = 0
        while j < 20:
            if loop2[j]==y:

                #if there is a match set element to -1
                delfy[357+20+j]=-1
            j+=1

        #adding new row to jacobian matrix
        jaco.append(delfy)
        y+=1
        
    t=0
    while t < odocl :
        #creating blank row in change in x matrix
        delft=jnp.zeros(420).tolist()
        
        #if element is smaller than 119 set 356th value to -1
        if  t==119:
            delft[237+119]=-1

        #if first index set last element and 238th in list to 1 (setting initial conditions)
        elif t==0:
            delft[-1]=1
            delft[238+t]=1

        #for all other values set values 237 + t to -1 and 238 + t to 1 (representing relationship between consecutive theta values)
        else:
            delft[237+t]=-1
            delft[238+t]=1

        #if theta is less than 199 calculate two elemtnts on the delft list.
        if t<119:
            
            #change the values values to the divirtieves of x and y in respect to theta
            delft[t] = (-delx[t]*jnp.sin(thetha[t])-dely[t]*jnp.cos(thetha[t]))
            delft[119+t] = (-dely[t]*jnp.sin(thetha[t])+delx[t]*jnp.cos(thetha[t]))
        
        #sub-loop runs 20 times checking if current y matches any indices in loop2
        i = 0 
        while i < 20:
            if loop1[i]==t:

                #if there is a match set element to 1
                delft[357+i] = (-delx[119+i]*jnp.sin(thetha[t])-dely[119+i]*jnp.cos(thetha[t]))
                delft[357+20+i] = (-dely[119+i]*jnp.sin(thetha[t])+delx[119+i]*jnp.cos(thetha[t]))
                delft[357+40+i] = 1
            i+=1

        #sub-loop runs 20 times checking if current y matches any indices in loop2
        j=0
        while j < 20 :
            if loop2[j]==t:

                #if there is a match set element to -1
                delft[357+40+j] = -1
            j+=1
        
        #adding new row to jacobian matrix
        jaco.append(delft)
        t+=1
        

    #convert to jax array
    final = jnp.array(jaco).T
    return final

#apply the 538 sytle to the matlabs graph
plt.style.use('fivethirtyeight')

#calculate the frobenius norm of the difference between the two matrices
def frobNorm(P1, P2, str1="mat1", str2="mat2"):

    #suppresses scientific notation for NumPy
    jnp.set_printoptions(suppress=True)

    #calculating the frobenious norm from the difference of two element
    val = jnp.linalg.norm(P1 - P2, 'fro')

    #printing the result
    print(f"Frobenius norm between {str1} and {str2} is: {val}")        
    
#implemeting Levenberg-Marquardt algoithm (solving non-linear least squares problems)
def levenMarq(x,y,theta,edges):

    #initializing variables
    cost_itr, x_itr =[], []
    it = 0
    lamda=0.1

    #creates weight matrix
    w=CreateOmega(2,500,1000)

    #seting max_iterations and tolerance
    max_itr = 50
    tolerance = 0.000001

    while it < max_itr:

        #concatenates x,y,theta arrays into single vertices array
        vertices=jnp.concatenate((x,y,theta)).reshape(-1)

        #creates constraints array using vertices and edges
        f=get_fX(vertices,edges)

        #using lambda function alter constraints array
        lambf = lambda aa : get_fX(aa,edges)

        #calculate hacobian matrix using lambf and current vertices
        j=jacfwd(lambf)(vertices)

        #Calculates approximate hessian matrix and gradient vector for current state
        H, b = j.T@w@j + lamda*jnp.eye(360), j.T@w.T@f

        #finds delx by inverting hessian matrix and multiplying by negative gradient vector
        delx = -jnp.linalg.inv(H)@b

        #reshapes delx into three updates
        dX,dY,dT = jnp.reshape(delx,(3,len(delx)//3))

        #applies x,y,theta updates
        x+=dX
        y+=dY
        theta+=dT

        #calculates new costn using new constraints array
        costn = 0.5*f.T@w@f

        #appends new cost to list
        cost_itr.append(costn)

        #prints current iteration number and lastest cost
        l = len(cost_itr)
        print("err",it+1,"=",cost_itr[l-1])

        #appending current state to x_itr
        x_itr.append([x,y,theta])

        #if not first iteration
        if it!=0:

            #adjusts damping factor lamda based on if cost has increased or decreased
            l = len(cost_itr)
            if cost_itr[l-1]>cost_itr[l-2]:
                lamda/=10
            else:
                lamda*=10

            #if change in cost is less than tolerance if so breaking loop                
            if cost_itr[l-2] - cost_itr[l-1] < tolerance:
                break
        it+=1
        
    return cost_itr,x_itr

#creating global positional and rotational arrays
X = jnp.array(X)
Y = jnp.array(Y)
THETA = jnp.array(THETA)

#calculating opimisation
cost,x=levenMarq(X, Y, THETA,edges) 
l = len(x)

#sending back up files to .g2o
makeg2o(x[l-1][0],x[l-1][1], x[l-1][2],g2ofile="opt.g2o")
makeg2o(x[l-1][0],x[l-1][1], x[l-1][2],g2ofile="opt-backup.g2o")

#output plot to compare three different trajectories on a graph
def final_output(vertex_gt, xn, yn, tn):  

    #setting local variables
    ax = plt.subplot()
    c = 0.25
    xgt, ygt=vertex_gt[0], vertex_gt[1]
    tg = vertex_gt[2]


    i = 0 
    while i < len(tg):

        #calculates end points of the arrows to show direction and magnitude
        pls, ply = c*math.cos(tg[i]) + xgt[i],  c*math.sin(tg[i]) + ygt[i]
        
        #ploting each point of "ground truth"
        if i == 0:
            plt.plot([xgt[i], pls], [ygt[i], ply], 'r->', label="Ground Truth")
        else :
             plt.plot([xgt[i], pls], [ygt[i], ply], 'r->')
        i+=1

    
    i = 0
    while i < len(tn):

        #calculates end points of the arrows to show direction and magnitude
        pls, ply = c*math.cos(tn[i]) + xn[i],  c*math.sin(tn[i]) + yn[i]

        #ploting each point of "Final trajectory"
        if i == 0 :
            plt.plot([xn[i], pls], [yn[i], ply], 'g->', label="Final Trajectory")
        else :
             plt.plot([xn[i], pls], [yn[i], ply], 'g->')
        i+=1
        

    i = 0
    while i < len(THETA):

        #calculates end points of the arrows to show direction and magnitude
        pls, ply = c*math.cos(THETA[i]) + X[i],  c*math.sin(THETA[i]) + Y[i]

        #ploting each point of "Initial trajectory"
        if i == 0 :
            plt.plot([X[i], pls], [Y[i], ply], 'b->', label="Initial Trajectory")
        else :
            plt.plot([X[i], pls], [Y[i], ply], 'b->')
        i+=1
    
    #creates a legnend and shows plot with trajectories
    plt.legend()
    plt.show()
    
#reading vertexes of dataset
vertexgt = readVertex('./dataset/gt.txt')

#printing iterations number, cost and outputing graph to compare trajectories
for i in range(1, l+1):
    print("Iterations : ",str(i)," Loss : ",cost[i-1])
    xn,yn,tn=x[i-1][0],x[i-1][1],x[i-1][2]
    final_output(vertexgt, xn,yn,tn)