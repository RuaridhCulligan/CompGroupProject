#Boundary Conditions
#Very much work in progress
#Initialise Potetial
import numpy as np




# if V == "Tunnel":
#     V = b.tunnel(V0,d,x)
# elif V == "Diffraction":
#     V = b.diffraction(w,d,x)
# elif V == "Collision":
#     V = b.collision(V0,alpha,x1,x2)
# else:
#     V = np.zeros(xn)



def tunnel(V0,d,x0,x1,dx):

    x = np.arange(x0,x1+dx,dx) 
    xn = np.shape(x)[0]

    V = np.zeros(xn)

    for i in range(xn):
        if np.abs(x[i]) < d/2:
            V[i] = V0
        else:
            continue
    return 
    
def diffraction(x0,x1,dx,y0,y1,dy,d,w):
    
    x = np.arange(x0,x1+dx,dx) 
    xn = np.shape(x)[0]

    y = np.arange(y0,y1+dy,dy) 
    yn = np.shape(y)[0]

    V = np.zeros(xn,yn)

    for i in range(xn):
        for j in range(yn):
            if x[i] < d/2 or y[j] > w/2:
                V[i,j] = np.inf
            else:
                continue 

    return V

def collision(x1_init, x1_fin, dx1, x2_init, x2_fin, dx2, alpha, v0):

    x1 = np.arange(x1_init,x1_fin+dx1,dx1) 
    x1n = np.shape(x1)[0]

    x2 = np.arange(x2_init,x2_fin+dx2,dx2) 
    x2n = np.shape(x1)[0]

    

    return V
                
    


