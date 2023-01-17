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


