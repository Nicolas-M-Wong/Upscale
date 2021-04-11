#%% Import libraries

import cv2
import numpy as np
import time
import enregistrement as record
from tqdm import tqdm
from itertools import product

#%% Function concerning only the upscale part

def upscale(image,nomfichier):
    
    z = ImC(image,nomfichier)
    z = cv2.convertScaleAbs(z) 
    
    return z

def ImC(image,nomfichier) : 
    
    record.write ('\n',nomfichier)

    R = Imc_monochrome_red(image,nomfichier)

    G = Imc_monochrome_green(image,nomfichier)

    B = Imc_monochrome_blue(image,nomfichier)

    C=np.zeros((R.shape[0],R.shape[1],3))
    C[:,:,0]=B
    C[:,:,1]=G
    C[:,:,2]=R
    
    return(C)

def Imc_monochrome_blue(image,nomfichier):
    
    b1 = time.time()
    b = image[:,:,0]
    B = black_edge(b,b.shape)
    B = Interpol(B,0)
    b2 = time.time()
    record.write ("Blue : "+str(b2-b1)+'\n',nomfichier)
    
    return(B)
    
    
def Imc_monochrome_green(image,nomfichier):
    
    g1 = time.time()
    g = image[:,:,1]
    Gr = black_edge(g,g.shape)
    Gr = Interpol(Gr,0)
    g2 = time.time()
    record.write ("Green : "+str(g2-g1)+'\n',nomfichier)
    
    return (Gr)    


def Imc_monochrome_red (image,nomfichier) :
    
    r1 = time.time()
    r = image[:,:,2]
    R = black_edge(r,r.shape)
    R = Interpol(R,0)
    r2 = time.time()
    record.write ("Red : "+str(r2-r1)+'\n',nomfichier)
    
    return (R)
    
def black_edge(E,g):
    
    O = np.zeros((g[0]+2,g[1]+2))    
    for p in range (0,np.shape(E)[0]):
        for q in range (0,np.shape(E)[1]):
            O[p+1,q+1] = E[p,q]
            
    return (O)

def matrix_G(i,j,F):                #Theoritical matrix found using interpolation closed to the cubic interpolation
    
    G = np.array([[F[i,j], F[i,j+1], (F[i,j+1]-F[i,j-1])/2, (F[i,j+2]-F[i,j])/2],
            [F[i+1,j], F[i+1,j+1], (F[i+1,j+1]-F[i+1,j-1])/2, (F[i+1,j+2]-F[i+1,j])/2],
            [(F[i+1,j]-F[i-1,j])/2, (F[i+1,j+1]-F[i-1,j+1])/2, (F[i+1,j+1]-F[i+1,j]-F[i,j+1]+2*F[i,j]-F[i-1,j]-F[i,j-1]+F[i-1,j-1])/2, (F[i+1,j+2]-F[i+1,j+1]-F[i,j+2]+2*F[i,j+1]-F[i-1,j+1]-F[i,j]+F[i-1,j])/2],
            [(F[i+2,j]-F[i,j])/2, (F[i+2,j+1]-F[i,j+1])/2, (F[i+2,j+1]-F[i+2,j]-F[i+1,j+1]+2*F[i+1,j]-F[i,j]-F[i+1,j-1]+F[i,j-1])/2, (F[i+2,j+2]-F[i+2,j+1]-F[i+1,j+2]+2*F[i+1,j+1]-F[i,j+1]-F[i+1,j]+F[i,j])/2]])
    
    return G

def Pf(x,y,C):
    
    X = np.array([1, x, x**2, x**3])
    Y = np.array([1, y, y**2, y**3])
    Z = X@C@np.transpose(Y)
    
    return (Z)

def Interpol(H,progress):
    
    A = np.array([[1, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 2, 3]])

    B = np.transpose(A)
    A = np.linalg.inv(A)
    B = np.linalg.inv(B)
    h = H.shape
    N = np.zeros((2*h[0]-2,2*h[1]-2))
    
    with tqdm(total = 100) as pbar:
        
        for i in range (0,h[0]-2):
            
            pbar.update(100/(h[0]-2))
            
            limite1 = range (0,h[1]-2)
            limite2 = range(0,2,1)
            limite3 = range(0,2,1)
            
            for j,y,x in product (limite1,limite2,limite3):
                
                if x == 0 and y == 0:
                    N[2*i,2*j] = H[i,j]
                    
                if x == 0 and y == 2:
                    N[2*i,2*j+1] = H[i,j+1]
                    
                if x == 2 and y == 0:
                    N[2*i+1,2*j] = H[i+1,j]
                    
                if x == 2 and y == 2:
                    N[2*i+1,2*j+1] = H[i+1,j+1]
                    
                else:
                    C = A @ matrix_G(i,j,H)@B
                    N[2*i+x,2*j+y] = Pf(x/2,y/2,C)
    
    return N

#%% Other function unused but could be usefull for working on matrix or image

def sharpness (image):            #Sharpness improvement (unused here but can be added at the end of the process of upscalling if needed)
    Mconv = np.array([[0, 0, 0, 0, 0],[0, 0, -1, 0, 0],[0, -1, 5, -1, 0],[0, 0, -1, 0, 0],[0, 0, 0, 0, 0]])
    m,n,p = np.shape(image)
    mc, nc = np.shape(Mconv)
    
    U = np.zeros((m, int(mc/2)))
    V = np.zeros((int(nc/2), n+2*int(mc/2)))
    
    Sconv = np.sum(Mconv)
    if Sconv == 0:
        Sconv = 1
    
    S = np.zeros((m, n,p))
    
    with tqdm (total = 100) as pbar : 
        for q in range(p):
            pbar.update(100/p)
            
            Image = np.concatenate((np.concatenate((V, np.concatenate((np.concatenate((U, image[:,:,q]), axis=1), U), axis=1))), V))
            for i in range(0,m):
                for j in range(0,n):
                    N = 0
                    for k in range(0,mc):
                        for l in range(nc):
                            N += Image[i+k, j+l] * Mconv[k,l]
                    S[i,j, q] = N/Sconv
    S = cv2.convertScaleAbs(S)
    cv2.imwrite("Sharp.jpg", S)

#%% Wavelet compression 
    
def checkdim(A):                #Adjusting the dimension so that we can apply the wavelet compression (dimension/8 in both axis)
    n,m = np.shape(A)
    rn = n%8
    rm = m%8
    if rn !=0 or rm != 0:
        if rn != 0:
              
              h = 8-rn
              nfin = n+h
             
              U = np.zeros([nfin,m])
              for i in range (0,n):
                  for j in range (0,m):
                      U[i,j] = A[i,j]
        
              A = U
              n = nfin 
              
        if rm != 0:
              g = 8-rm
              mfin = m+g
              U = np.zeros([n,mfin])
              for i in range (0,n):
                  for j in range (0,m):
                      U[i,j] = A[i,j]
                 
        U = cv2.convertScaleAbs(U)
        return U
    else :
        
        return A


def compression_wavelet (A):    #Compression to get the sparse matrix

    W = np.array([[1/8, 1/8, 1/4, 0, 1/2, 0, 0, 0],
                  [1/8, 1/8, 1/4, 0, -1/2, 0, 0, 0],
                  [1/8, 1/8, -1/4, 0, 0, 1/2, 0, 0],
                  [1/8, 1/8, -1/4, 0, 0, -1/2, 0, 0],
                  [1/8, -1/8, 0, 1/4, 0, 0, 1/2, 0],
                  [1/8, -1/8, 0, 1/4, 0, 0, -1/2, 0],
                  [1/8, -1/8, 0, -1/4, 0, 0, 0, 1/2],
                  [1/8, -1/8, 0, -1/4, 0, 0, 0, -1/2]])
    
    A = checkdim(A)
    n,m = np.shape (A)
    B = np.zeros([n,m])
    C = np.zeros([8,8])
    limite1 = range(0,n,8)
    limite2 = range(0,m,8)
    limiteC = range(0,8)

    with tqdm(total=100) as pbar:
        for k in limite1:
            pbar.update(800/n)
            for l in limite2:
                for i,j in product(limiteC,limiteC):
                    
                    C[i,j] = A[k+i,j+l]
                D = W@C@(np.transpose(W))
                
                for y,z in product (limiteC,limiteC):
                    B[k+y,l+z] = D[y,z]
            
    return(B)
                    
def decompression_wavelet (B):  #Get the new matrix from the sparse matrix
    
    B = checkdim(B)
    n,m = np.shape (B)
    
    W = np.array([[1/8, 1/8, 1/4, 0, 1/2, 0, 0, 0],
                  [1/8, 1/8, 1/4, 0, -1/2, 0, 0, 0],
                  [1/8, 1/8, -1/4, 0, 0, 1/2, 0, 0],
                  [1/8, 1/8, -1/4, 0, 0, -1/2, 0, 0],
                  [1/8, -1/8, 0, 1/4, 0, 0, 1/2, 0],
                  [1/8, -1/8, 0, 1/4, 0, 0, -1/2, 0],
                  [1/8, -1/8, 0, -1/4, 0, 0, 0, 1/2],
                  [1/8, -1/8, 0, -1/4, 0, 0, 0, -1/2]])
    
    A = np.zeros([n,m])
    C = np.zeros([8,8])
    
    with tqdm(total=100) as pbar:
        for k in range (0,n,8):
            pbar.update(800/n)
            for l in range (0,m,8):
                for i in range (0,8):
                    for j in range (0,8):
                        C[i,j] = B[k+i,j+l]
                        
                Winv = np.linalg.inv(W)
                WTinv = np.linalg.inv(np.transpose(W))
                D = Winv@C@WTinv
                
                for u in range (0,8):
                    for v in range (0,8):
                        A[k+u,l+v] = D[u,v]

    A = cv2.convertScaleAbs(A)   
    return (A)


def compression_couleur (img):
    
    monochrome0 = img[:,:,0]
    monochrome0 = compression_wavelet(monochrome0)
    
    monochrome1 = img[:,:,1]
    monochrome1 = compression_wavelet(monochrome1)
    
    monochrome2 = img[:,:,2]   
    monochrome2 = compression_wavelet(monochrome2)
    
    imagetotal = np.zeros((monochrome0.shape[0],monochrome0.shape[1],3))
    imagetotal[:,:,0] = monochrome0
    imagetotal[:,:,1] = monochrome1
    imagetotal[:,:,2] = monochrome2
    
    imagetotal = cv2.convertScaleAbs(imagetotal)
    
    return (imagetotal)

def decompression_couleur (img):
    
    monochrome0 = img[:,:,0]
    monochrome0 = decompression_wavelet(monochrome0)
    
    monochrome1 = img[:,:,1]
    monochrome1 = decompression_wavelet(monochrome1)
    
    monochrome2 = img[:,:,2]   
    monochrome2 = decompression_wavelet(monochrome2)
    
    imagetotal = np.zeros((monochrome0.shape[0],monochrome0.shape[1],3))
    imagetotal[:,:,0] = monochrome0
    imagetotal[:,:,1] = monochrome1
    imagetotal[:,:,2] = monochrome2
    
    imagetotal = cv2.convertScaleAbs(imagetotal)
    
    return (imagetotal)

def Matrice2CSR (A):            #From initial matrix to a CSR (compressed Row Storage) matrix
    
    n,m = np.shape(A)
    AX = []
    AJ = []
    AI = [0]
    k = 0
    
    for i in range (m):
        for j in range (n):
            
            if A[i,j]!= 0:
                
                k += 1
                AX = np.concatenate( (AX,[A[i,j]]),axis = 0)
                AJ = np.concatenate ( (AJ,[j]), axis = 0)
                
        AI = np.concatenate((AI,[k]), axis=0) 
        
    return (AX,AJ,AI)


def CSR2Matrice(AX,AJ,AI):  #From CSR matrix to the initial matrix
   
    n = np.size(AI) - 1
    A = np.zeros((n,n))
    s = 0
    
    for i in range(n):
        nbr_term = AI[i+1]
        
        while s < nbr_term:
            for j in range(s,nbr_term):
                ind_col = int(AJ[j])
                A[i,ind_col] = AX[s]
                s+=1
    return A


def ProdCSR(AX,AJ,AI,x):    #Multiplication using CSR matrix (using sparse matrix to improve performance)
   
    P = np.zeros(np.shape(x))   
    n = np.size(AI) - 1
    c = 0
    
    for i in range(n):
        nbr_term = AI[i+1] 
        S = 0 
        
        while c < nbr_term:
            for j in range(c,nbr_term):
                ind_col = int(AJ[j])
               
                S += AX[c]*x[ind_col]
                c+=1
                
        P[i] = S
   
    return P

#%% Singular Value Decomposition compression of a matrix (unused here)
def SVD_compression(img,r):
    U,S,VT = np.linalg.svd(img)
    n,m = np.shape(img)

    for i in range(0, r):
        reconstimg = np.matrix(U[:, :i]) * np.diag(S[:i]) * np.matrix(VT[:i, :])
    reconstimg = cv2.convertScaleAbs(reconstimg)
    return (reconstimg)


