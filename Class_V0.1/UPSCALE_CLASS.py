import cv2
import numpy as np
from tqdm import tqdm
from itertools import product
import time as t

class upsc:
    
   def __init__ (self):
       
       #import the picture using cv2 librairy
       
       self.picture_color = cv2.imread('fraise4.jpg')
       self.picture_color = cv2.convertScaleAbs(self.picture_color) 
       
       #Creating the three matrices corresponding to each layer of the initial colored pictured 
       
       self.picture_B = self.picture_color[:,:,0]
       self.picture_G = self.picture_color[:,:,1]
       self.picture_R = self.picture_color[:,:,2]
       
   def black_edge (self,matrix_init,matrix_shape):
       
        #Adding a black line of two pixels around the initial matrix
       
        self.mag_matrix = np.zeros((matrix_shape[0] + 2,matrix_shape[1] + 2))
        
        for i in range (0, np.shape(matrix_init)[0]):
             for j in range (0, np.shape(matrix_init)[1]):
                 self.mag_matrix[i+1, j+1] = matrix_init[i, j]
                 
   def matrix_G (self,i,j,F): 
    
        self.G = np.array([[F[i,j], F[i,j+1], (F[i,j+1]-F[i,j-1])/2, (F[i,j+2]-F[i,j])/2],
                [F[i+1,j], F[i+1,j+1], (F[i+1,j+1]-F[i+1,j-1])/2, (F[i+1,j+2]-F[i+1,j])/2],
                [(F[i+1,j]-F[i-1,j])/2, (F[i+1,j+1]-F[i-1,j+1])/2, (F[i+1,j+1]-F[i+1,j]-F[i,j+1]+2*F[i,j]-F[i-1,j]-F[i,j-1]+F[i-1,j-1])/2, (F[i+1,j+2]-F[i+1,j+1]-F[i,j+2]+2*F[i,j+1]-F[i-1,j+1]-F[i,j]+F[i-1,j])/2],
                [(F[i+2,j]-F[i,j])/2, (F[i+2,j+1]-F[i,j+1])/2, (F[i+2,j+1]-F[i+2,j]-F[i+1,j+1]+2*F[i+1,j]-F[i,j]-F[i+1,j-1]+F[i,j-1])/2, (F[i+2,j+2]-F[i+2,j+1]-F[i+1,j+2]+2*F[i+1,j+1]-F[i,j+1]-F[i+1,j]+F[i,j])/2]])

   def Pf(self,x,y,C):
       
        X = np.array([1, x, x**2, x**3])
        Y = np.array([1, y, y**2, y**3])
        self.Z = X@C@np.transpose(Y)


   def Interpol(self,H):
       
        A=np.array([[1, 0, 0, 0],
                    [1, 1, 1, 1],
                    [0, 1, 0, 0],
                    [0, 1, 2, 3]])
    
        B = np.transpose(A)
        A = np.linalg.inv(A)
        B = np.linalg.inv(B)
        h_shape = H.shape
        self.N = np.zeros((2*h_shape[0]-2,2*h_shape[1]-2))
        
        #Creating the loading bar 
        with tqdm(total = 100) as pbar:
            
            for i in range (0,h_shape[0]-2):
                
                #Updating the loading bar
                pbar.update(100/(h_shape[0]-2))
                
                limit1 = range (0,h_shape[1]-2)
                limit2 = range(0,2,1)
                limit3 = range(0,2,1)
                
                for j,y,x in product (limit1,limit2,limit3):
                    
                    if x == 0 and y == 0:
                        self.N[2*i,2*j] = H[i,j]
                        
                    if x == 0 and y == 2:
                        self.N[2*i,2*j+1] = H[i,j+1]
                        
                    if x == 2 and y == 0:
                       self.N[2*i+1,2*j] = H[i+1,j]
                        
                    if x == 2 and y == 2:
                        self.N[2*i+1,2*j+1] = H[i+1,j+1]
                        
                    else:
                        self.matrix_G(i,j,H)
                        C = A@self.G@B
                        
                        self.Pf(x/2,y/2,C)
                        self.N[2*i+x,2*j+y] = self.Z


   def save_image (self,file_name):
       
        #Recreating a matrix with three colored layer
       
        self.black_edge(self.picture_B, self.picture_B.shape)
        matrix_blue_edge = self.mag_matrix
        
        self.Interpol(matrix_blue_edge)
        matrix_blue = self.N
        
        self.black_edge(self.picture_G, self.picture_G.shape)
        matrix_green_edge = self.mag_matrix

        self.Interpol(matrix_green_edge)
        matrix_green = self.N
        
        self.black_edge(self.picture_R, self.picture_R.shape)
        matrix_red_edge = self.mag_matrix
        
        self.Interpol(matrix_red_edge)
        matrix_red = self.N
        
        C = np.zeros((matrix_blue.shape[0],matrix_blue.shape[1],3))
        
        C[:,:,0] = matrix_blue
        C[:,:,1 ]= matrix_green
        C[:,:,2] = matrix_red
        
        #Saving the file
    
        cv2.imwrite (file_name,C)

    
image = upsc()
image.save_image("test.jpg")
