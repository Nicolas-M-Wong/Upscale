#%% Import libraries

import cv2
import traitement_image_lib as img
import time as t
import enregistrement as record

#%% Init setup

record.clear("result.txt")          #Create and empty the file result
record.clear("couleur.txt")         #Same for the file couleur (which record the time taken by each sub-matrix R G B)

picture_color = cv2.imread('fraise4.jpg')   #Import the matrix corresponding to the picture we want to upscale

namedir = "PhotoUpscale"            #Name of the directory that will contain the final picture

# !!!! Be careful if another folder has the same name it will be deleted !!!!

record.cleardossier(namedir)        #Clear the folder created to avoid conflict a folder has the same name
record.dossier(namedir)             #Create the new folder 


#%% Upscale Classique

titre = "Upscale.jpg"               #Name of the new picture
t0 = t.time()                       #Init the timer

#%% Upscaling using the function in the library traitement_image_lib

cv2.imwrite(record.path(titre,namedir),img.upscale(picture_color,"couleur.txt"))
t1 = t.time()                       #End of the timer


#%% Writing the perf metrics

record.write ("Temps image "+" : "+str(t1-t0)+'\n',"result.txt")
record.write("\n   Temps image moyen : "+str(t1-t0)+'\n',"result.txt")
record.write("   Heure de fin : "+str(record.heure())+'\n',"result.txt")

record.save("result.txt")  


"""#%% Amélioration de la netteté
    t2 = t.time()
    image = cv2.imread(record.path("Resultat"+str(i+1)+".jpg",namedir))#e.path(titre,nomdossier)
    img.nettete(image)
    t3 = t.time()
#%% Ecriture des resultat et estimation
    print('temps netteté : ', t3-t2)
    record.write("Temps image netteté "+str(i+1)+" : "+str(t3-t2)+'\n',"result.txt")
"""
