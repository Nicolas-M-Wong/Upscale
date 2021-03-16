import cv2
import traitement_image_lib as img
import numpy as np
import time as t
import enregistrement as record

record.clear("result.txt")
record.clear("couleur.txt")

FraiseC=cv2.imread('fraise4.jpg')
nbiteration = 1
list_time = np.zeros(nbiteration)
namedir = "PhotoUpscale"

record.cleardossier(namedir)
record.dossier(namedir)

for i in range (0,nbiteration):
    #%% Upscale Classique
    titre = "Upscale"+str(i)+".jpg"
    t0 = t.time() 
    
    #%% Calcul des temps d'upscale
    cv2.imwrite(record.path(titre,namedir),img.upscale(FraiseC,"couleur.txt"))
    t1 = t.time()
    list_time[i] = (t1 - t0)


#%% Ecriture des résultats dans un fichier
    
    record.write ("Temps image "+str(i+1)+" : "+str(list_time[i])+'\n',"result.txt")
    print("\n\nTemps upscale : ",list_time[i])
    if nbiteration !=1 and i == 0:
        img.estimation(nbiteration-1,t1-t0)
    
"""#%% Amélioration de la netteté
    t2 = t.time()
    image = cv2.imread(record.path("Resultat"+str(i+1)+".jpg",namedir))#e.path(titre,nomdossier)
    img.nettete(image)
    t3 = t.time()
#%% Ecriture des resultat et estimation
    print('temps netteté : ', t3-t2)
    record.write("Temps image netteté "+str(i+1)+" : "+str(t3-t2)+'\n',"result.txt")
"""

#%%affichage


record.write("\n   Temps image moyen : "+str(np.mean(list_time))+'\n',"result.txt")
record.write("   Heure de fin : "+str(record.heure())+'\n',"result.txt")
record.save("result.txt")  
