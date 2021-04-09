import time
import os
import shutil

def dossier (nom):
    os.mkdir(nom)
    
def path(nomphoto,nomdossier):
    chemin = os.path.sep.join([directory(),nomdossier,nomphoto])
    return(chemin)

def cleardossier(chemin) :
    a = str(os.path.abspath(os.getcwd()))
    path = os.path.sep.join([a,chemin])
    if os.path.exists(path):
        shutil.rmtree(os.path.sep.join([directory(),chemin]))
        time.sleep(0.5)
        
def heure ():
    now = time.localtime(time.time())
    year, month, day, hour, minute, second, weekday, yearday, daylight = now
    return "%02dm%02dd%02dh%02dm%02ds" % (month,day,hour, minute, second)

def clear (nomfichier):
    fichier = open(nomfichier,"w")
    fichier.close()
    
def write (string,nomfichier):
    fichier = open(nomfichier,"a")
    fichier.write(string)
    fichier.close()
    
def save (nomfichier):
    chemin = "Archivage résultat/"
    if os.path.exists(os.path.sep.join([directory(),chemin])):
        cheminfichier = "Archivage résultat/"+"résultat ("+str(heure())+").txt"
        shutil.copy(os.path.sep.join([directory(),nomfichier]),os.path.sep.join([directory(),cheminfichier]))
        
    else :
        dossier("Archivage résultat")
        cheminfichier = "Archivage résultat/"+"résultat ("+str(heure())+").txt"

        shutil.copy(os.path.sep.join([directory(),nomfichier]),os.path.sep.join([directory(),cheminfichier]))
        
def directory ():
    a = str(os.path.abspath(os.getcwd()))
    return(a)
