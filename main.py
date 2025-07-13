import numpy as np
from matplotlib import pyplot as plt
from decimal import *
import time
import json
import random
import math

getcontext().prec = 25 #détermine la précision des calculs

epsilon = Decimal(1e-15)

class Demi_espace:
    def __init__(self, dir, b):
        self.dir = dir
        self.b = b

#projette sur un demi-espace
def proj_demi_espace(dir, b, x):
    prod = Decimal(np.dot(x, dir)) - b
    if (prod >= -epsilon):
        return x
    else:
        return x -(dir * (prod/Decimal(np.linalg.norm(dir))**Decimal(2)))

#forme linéaire associée à l'hyperplan
def phi(w, b, x):
    return np.dot(w, x) + b

#exécute POCS
def pocs(x0, convexes):
    d = len(convexes)
    u = x0
    i = 0
    while True:
        u = proj_demi_espace(convexes[i].dir, convexes[i].b, u)
        if (dans_inter(u, convexes)):
            return u
        i = (i+1) % d

#vérifie si un point est dans l'intersection des convexes dans l'ensemble convex
def dans_inter (x, convex):
    for e in convex:
        print(x)
        print(e.dir)
        print(e.b)
        print("valeur : ", np.dot(x, e.dir) - e.b)
        if ((np.dot(x, e.dir) - e.b) < -epsilon):
            return False
    return True

#projection de Dykstra pour n = 2
def dykstra_projection_2(x0, convexes):
    p = np.full(x0.size, Decimal(0))
    q = np.full(x0.size, Decimal(0))
    y = np.full(x0.size, Decimal(0))

    while (not dans_inter(x0, convexes)):
        y = proj_demi_espace(convexes[0].dir, convexes[0].b, x0 + p)
        p = (x0 + p) + ((-1) * y)
        x0 = proj_demi_espace(convexes[1].dir, convexes[1].b, y + q)
        q = (y + q) + ((-1) * x0)
    
    return x0

#projection de Dykstra
def dykstra_projection(x0, convexes):
    d = len(convexes)
    z = np.full((d, x0.size), Decimal(0))
    u = np.full((d+1, x0.size), Decimal(0))
    u[d] = x0
    while (not dans_inter(u[d], convexes)):
        u[0] = u[d]
        for i in range (d):
            u[i+1] = proj_demi_espace(convexes[i].dir, convexes[i].b, u[i] + z[i])
            if (dans_inter(u[i+1], convexes)):
                return u[i+1]
            z[i] = (u[i] + z[i]) + ((-1) * u[i+1])
    return u[d]

#renvoie le gradient dans le cas linéairement séparable
def gradient_lineairement_separable(u):
    a = np.append(u[0:u.size-1], Decimal(0))
    return a

#exécute l'algorithme de la descente de gradient projeté
def gradient_projete(u0, pas, limite, gradient, convexes):
    u = u0
    oldU = u0
    first = True
    if(dans_inter(u0, convexes)):
        first = False
    while(first or abs((np.linalg.norm(gradient(u)) - np.linalg.norm(gradient(oldU)))) > limite):
        oldU = u
        #u = pocs(u-pas*gradient(u), convexes)
        u = dykstra_projection(u-pas*gradient(u), convexes)
        first = False
        print(u)
    print("dans inter", dans_inter(u, convexes))
    return (dykstra_projection(u, convexes))

#résoud le problème SVM dans le cas linéairement séparable 
def svm_lineairement_separable(points, pas, limite):
    dim = points[0].size
    for point in points:
        for i in range (dim-1):
            point[i] = point[i]*point[dim-1] #on multiplie par la classe
    hyperplan = np.full(dim, Decimal(0))
    print(hyperplan)
    convexes = [Demi_espace(p, Decimal(1)) for p in points]
    hyperplan = gradient_projete(hyperplan, pas, limite, gradient_lineairement_separable, convexes)
    for point in points:
        for i in range (dim-1):
            point[i] = point[i]*point[dim-1] #on remultiplie par la classe
    return hyperplan

#renvoie le gradient dans le cas non linéairement séparable
def gradient_non_lineairement_separable(u, C, points):
    a = np.append(u[0:u.size-1], Decimal(0))
    for point in points:
        if (1 - np.dot(u, point) > -epsilon): 
            a -= C*point
    return a

#effectue une descente de gradient classique, à pas variable si souhaité
def descente_de_gradient(u0, pas, limite, gradient, points, C, pasvariable):
    u = u0
    oldU = u0
    k = 0
    first = True
    while(first or abs((np.linalg.norm(gradient(u, C, points)) - np.linalg.norm(gradient(oldU, C, points)))) > limite):
        oldU = u
        if pasvariable:
            u = u-pas*gradient(u, C, points)/(Decimal(np.power(k+1, (1/2))))
            k +=1
        else :
            u = u-pas*gradient(u, C, points)
        first = False
        print(u)
    return u

#résoud le problème SVM dans le cas non linéairement séparable
def svm_non_lineairement_separable(points, pas, C, limite, pasvariable):
    dim = points[0].size
    n = 0
    for point in points:
        print(n)
        n += 1
        for i in range (dim-1):
            point[i] = point[i]*point[dim-1] #on multiplie par la classe
    hyperplan = np.full(dim, Decimal(1))
    hyperplan = descente_de_gradient(hyperplan, pas, limite, gradient_non_lineairement_separable, points, C, pasvariable)
    convexes = [Demi_espace(p, Decimal(1)) for p in points]
    print("Dans inter : ", dans_inter(hyperplan, convexes))
    for point in points:
        for i in range (dim-1):
            point[i] = point[i]*point[dim-1] #on remultiplie par la classe
    return hyperplan

#LECTURE DE FICHIER : -------------------------------------------------------------------------------------------------------------------------------

# Fonction pour lire les données du fichier JSON
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Fonction pour extraire les parties sous forme de vecteurs
def extract_game_vectors(data):
    game_vectors = []
    for game in data:
        a = []
        a.append(int(game[1])) # Résultat de la partie
        for c in game[2]:
            a.append(c)
        game_vector = [
            int(game[0]),  # Couleur de l'IA (0 = Blanc, 1 = Noir)
            a       
        ]
        game_vectors.append(game_vector)
    return game_vectors

#------------------------------------------------------------------------------------------------------------------------------------------------------

#Résolution de notre problème initial
def detection_tricheurs(N, Z, K, b):
    debut = time.process_time()
    accuracy = []
    n2 = 0
    n = 0
    n4 = 0
    n5 = 0
    for z in range(Z):
        #Lire le fichier JSON
        file_path = 'data.txt'
        data = read_json(file_path)

        points_classes = extract_game_vectors(data)

        # Afficher les vecteurs des parties
        for i, vector in enumerate(points_classes):
            print(f"Partie {i+1}: {vector[1]}")
        random.shuffle(points_classes)
        points = [np.array(p[1]) for p in points_classes]
        
        points = points[:N]

        for p in points:    #on utilise le type de données Decimal
            for i in range(len(p)):
                p[i] = Decimal(float(p[i]))
        n = len(points)
        n2 = int(n*0.1)
        n4 = 0
        n5 = 0
        n6 = 0
        n7 = 0
        for i in range (n): #on ajoute la classe à la fin des points
            if points_classes[i][0] == 0:
                np.append(points[i], Decimal(1))
                n4 += 1
            else:
                np.append(points[i], Decimal(-1))
                n5 += 1

        pas = Decimal(1e-4)
        limite = Decimal(1e-5)
        C = Decimal(1e5)

        m = 0
        l = 0
        for i in range(n):
            a = len(points[i])
            if a > m:
                l = i
                m = a
        
        temp = points[0]
        points[0] = points[l]
        points[l] = temp

        for i in range(n):
            p = len(points[i])
            s = m - p
            for j in range(s):
                points[i] = np.append(points[i], points[i][p-1])
        
        if (b and K < m):
            m = K
            for i in range(n):
                points[i] = points[i][:K]
        
        training_set = points[:n2]
        test_set = points[n2:]

        a = svm_non_lineairement_separable(training_set, pas, C, limite, 1)
        
        k = len(test_set)
        e = 0
        for i in range(n2, n):
            if (phi(a[0:m-1],a[m-1], points[i][:-1]) >= 0):
                n6 += 1
            else:
                n7 += 1
            if ((points_classes[i][0] == 0 and phi(a[0:m-1],a[m-1], points[i][:-1]) >= 0) or (points_classes[i][0] == 1 and phi(a[:m-1],a[m-1], points[i][:-1]) <= 0)):
                e += 1
            
        accuracy.append(e/k)
    print("valeur de C :", C)
    print("pas initial :", pas)
    print("limite", limite)
    print("nombre d'éléments dans la classe 1 :", n4)
    print("nombre d'éléments dans la classe -1 :", n5)
    print("proportion de la classe 1 :", n4/(n4+n5))
    print("nombre de points utilisés pour entraîner :", n2)
    print("nombre de points utilisés pour tester :", n-n2)
    print("précision moyenne :", sum(accuracy)/len(accuracy))
    print("nombre de points classés en classe 1 :", n6)
    print("nombre de points classés en classe -1 :", n7)
    print("duree :", time.process_time() - debut)

#detection_tricheurs(500, 1, 100, 1)


#fonctions de test : ------------------------------------------------------------------------------------------------------------------------------------------------------

def test_proj_demi_espace():
    print("test proj_demi_espace : -------------------")

    convexe = Demi_espace(np.array([Decimal(1), Decimal(0)]), Decimal(0))
    x0 = np.array([Decimal(5), Decimal(-2.3)])
    print ("avant projection :", x0)
    x0 = proj_demi_espace(convexe.dir, convexe.b, x0)
    print ("après projection :", x0)

def test_dykstra_2():
    print("test Dykstra pour n = 2 : -------------------")

    convexe1 = Demi_espace(np.array([Decimal(1), Decimal(0)]), Decimal(0))
    convexe2 = Demi_espace(np.array([Decimal(0), Decimal(1)]), Decimal(0))
    convexes = [convexe1, convexe2]
    x0 = np.array([Decimal(-5), Decimal(-2.3)])
    print ("avant projection :", x0)
    x0 = dykstra_projection_2(x0, convexes)
    print ("après projection :", x0)

def test_dykstra_projection():
    print("test Dykstra pour n quelconque : ------------------")
    convexe1 = Demi_espace(np.array([Decimal(1), Decimal(0)]), Decimal(0))
    convexe2 = Demi_espace(np.array([Decimal(0), Decimal(1)]), Decimal(0))
    convexes = [convexe1, convexe2]
    x0 = np.array([Decimal(-5), Decimal(-2.3)])
    print ("avant projection :", x0)
    x0 = dykstra_projection(x0, convexes)
    print ("après projection :", x0)


def test_svm_lineairement_separable(): #l'hyperplan est symbolisé par sa normale
    print("test svm_lineairement_separable")
    debut = time.process_time()
    pas = Decimal(2e-5)
    limite = Decimal(2e-1000)
    point1 = np.array([Decimal(-1), Decimal(1), Decimal(-1)]) #(p.x, p.y, classe)
    point2 = np.array([Decimal(1), Decimal(-1), Decimal(1)])
    point3 = np.array([Decimal(2), Decimal(-2), Decimal(1)])
    point4 = np.array([Decimal(3), Decimal(-1), Decimal(1)])
    point5 = np.array([Decimal(-1), Decimal(7), Decimal(-1)])
    points = [point1, point2, point3, point4, point5]

    a = svm_lineairement_separable(points, pas, limite)

    print(a)
    print(time.process_time() - debut)

    plt.title("SVM linéairement séparable") 
    plt.xlabel("x") 
    plt.ylabel("y") 
    classe1_x = [point[0] for point in points if (point[2] == 1)]
    classe1_y = [point[1] for point in points if (point[2] == 1)]
    classe2_x = [point[0] for point in points if (point[2] == -1)]
    classe2_y = [point[1] for point in points if (point[2] == -1)]
    t = 1000000000
    plt.plot([-t*(-a[2]/a[0]), t*(-a[2]/a[0])],[-a[2]/a[1] - t*(a[2]/a[1]), -a[2]/a[1] + t*(a[2]/a[1])], marker='o', linestyle='-', color='r', label='Data Points') #affiche l'hyperplan
    plt.plot(classe1_x, classe1_y, marker='+', linestyle='None', color='b', label='Data Points') 
    plt.plot(classe2_x, classe2_y, marker='_', linestyle='None', color='b', label='Data Points')
    plt.axis([-10, 10, -10, 10])
    plt.show()

def test_svm_non_lineairement_separable():
    print("test svm_non_lineairement_separable")
    debut = time.process_time()
    pas = Decimal(1e-8)
    limite = Decimal(1e-5)
    C = Decimal(1e2)
    point1 = np.array([Decimal(-1), Decimal(1), Decimal(-1)]) #(p.x, p.y, classe)
    point2 = np.array([Decimal(1), Decimal(-1), Decimal(1)])
    point3 = np.array([Decimal(2), Decimal(-2), Decimal(1)])
    point4 = np.array([Decimal(3), Decimal(-1), Decimal(1)])
    point5 = np.array([Decimal(-1), Decimal(7), Decimal(-1)])
    point6 = np.array([Decimal(-1), Decimal(1), Decimal(1)])
    point7 = np.array([Decimal(-7), Decimal(1), Decimal(1)])
    point8 = np.array([Decimal(2.5), Decimal(5.2), Decimal(1)])
    point9 = np.array([Decimal(0), Decimal(-2.5), Decimal(-1)])
    points = [point1, point2, point3, point4, point5, point6, point7, point8, point9]

    a = svm_non_lineairement_separable(points, pas, C, limite, 0)
    a = a/np.linalg.norm(a)

    print(a)
    print(time.process_time() - debut)

    plt.title("SVM non linéairement séparable") 
    plt.xlabel("x") 
    plt.ylabel("y") 
    classe1_x = [point[0] for point in points if (point[2] == 1)]
    classe1_y = [point[1] for point in points if (point[2] == 1)]
    classe2_x = [point[0] for point in points if (point[2] == -1)]
    classe2_y = [point[1] for point in points if (point[2] == -1)]
    t = 1000
    a1 = [-t*(-a[2]/a[0]), -a[2]/a[1] - t*(a[2]/a[1])]
    a2 = [t*(-a[2]/a[0]), -a[2]/a[1] + t*(a[2]/a[1])]
    print("point1 :", a1)
    print("point2 :", a2)
    plt.plot([-t*(-a[2]/a[0]), t*(-a[2]/a[0])],[-a[2]/a[1] - t*(a[2]/a[1]), -a[2]/a[1] + t*(a[2]/a[1])], marker='o', linestyle='-', color='r', label='Data Points')
    plt.plot(classe1_x, classe1_y, marker='+', linestyle='None', color='b', label='Data Points') 
    plt.plot(classe2_x, classe2_y, marker='_', linestyle='None', color='b', label='Data Points')
    plt.axis([-10, 10, -10, 10])
    plt.show()
    print("devrait être < 0 :", phi(a[:(len(a)-1)], a[len(a)-1], point1[:2]))
    print("devrait être > 0 :", phi(a[:(len(a)-1)], a[len(a)-1], point2[:2]))