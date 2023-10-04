# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:49:27 2023

@author: Vinicius
"""

# Algoritmo Genético Funcional

import math
import numpy as np
from numpy import nan
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import xlwt
from xlwt import Workbook
from scipy.optimize import curve_fit

pi = math.pi

random.seed(456543)
np.random.seed(456543)


#Matriz Desenvolvimento 1314
dfIM = pd.read_excel(r'C:\Users\Vinicius\Documents\Vinícius\Faculdade\Iniciação Científica\Dados\Automata\Croquis\Remanejados\IM Roll 1.xlsx')
p1IM = dfIM['p1'].values.tolist()
p2IM = dfIM['p2'].values.tolist()
p3IM = dfIM['p3'].values.tolist()
p4IM = dfIM['p4'].values.tolist()
p5IM = dfIM['p5'].values.tolist()
p6IM = dfIM['p6'].values.tolist()
p7IM = dfIM['p7'].values.tolist()
p8IM = dfIM['p8'].values.tolist()
p9IM = dfIM['p9'].values.tolist()
p10IM = dfIM['p10'].values.tolist()
matIM = np.c_[p1IM, p2IM, p3IM, p4IM, p5IM, p6IM, p7IM, p8IM, p9IM, p10IM]

#Matriz Adultos 1314
dfAD = pd.read_excel(r'C:\Users\Vinicius\Documents\Vinícius\Faculdade\Iniciação Científica\Dados\Automata\Croquis\Remanejados\AD Roll 1.xlsx')    
p1AD = dfAD['p1'].values.tolist()
p2AD = dfAD['p2'].values.tolist()
p3AD = dfAD['p3'].values.tolist()
p4AD = dfAD['p4'].values.tolist()
p5AD = dfAD['p5'].values.tolist()
p6AD = dfAD['p6'].values.tolist()
p7AD = dfAD['p7'].values.tolist()
p8AD = dfAD['p8'].values.tolist()
p9AD = dfAD['p9'].values.tolist()
p10AD = dfAD['p10'].values.tolist()
matAD = np.c_[p1AD, p2AD, p3AD, p4AD, p5AD, p6AD, p7AD, p8AD, p9AD, p10AD]



#no desprop
def matriz_inicial(matrix):
    matriz_in = np.copy(matrix)
    
    return matriz_in


def DistProp(matrix,funcprop):
    B = np.zeros((100,100))
    desprop = funcprop(matrix)
    nova = np.copy(B)
    
    k0 = 0            
    k1 = 0
    k2 = 0
    k3 = 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10        
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10        
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10        
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10        
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    
    return nova
    

IMprop = DistProp(matIM, matriz_inicial)
ADprop = DistProp(matAD, matriz_inicial)




def CI_adulto(matrix):
    CI_AD = np.copy(matrix)
    n = len(CI_AD)
    for i in range(0, n):
        for j in range(0, n):
            if CI_AD[i][j] == 2:
                distAD = random.randrange(0,100)
                if distAD <= 30:
                    CI_AD[i][j] = 1
    return CI_AD
    
    
CI_Adulto = CI_adulto(ADprop)

    
def check_contorno(matrix, index_x, index_y, vizinho):
    if index_x + vizinho[0] < matrix.shape[0] and index_y + vizinho[1] < matrix.shape[1]:
        return True


def vizinhos_vazios_am(matrix, index_x, index_y):
    lista_true_vizinhos_ovp = []
    
    lista_de_duplas = [[-1,-1] , [-1,0] , [-1, +1], [0,-1] , [0, +1] , [+1, -1] , [+1, 0] , [+1,+1]]

    for item in lista_de_duplas:
        contorno_ = check_contorno(matrix, index_x, index_y, item)
        
        if contorno_:
            if matrix[index_x + item[0]][index_y + item[1]] == 0:
                lista_true_vizinhos_ovp.append(item)
    
    return lista_true_vizinhos_ovp


def vizinhos_vazios_ovp(matrix, index_x, index_y):
    lista_true_vizinhos_ovp = []
    
    lista_de_duplas = [[-1,-1] , [-1,0] , [-1, +1], [0,-1] , [0, +1] , [+1, -1] , [+1, 0] , [+1,+1]]

    for item in lista_de_duplas:
        contorno_ = check_contorno(matrix, index_x, index_y, item)
        
        if contorno_:
            if matrix[index_x + item[0]][index_y + item[1]] == 0:
                lista_true_vizinhos_ovp.append(item)
    
    return lista_true_vizinhos_ovp
    

def vizinhos_vazios(matrix, index_x, index_y):
    lista_true_vizinhos = []
    
    #lista_de_duplas = [[-3,-3] , [-3,0] , [-3, +3], [0,-3] , [0, +3] , [+3, -3] , [+3, 0] , [+3,+3]]
    lista_de_duplas = [[-2,-2] , [-2,-1], [-2,0], [-2,1], [-2,+2], [-1,2], [0,2], [1,2], [2,2], [2,1], [2,0], [2,-1], [2,-2] , [1,-2], [0,-2], [-1,-2], [-1,-1], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1] , [0,-1]]

    for item in lista_de_duplas:
        contorno_ = check_contorno(matrix, index_x, index_y, item)
        
        if contorno_:
            if matrix[index_x + item[0]][index_y + item[1]] == 0:
                lista_true_vizinhos.append(item)
    
    return lista_true_vizinhos


df_gss = pd.read_excel(r'C:\Users\Vinicius\Documents\Vinícius\Faculdade\Iniciação Científica\Dados\Boxplots\Boxplot AC\Gaussiana\gaussIM.xlsx')

meses = df_gss['Meses'].values.tolist()
coleta = df_gss['Coleta'].values.tolist()
x = meses
y = coleta

def gauss_f(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2 / sigma**2)

popt, pcov = curve_fit(gauss_f, x , y)

A_opt, mu_opt, sigma_opt = popt
x_model = np.linspace(min(x), max(x), 365)
y_model = gauss_f(x_model, A_opt, mu_opt, sigma_opt)

max_y = max(y_model)
y_model_norm = []

for i in y_model:
    j = i/max_y
    y_model_norm.append(j)

norm_crr_1 = []
for i in y_model_norm:
    j = i - 0.369
    norm_crr_1.append(j)

norm_crr_2 = []
for i in norm_crr_1:
    norm_crr_2.append(i)
del norm_crr_2[205:]
for i in range(205,365):
    norm_crr_2.append(0)
    
y_model_ref = []
for i in norm_crr_2:
    y_model_ref.append(i)
del y_model_ref[365:]
for i in norm_crr_2:
    y_model_ref.append(i)
    

X = np.arange(0,730)

X_shift = []
for i in X:
    j = i + 0
    X_shift.append(j)



#plt.plot(X, y_model_ref, color = 'r', label = 'oviposição')
#plt.plot(X_shift, y_model_ref, color = 'b', label = 'mortalidade imaturos')
#plt.gca().legend(('oviposição','mort. imaturos'))
#plt.title('Curvas Sobrepostas')
#plt.show()


def função_sazonal_ovp(frames, phi): #, taxa_ovp):
    #taxa_ovp = 0.4
    Y = y_model_ref
    nova_taxa_ovp = phi * Y[frames]
    #print('ovp = ' , Y[frames])
    
    return nova_taxa_ovp


def função_sazonal_mort(frames, mu):  #, taxa_mortalidade_IM):
    #taxa_mortalidade_IM = 0.2
    if frames >= 719:
        nova_taxa_mort = 0
        #print('m = ' , nova_taxa_mort)
    else:
        Y = y_model_ref
        nova_taxa_mort = mu * Y[frames]
        #print('m = ' , nova_taxa_mort)

    return nova_taxa_mort


def matriz_borda(matrix1, matrix2, frames, phi, mu, sigma, gamma, qsi):


    N = 102
    M = 102
    borda = 2
    bordaL = N - borda
    bordaC = M - borda
    
    matbordaIM = np.zeros((N, M))
    matbordaAD = np.zeros((N, M))
    for i in range(borda,bordaL):
        for j in range(borda,bordaC):
            matbordaIM[i][j] = matrix1[int(i)][int(j)]
            matbordaAD[i][j] = matrix2[int(i)][int(j)]
            
            northIM = matbordaIM[i][j-1] if j > 0 else 0
            southIM = matbordaIM[i][j+1] if j < (np.size(matbordaIM,1)-2) else 0
            westIM = matbordaIM[i+1][j] if i < (np.size(matbordaIM,0)-2) else 0
            eastIM = matbordaIM[i-1][j] if i > 0 else 0
            seIM = matbordaIM[i+1][j+1] if i < (np.size(matbordaIM,0)-2) and j < (np.size(matbordaIM,1)-2) else 0
            swIM = matbordaIM[i+1][j-1] if i < (np.size(matbordaIM,0)-2) and j > 0 else 0
            neIM = matbordaIM[i-1][j+1] if i > 0 and j < (np.size(matbordaIM,1)-2) else 0
            nwIM = matbordaIM[i-1][j-1] if i > 0  and j > 0 else 0
            neighboursIM = np.sum([northIM, southIM, westIM, eastIM, seIM, swIM, neIM, nwIM])
            
            cellAD = matbordaAD[i][j] #if j > 0 else 0
            northAD = matbordaAD[i][j-1] #if j > 0 else 0
            southAD = matbordaAD[i][j+1] #if j < (np.size(matbordaAD,1)-2) else 0
            westAD = matbordaAD[i+1][j] #if i < (np.size(matbordaAD,0)-2) else 0
            eastAD = matbordaAD[i-1][j] #if i > 0 else 0
            seAD = matbordaAD[i+1][j+1] #if i < (np.size(matbordaAD,0)-2) and j < (np.size(matbordaAD,1)-2) else 0
            swAD = matbordaAD[i+1][j-1] #if i < (np.size(matbordaAD,0)-2) and j > 0 else 0
            neAD = matbordaAD[i-1][j+1] #if i > 0 and j < (np.size(matbordaAD,1)-2) else 0
            nwAD = matbordaAD[i-1][j-1] #if i > 0  and j > 0 else 0
            
            R2_1 = matbordaAD[i-2][j-2]
            R2_2 = matbordaAD[i-2][j-1]
            R2_3 = matbordaAD[i-2][j]
            R2_4 = matbordaAD[i-2][j+1]
            R2_5 = matbordaAD[i-2][j+2]
            R2_6 = matbordaAD[i-1][j+2]
            R2_7 = matbordaAD[i][j+2]
            R2_8 = matbordaAD[i+1][j+2]
            R2_9 = matbordaAD[i+2][j+2]
            R2_10 = matbordaAD[i+2][j+1]
            R2_11 = matbordaAD[i+2][j]
            R2_12 = matbordaAD[i+2][j-1]
            R2_13 = matbordaAD[i+2][j-2]
            R2_14 = matbordaAD[i+1][j-2]
            R2_15 = matbordaAD[i][j-2]
            R2_16 = matbordaAD[i-1][j-2]
            
            neighboursAD = np.sum([cellAD, northAD, southAD, westAD, eastAD, seAD, swAD, neAD, nwAD, R2_1, R2_2, R2_3, R2_4, R2_5, R2_6, R2_7, R2_8, R2_9, R2_10, R2_11, R2_12, R2_13, R2_14, R2_15, R2_16])
            
                        
            # oviposição dos adultos:

            if neighboursAD >= 4:
                index_x = i
                index_y = j
                vizinhos_possiveis_ovp = vizinhos_vazios_ovp(matrix1, index_x, index_y)
                num_vizinhos_ovp = len(vizinhos_possiveis_ovp)
                
                if num_vizinhos_ovp > 0:
                    numero_aleatorio_ovp = random.randrange(0, num_vizinhos_ovp) #np.random.choice(np.arange(num_vizinhos_ovp))
                    vizinho_escolhido_ovp = vizinhos_possiveis_ovp[numero_aleatorio_ovp]
                    ovp_sim = função_sazonal_ovp(frames, phi)  # função_sazonal_ovp(frames , taxa_ovp)
                    ovp_nao = 1 - ovp_sim
                    #print('oviposição =' , ovp_sim)
                    opcoes = ['SIM' , 'NAO']
                    
                    resposta = np.random.choice(opcoes , p = [ovp_sim , ovp_nao])
                    
                    if resposta == 'SIM': 
                        matbordaIM[index_x + vizinho_escolhido_ovp[0]][index_y + vizinho_escolhido_ovp[1]] = 2
                        neighboursAD = 0                        
                            

            # mortalidade dos imaturos: 0,03373      min: 0.0228    max: 0.0633   
            if matbordaIM[i][j] == 2:
                mort_sim = função_sazonal_mort(frames, mu)          #função_sazonal_mort(frames, taxa_mortalidade_IM)                 # mortIM = random.uniform(0 , 1)
                mort_nao = 1 - mort_sim
                opcoes = ['SIM' , 'NAO']
                resposta = np.random.choice(opcoes , p = [mort_sim , mort_nao])
                if resposta == 'SIM': 
                    matbordaIM[i][j] = 0
                    
                    
            # emergência dos imaturos: 0,01834      
            if matbordaIM[i][j] == 2:
                vizinhos_possiveis_am = vizinhos_vazios_am(matrix2, i , j)
                num_vizinhos_am = len(vizinhos_possiveis_am)                
                if num_vizinhos_am > 0:
                    numero_aleatorio = np.random.choice(np.arange(num_vizinhos_am))
                    vizinho_escolhido_am = vizinhos_possiveis_am[numero_aleatorio]
                    emIM = random.uniform(0 , 1)
                    if emIM < sigma:           # taxa_emergencia:
                        #print('amadurece' , emIM)
                        matbordaIM[i][j] = 0
                        matbordaAD[i + vizinho_escolhido_am[0]][j + vizinho_escolhido_am[1]] = 1
                #else:
                  #  print('nao amadurece' , emIM)


            # mortalidade dos adultos: 0,00840      min: 0.0034      max: 0.0166 
            if matbordaAD[i][j] == 2 or matbordaAD[i][j] == 1:
                mortAD = random.uniform(0 , 1)
                #print('mortad' , mortAD)
                if mortAD < gamma:                # taxa_mortalidade_AD:
                    matbordaAD[i][j] = 0

            
            # amadurecimento dos adultos: 0,03571     min: 0.0238    max: 0.0714 
            if matbordaAD[i][j] == 1:
                amAD = random.uniform(0 , 1)
                #print('amAD' , amAD)
                if amAD < qsi:         # taxa_amadurecimento:
                    matbordaAD[i][j] = 2

                    

                    
    matbordaIM[0] = 0
    matbordaIM[1] = 0
    matbordaIM[-1] = 0
    matbordaIM[-2] = 0        
    matbordaIM[:,0] = 0
    matbordaIM[:,1] = 0
    matbordaIM[:,-1] = 0
    matbordaIM[:,-2] = 0
    
    matbordaAD[0] = 0
    matbordaAD[1] = 0
    matbordaAD[-1] = 0
    matbordaAD[-2] = 0        
    matbordaAD[:,0] = 0
    matbordaAD[:,1] = 0
    matbordaAD[:,-1] = 0
    matbordaAD[:,-2] = 0        
            
    
    return matbordaIM, matbordaAD



IM, AD = matriz_borda(IMprop , CI_Adulto, 0, 0.4, 0.2, 0.0357, 0.0034, 0.0857)


def anima(frames, matrix1, matrix2, X_0, X_1, X_2, X_3, X_4): #, X_1, X_2, X_3, X_4):
    c = 0
    contador = []
    for j in range(0,26):
        contador.append(c)
        c = c + 14
    #print(len(contador))
    #print(contador)    
    n = len(contador)

    quant_IM_S = []
    quant_AD_S = []
    

    for i in np.arange(frames):
        print('i' , i)        
        matrix1, matrix2 = matriz_borda(matrix1, matrix2, i, X_0, X_1, X_2, X_3, X_4) # , X_1, X_2, X_3, X_4)
      
        for j in range(0,n):
            if i == contador[j]:
                quant_IM_S.append(matrix1.sum())
                quant_AD_S.append(matrix2.sum())
                #print(quant_IM_S, quant_AD_S)



    print('X0 (phi) =' , X_0)
    print('X1 (mu) =' , X_1)
    print('X2 (sigma) =' , X_2)
    print('X3 (gamma) =' , X_3)
    print('X4 (qsi) =' , X_4)
    
    return quant_IM_S, quant_AD_S

#anima(30, IM, AD)

frames_x = 56

def f_mean(X):
    df_IM = pd.read_excel(r'C:\Users\Vinicius\Documents\Vinícius\Faculdade\Iniciação Científica\Dados\Algoritmo Genético\Quinzenais\Primeira Rodada\Totais\BP IM 1314 tot.xlsx')
    df_AD = pd.read_excel(r'C:\Users\Vinicius\Documents\Vinícius\Faculdade\Iniciação Científica\Dados\Algoritmo Genético\Quinzenais\Primeira Rodada\Totais\BP AD 1314 tot.xlsx')
    df_q_IM = df_IM['Totais'][:4].values.tolist()
    df_q_AD = df_AD['Totais'][:4].values.tolist()

     
    q_IM, q_AD = anima(frames_x, IM, AD, X[0], X[1], X[2], X[3], X[4])
    

    #print('X0 fmean' , X[0])


    print('qIM, qAD =' , q_IM , q_AD)
    
    #print('lens' , len(q_df_IM) , len(q_df_AD) , len(q_IM) , len(q_AD))

        
    fitness_list = []
    
    
    n = len(df_q_IM)
    for i in range(0,n):
        fitness = np.sum(df_q_IM[i] - q_IM[i])**2 + np.sum(df_q_AD[i] - q_AD[i])**2
        fitness_list.append(fitness)
    
    print('fitness_list = ' , fitness_list)
    
    fitness_sum = np.sum(fitness_list)
    print('fitness_sum = ' , fitness_sum)
    
    return fitness_sum

    

varbounds = np.array([[0.1, 0.5],      # Taxa de oviposição (φ)
                     [0.06, 0.4],      # Mortalidade dos imaturos (μ)
                     [0.01, 0.06],     # Emergência dos imaturos (σ)
                     [0.003, 0.02],    # Mortalidade dos adultos (γ)
                     [0.02, 0.09]])    # Amadurecimento dos adultos (ξ)



def ag_run(f):
    
    algorithm_param = {'max_num_iteration': 10,\
                    'population_size':10,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}

    
    var_holder = {}
    sol_holder = {}
    n_modelos = 2
    
    for i in range(n_modelos):
        var_holder['model_' + str(i)] = ga(function=f,dimension=5,variable_type='real',variable_boundaries=varbounds, algorithm_parameters=algorithm_param, function_timeout = 60)
    locals().update(var_holder)

    for i in (var_holder):
        print('solution' , i)
        var_holder[i].run()
        print('-' * 60)
        
    for i in (var_holder):
        sol_holder['solution_' + str(i)] = var_holder[i].best_variable
    locals().update(sol_holder)
        
    for i in sol_holder:
        return sol_holder[i]
        
        
        
        
        
    #model=ga(function=f,dimension=5,variable_type='real',variable_boundaries=varbounds, algorithm_parameters=algorithm_param, function_timeout = 40)    
    #model.run()
    #solution = model.best_variable

    #return solution

ag_run(f_mean)










