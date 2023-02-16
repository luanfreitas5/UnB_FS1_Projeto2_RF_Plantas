# Nome: Luan Mendes Gonçalves Freitas
# Matricula: 15/0015585
# Disciplina: Fundamentos de Sistemas Inteligentes
# Projeto 2 - Florestas Randômicas
# Módulo floresta

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import confusion_matrix
        

# Classe do Algoritmo de Floresta Randômica
class FlorestaRandomica ():
    
    def __init__ (self):
        self.caracteristicas = None
        self.rotulos = None
        self.colunas = None
        self.folhas = None
        
    def leitura(self):
        self.colunas = ['Class', 'Specimen Number', 'Eccentricity', 'Aspect Ratio',
                      'Elongation', 'Solidity', 'Stochastic Convexity', 'Isoperimetric Factor',
                      'Maximal Indentation Depth', 'Lobedness', 'Average Intensity', 'Average Contrast',
                      'Smoothness', 'Third moment', 'Uniformity', 'Entropy']
        
        self.folhas = pd.read_csv("leaf.csv", names=self.colunas, header=None)
        
        self.rotulos = self.folhas.pop('Class').values 
        self.caracteristicas = self.folhas.values
        
    def processamento(self):
        
        self.leitura()
        floresta = RandomForestClassifier(n_estimators=50)
        self.modelo = floresta.fit(self.caracteristicas, self.rotulos)
        
        self.predicao1 = cross_val_predict(floresta, self.caracteristicas, self.rotulos, cv=10)
        self.score1 = cross_val_score(floresta, self.caracteristicas, self.rotulos, cv=10)
        
        self.stratifiedKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        self.predicao2 = cross_val_predict(floresta, self.caracteristicas, self.rotulos, cv=self.stratifiedKFold)
        self.score2 = cross_val_score(floresta, self.caracteristicas, self.rotulos, cv=self.stratifiedKFold)
            
        self.leaveOneOut = LeaveOneOut()
        self.predicao3 = cross_val_predict(floresta, self.caracteristicas, self.rotulos, cv=self.leaveOneOut)
        self.score3 = cross_val_score(floresta, self.caracteristicas, self.rotulos, cv=self.leaveOneOut)
           
        self.kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        self.predicao4 = cross_val_predict(floresta, self.caracteristicas, self.rotulos, cv=self.kfold)
        self.score4 = cross_val_score(floresta, self.caracteristicas, self.rotulos, cv=self.kfold)
        
        self.impressao(floresta)
      
      
    def impressao(self, floresta):  
        os.system('clear') or None
        
        print("\n\nProjeto 2 Floresta Randômica\n")
      
        print("Floresta Randômica score medio 1: ", str(self.score1.mean() * 100) + ' %')
        print("Floresta Randômica Acurácia (precisão média) 1: ", str(accuracy_score(self.rotulos, self.predicao1) * 100) + ' %')
        print("\n")
        
        print("stratifiedKFold score medio 2: ", str(self.score2.mean() * 100) + ' %')
        print("stratifiedKFold Acurácia (precisão média) 2: ", str(accuracy_score(self.rotulos, self.predicao2) * 100) + ' %')         
        print("\n")
            
        print("leave-one-out score medio 3: ", str(self.score3.mean() * 100) + ' %')          
        print("leave-one-out Acurácia (precisão média) 3: ", str(accuracy_score(self.rotulos, self.predicao3) * 100) + ' %')
        print("\n")
           
        print("KFold score medio 4: ", str(self.score4.mean() * 100) + ' %')
        print("KFold Acurácia (precisão média) 4: ", str(accuracy_score(self.rotulos, self.predicao4) * 100) + ' %')         
        print("\n")
    
        feat_importances = pd.Series(floresta.feature_importances_, index=self.folhas.columns)
        feat_importances.nlargest(16).plot(kind='barh', title='Floresta Randômica - Importancia dos Atributos')
        plt.show()
         
        matrizConfusao = confusion_matrix(self.rotulos, self.predicao1)
        plt.matshow(matrizConfusao)
        plt.ylabel('Caracteristicas - Conhecida')
        plt.xlabel('Predição')
        plt.title('Floresta Randômica Matriz de Confusão')
        plt.colorbar()
        plt.show()
           
        matrizConfusao2 = confusion_matrix(self.rotulos, self.predicao2)
        plt.matshow(matrizConfusao2)
        plt.ylabel('Caracteristicas - Conhecida')
        plt.xlabel('Predição')
        plt.title('stratifiedKFold Matriz de Confusão')
        plt.colorbar()
        plt.show()
            
        matrizConfusao3 = confusion_matrix(self.rotulos, self.predicao3)
        plt.matshow(matrizConfusao3)
        plt.ylabel('Caracteristicas - Conhecida')
        plt.xlabel('Predição')
        plt.title('leave-one-out Matriz de Confusão')
        plt.colorbar()
        plt.show()
        
        matrizConfusao4 = confusion_matrix(self.rotulos, self.predicao4)
        plt.matshow(matrizConfusao4)
        plt.ylabel('Caracteristicas - Conhecida')
        plt.xlabel('Predição')
        plt.title('KFold Matriz de Confusão')
        plt.colorbar()
        plt.show()

