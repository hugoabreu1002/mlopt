from numpy.random import choice
from scipy import spatial
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm
import numpy as np
import sys

class Formiga:
    def __init__(self, ponto_atual):
        self.ponto_atual = ponto_atual
        self.rota = [ponto_atual]

    def andar(self, ponto):
        self.ponto_atual = ponto
        self.rota.append(ponto)

class Ponto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Caminho:
    def __init__(self, ponto_i, ponto_j):
        self.ponto_i = ponto_i
        self.ponto_j = ponto_j
        self.comprimento = math.sqrt((ponto_i.x - ponto_j.x)**2 + (ponto_i.y - ponto_j.y)**2)
        self.feromonio = 0
        self.formigas_passantes = []

    def contem(self, formiga, qtd_pontos):
        if self.ponto_i == formiga.ponto_atual:
            return self.ponto_j not in formiga.rota or (len(formiga.rota) == qtd_pontos and self.ponto_j == formiga.rota[0])
        elif self.ponto_j == formiga.ponto_atual:
            return self.ponto_i not in formiga.rota or (len(formiga.rota) == qtd_pontos and self.ponto_i == formiga.rota[0])
        else:
            return False

    def ponto_adjacente(self, ponto):
        if self.ponto_i == ponto:
            return self.ponto_j
        elif self.ponto_j == ponto:
            return self.ponto_i
        else:
            return None

class Grafo:
    def __init__(self, caminhos):
        self.caminhos = caminhos
        self.melhor_rota = []

    def possiveis_caminhos(self, formiga, qtd_pontos):
        return [caminho for caminho in self.caminhos if caminho.contem(formiga, qtd_pontos)]

class ACO_Graph:
    """
    pontos_list - is a list of lists. where each lists is a point in R2 space.
    """
    def __init__(self, pontos_list, alpha, beta, rho):
        self.qtd_pontos = len(pontos_list)
        self.pontos = list(map(lambda p: Ponto(p[0],p[1]), pontos_list))
        self.caminhos = self.init_caminhos()
        self.grafo = Grafo(self.caminhos)
        self.alfa = alpha
        self.beta = beta
        self.rho = rho
        self.ants_number = None
        self.historico = []
        self.melhor_rota = None

    def init_caminhos(self):
        # criando os caminhos
        caminhos = []

        for i in range(self.qtd_pontos - 1):
            ponto_atual = self.pontos[i]
            pontos_para_conectar = self.pontos[i + 1:]

            for ponto_para_conectar in pontos_para_conectar:
                caminhos.append(Caminho(ponto_atual, ponto_para_conectar))

        return caminhos

    def plotGraphs(self, fig_size=(20,10), text_size=14, marker='o', points_color='r', line_colors='k', line_marker='>'):
        plt.figure(figsize=fig_size)

        for ponto in self.pontos:
            plt.plot(ponto.x, ponto.y, marker=marker, color=points_color)

        x = []
        y = []

        for caminho in self.caminhos:
            x_i = caminho.ponto_i.x
            x_j = caminho.ponto_j.x
            y_i = caminho.ponto_i.y
            y_j = caminho.ponto_j.y

            x_texto = (x_i + x_j) / 2
            y_texto = (y_i + y_j) / 2

            plt.text(x_texto, y_texto, s="{:.2f}".format(caminho.comprimento), fontdict={'size':text_size})

            x.append(x_i)
            x.append(x_j)
            y.append(y_i)
            y.append(y_j)
        
        plt.plot(x, y, color=line_colors, marker=line_marker)

        plt.show()

    def inicializar_colonia(self, ants_number):
        formigas = []

        for _ in range(ants_number):
            formigas.append(Formiga(random.choice(self.pontos)))

        return formigas

    def escolher_caminho(self,possiveis_caminhos):
        denominador = sum([(caminho.feromonio)**self.alfa * (1 / caminho.comprimento)**self.beta for caminho in possiveis_caminhos])
        distribuicao_probabilidades = None

        if denominador == 0:
            distribuicao_probabilidades = [1 / len(possiveis_caminhos)  for _ in possiveis_caminhos]
        else:
            distribuicao_probabilidades = [((caminho.feromonio)**self.alfa * (1 / caminho.comprimento)**self.beta) / denominador for caminho in possiveis_caminhos]

        return choice(possiveis_caminhos, 1, p=distribuicao_probabilidades)[0]

    def distancia_rota(self,rota):
        distancia_rota = 0

        for i in range(0, len(rota) - 1):
            distancia = math.sqrt((rota[i].x - rota[i + 1].x)**2 + (rota[i].y - rota[i + 1].y)**2)
            distancia_rota += distancia

        return distancia_rota

    def atualizar_feromonios(self, caminhos, method, tal_saturation):
        if method == 'aco':
            for caminho in caminhos:
                delta_tau = sum([1 / self.distancia_rota(formiga.rota) for formiga in caminho.formigas_passantes])
                caminho.feromonio = (1 - self.rho) * caminho.feromonio + delta_tau
                caminho.formigas_passantes = []
        
        elif method == 'max_min':
            if tal_saturation == None:
                mean_feromonio = np.mean([caminho.feromonio for caminho in caminhos])
                tal_saturation = [0.2*mean_feromonio, 1.25*mean_feromonio]
            
            for caminho in caminhos:
                lista_delta_tau = [1 / self.distancia_rota(formiga.rota) for formiga in caminho.formigas_passantes]
                if len(lista_delta_tau) > 0:
                    delta_tau = max(lista_delta_tau)*len(caminho.formigas_passantes) #essa soma vale para todas as formigas que passam nesse caminho
                else:
                    delta_tau = 0                    
                
                # mesma coisa vale para a saturacao, ela eh para a soma toda das formigas no caminho
                caminho.feromonio += np.clip((1 - self.rho) * caminho.feromonio + delta_tau, a_min=min(tal_saturation)*len(caminho.formigas_passantes), a_max=max(tal_saturation)*len(caminho.formigas_passantes))
                caminho.formigas_passantes = []
        else:
            raise Exception("method - chose insert 'aco' to use canonical ACO. Insert 'max_min' to use max_min ant system method")

    def movimentar_formiga(self, formiga, grafo):
        while True:
            possiveis_caminhos = grafo.possiveis_caminhos(formiga, self.qtd_pontos)

            if possiveis_caminhos == []:
                break

            caminho_escolhido = self.escolher_caminho(possiveis_caminhos)
            caminho_escolhido.formigas_passantes.append(formiga)
            formiga.andar(caminho_escolhido.ponto_adjacente(formiga.ponto_atual))


    def search(self, ants_number, iteracoes, plot_at_every = 100, method = 'aco', tal_saturation=None):
        """
        method - chose insert 'aco' to use canonical ACO. Insert 'max_min' to use max_min ant system method
        tal_saturation - if using 'max_min' method, you should choose saturation limits for the pheromone. If no limits are defined, the algorithm will use 1.25*(mean_tal + 1), 
        where mean_tal is the mean pheromone addition for the current iteration and the paths already found.
        
        plot_at_every - chose number of iterations to plot the best result everytime it passes. Chose None if no plots wanted.
        """
        self.ants_number = ants_number
        distancia_melhor_rota = sys.maxsize
        plot_follow = plot_at_every

        for it in tqdm(range(iteracoes)):
            formigas = self.inicializar_colonia(self.ants_number)

            for formiga in formigas:
                self.movimentar_formiga(formiga, self.grafo)

                if self.distancia_rota(formiga.rota) < distancia_melhor_rota:
                    self.melhor_rota = formiga.rota
                    distancia_melhor_rota = self.distancia_rota(formiga.rota)

            self.atualizar_feromonios(self.grafo.caminhos, method, tal_saturation)

            if plot_at_every == None:
                pass
            elif it >= plot_follow:
                plot_follow += plot_at_every
                # mostrando a melhor rota a cada iteracao
                print("######## iteracao {0} ##########".format(it))
                for ponto in self.pontos:
                    plt.plot(ponto.x, ponto.y, marker='o', color='r')

                x = []
                y = []

                for ponto in self.melhor_rota:
                    x.append(ponto.x)
                    y.append(ponto.y)

                plt.plot(x, y, color='y')

                plt.show()
                print("Best Route Distance: {:.2f}".format(distancia_melhor_rota))

        return self.melhor_rota, distancia_melhor_rota
