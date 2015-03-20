import multiprocessing

from sklearn.decomposition import PCA
from sklearn import svm
import pandas as pd
import numpy as np


class BaseDados:
    def __init__(self, nome, dados, teste):
        self.nome = nome
        self.dados = dados
        self.teste = teste


class Main:
    numMaxComponentesPCA = 10
    numFolds = 10
    out_q = multiprocessing.Queue()

    def normaliza(self, dados):

        toreturn = (dados - dados.mean())/dados.std()
        return toreturn




    def run(self, pSVMGama=0.01, pSVMC=100.):

        base = self.carregaContextosDF()


        treino = self.normaliza(base.dados.as_matrix(base.dados.columns[1:]))
        treino = np.nan_to_num(treino)
        teste = self.normaliza(base.teste.as_matrix(base.teste.columns[1:]))
        teste = np.nan_to_num(teste)
        classesTreino = base.dados['CLASSE']
        classesTeste = base.teste['CLASSE']

        print treino



        for numComponentesPCA in range(1, self.numMaxComponentesPCA + 1):

            classifier = svm.SVC(gamma=pSVMGama, C=pSVMC)

            processos = []
            resultados = []


            modeloPCA = self.calculaTransformacaoPCADF(base.dados, numComponentesPCA)

            treinoPCA = modeloPCA.transform(treino)
            testePCA = modeloPCA.transform(teste)

            self.classificaSVM(classifier, processos, treinoPCA, classesTreino, testePCA, classesTeste, self.out_q)

            for p in processos:
                p.join()
                resultados.append(self.out_q.get())

                    # for i in range(self.numFolds):
                    # # appenda os resultados obtidos para cada fold para futura media
                    # resultados.append(self.out_q.get())

            results = []
            for result in resultados:
                results.append(result)

            accuracy = sum(results) / len(results)


            # Montar uma tabela e grafico com o nome da base, numero de componentes, acuracia media
            print numComponentesPCA, "  ", accuracy



        # for base in self.bases:
        #     for numComponentesPCA in range(1, self.numMaxComponentesPCA + 1):
        #         for fold in range(1, self.numFolds + 1):
        #             conjuntos = self.separaConjuntosDF(fold, base.dados)
        #
        #             modeloPCA, treinoPCA = self.calculaTransformacaoPCADF(conjuntos['treino'], numComponentesPCA)
        #
        #             #Com o modelo retornado, e possivel aplicar o modelo treinado no conjunto de testes.
        #             X = conjuntos['teste'].as_matrix(conjuntos['teste'].columns[0:-2])
        #             testePCA = modeloPCA.transform(X)
        #
        #             self.classificaSVM(treinoPCA, conjuntos['treino'])
        #             self.classificaSVM(testePCA, conjuntos['teste'])


    def get_classifier_accuracy(self, classifier, treinoPCA, classesTreino, testePCA, classesTeste, out_score):
        # realiza a classificacao e determina o percentual em cima dos dados de teste
        score = classifier.fit(treinoPCA, classesTreino).score(testePCA, classesTeste)

        # pega os resultados
        out_score.put(score)
        return


    def classificaSVM(self, classifier, processos, treinoPCA, classesTreino, testePCA, classesTeste, out_q):
        p = multiprocessing.Process(target=Main.get_classifier_accuracy, args=(self, classifier, treinoPCA, classesTreino, testePCA, classesTeste, out_q))
        p.start()
        processos.append(p)

        # Tirei aki pois ele bloqueia a thread
        # return out_q.get()

    def carregaContextosDF(self):
        dadosTreino = pd.read_csv('../dados/segmentation.data')
        dadosTeste = pd.read_csv('../dados/segmentation.test')
        return BaseDados("segmentos", dadosTreino, dadosTeste)

    def calculaTransformacaoPCADF(self, dados, numComponentes):
        pca = PCA(n_components=numComponentes)
        X = dados.as_matrix(dados.columns[1:])
        # A linha a seguir ja transforma o X em PCA, nos precisamos do modelo
        # Agora nos temos o modelo sendo retornado
        pca.fit(X)
        return pca


if __name__ == '__main__':
    Main().run()







