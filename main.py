import perceptron as ai
import random

if __name__ == '':
    # META PARAMETROS
    layers = [64, 32, 10]
    ratio_aprendizaje = 0.1
    tamanho_batch = 10
    iteraciones = 50

    per = ai.Perceptron(layers, ratio_aprendizaje)
    dataset = ai.load_dataset('dataset.csv', False, True)
    for epoch in range(iteraciones):
        random.shuffle(dataset)
        per.entrenar(dataset, tamanho_batch)
        acierto, error = per.calcular_stats(dataset)
        print("[ EPOCH: %d || Tasa acierto: %.3f %% || Error: %.3f ]" % (epoch, acierto * 100, error))
    per.guardar("digitReader")

if __name__ == '__main__':
    per = ai.Perceptron.cargar("digitReader")
    dataset = ai.load_dataset('dataset.csv', False, True)
    random.shuffle(dataset)
    acierto, error = per.calcular_stats(dataset)
    print("[ Tasa acierto: %.3f %% || Error: %.3f ]" % (acierto * 100, error))

