import math
import random
import pandas
import pickle


def load_dataset(archivo, indexar_salidas, normalizar):
    datos = pandas.read_csv(archivo, header=None, sep=';').values.tolist()
    # Indexamos salidas
    if indexar_salidas:
        diccionario = {}
        for fila in datos:
            if fila[-1] not in diccionario:
                diccionario.update({fila[-1]: len(diccionario)})
            fila[-1] = diccionario.get(fila[-1])
    # Normalizamos
    if not normalizar:
        return datos
    cols = [{'max': datos[0][i], 'min': datos[0][i]} for i in range(len(datos[0][:-1]))]
    for fila in datos[1:]:
        for i in range(len(fila) - 1):
            cols[i]['max'] = max(cols[i]['max'], fila[i])
            cols[i]['min'] = min(cols[i]['min'], fila[i])
    filas = []
    for fila in datos:
        aux = []
        for i in range(len(fila) - 1):
            if cols[i]['max'] - cols[i]['min'] == 0:
                aux.append(0)
                continue
            aux.append((fila[i] - cols[i]['min']) / (cols[i]['max'] - cols[i]['min']))
        aux.append(fila[-1])
        filas.append(aux)
    return filas


class Perceptron:

    def __init__(self, capas, ratio):
        if 0 in capas or len(capas) < 3:
            print("Configuración imposible, constructor abortado")
            return
        self.red = []
        for k in range(1, len(capas)):
            capa = [
                {'pesos': [random.uniform(-1, 1) for _ in range(capas[k - 1])],
                 'bias': random.uniform(-1, 1),
                 'salida': 0,
                 'delta': 0} for _ in range(capas[k])
            ]
            self.red.append(capa)
        self.ratio_aprendizaje = ratio

    def predecir(self, entradas):
        if len(entradas) != len(self.red[0][0]['pesos']):
            print("Número de entradas inválido, abortado")
            return
        salidas = entradas
        for capa in self.red:
            aux = salidas
            salidas = []
            for neurona in capa:
                # Calculamos la activacion
                resultado = neurona['bias']
                for i in range(len(neurona['pesos'])):
                    resultado += neurona['pesos'][i] * aux[i]
                neurona['salida'] = 1 / (1 + math.exp(-resultado))
                salidas.append(neurona['salida'])
        return salidas

    def entrenar(self, datos, batch_size):
        # Calculamos el número de batches
        n_batches = math.ceil(len(datos) / batch_size)

        for n_batch in range(n_batches):
            # Creamos el paquete de datos
            if n_batch != n_batches - 1:
                batch = datos[n_batch * batch_size:(n_batch + 1) * batch_size]
            else:
                batch = datos[n_batch * batch_size:len(datos)]

            # Predecimos y hacemos back-propagation respecto a las salidas de referencia para actualizar las deltas
            for fila in batch:
                salidas = [0 for _ in range(len(self.red[-1]))]
                salidas[fila[-1]] = 1
                self.predecir(fila[:-1])
                # Recorremos las capas en orden inverso desde la final hacia al inicio
                for i in reversed(range(len(self.red))):
                    capa = self.red[i]
                    for j in range(len(capa)):
                        if i != len(self.red) - 1:  # Capa oculta
                            err = 0
                            for neurona in self.red[i + 1]:
                                err += (neurona['pesos'][j] * neurona['delta'])
                            neurona = capa[j]
                        else:  # Capa final
                            neurona = capa[j]
                            err = salidas[j] - neurona['salida']
                        # Derivada de tranferencia
                        neurona['delta'] = err * neurona['salida'] * (1.0 - neurona['salida'])

            # Actualizamos los pesos en base a la información anterior
            for fila in batch:
                for i in range(len(self.red)):
                    inputs = fila[:-1]
                    if i != 0:
                        inputs = [neurona['salida'] for neurona in self.red[i - 1]]
                    for neurona in self.red[i]:
                        for j in range(len(inputs)):
                            neurona['pesos'][j] += self.ratio_aprendizaje * neurona['delta'] * inputs[j]
                        neurona['bias'] += self.ratio_aprendizaje * neurona['delta']

    def calcular_stats(self, datos):
        sum_error = 0
        acertadas = 0
        for fila in datos:
            output = self.predecir(fila[:-1])
            esperado = [0 for _ in range(len(self.red[-1]))]
            esperado[fila[-1]] = 1
            sum_error += sum([(esperado[i] - output[i]) ** 2 for i in range(len(esperado))])
            if output.index(max(output)) == fila[-1]:
                acertadas += 1
        return acertadas / len(datos), sum_error

    def guardar(self, nombre):
        with open(nombre + '.pickle', 'wb') as archivo:
            pickle.dump(self, archivo)

    @staticmethod
    def cargar(nombre):
        with open(nombre + '.pickle', 'rb') as archivo:
            return pickle.load(archivo)
