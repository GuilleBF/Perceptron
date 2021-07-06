import math
import random
import pandas
import pickle


# Load training dataset
def load_dataset(file, sep, index_outputs, normalize_outputs):
    # Read csv using pandas, create list of the rows
    data = pandas.read_csv(file, header=None, sep=sep).values.tolist()

    # If dataset outputs not numeric, must replace with numeric indices
    if index_outputs:
        dic = {}
        for row in data:
            if row[-1] not in dic:
                # If new key found, added to de dictionary (key, nextIndex)
                dic.update({row[-1]: len(dic)})
            # Row output updated to be indexed (numerical)
            row[-1] = dic.get(row[-1])

    if not normalize_outputs:
        return data

    # Find max and min of each column of the dataset (except the last column, which is the output)
    cols = [{'max': data[0][i], 'min': data[0][i]} for i in range(len(data[0][:-1]))]
    for row in data[1:]:
        for i in range(len(row) - 1):
            cols[i]['max'] = max(cols[i]['max'], row[i])
            cols[i]['min'] = min(cols[i]['min'], row[i])

    # For each row, normalize columns with formula (columnValue - columnMin) / (columnMax - columnMin)
    normalized_rows = []
    for row in data:
        aux = []
        for i in range(len(row) - 1):
            if cols[i]['max'] - cols[i]['min'] == 0:
                aux.append(0)
                continue
            aux.append((row[i] - cols[i]['min']) / (cols[i]['max'] - cols[i]['min']))
        aux.append(row[-1])
        normalized_rows.append(aux)
    return normalized_rows


class Perceptron:

    def __init__(self, layers, learning_rate):
        if 0 in layers or len(layers) < 3:
            print("Aborted, format must be [n_inputs, n_layer1, n_layer2..., n_outputs]")
            return

        self.network = []
        for k in range(1, len(layers)):
            layer = [
                {'weights': [random.uniform(-1, 1) for _ in range(layers[k - 1])],
                 'bias': random.uniform(-1, 1),
                 'output': 0,
                 'delta': 0} for _ in range(layers[k])
            ]
            self.network.append(layer)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        if len(inputs) != len(self.network[0][0]['weights']):
            print("Aborted, wrong number of inputs")
            return

        outputs = inputs
        for layer in self.network:
            aux = outputs
            outputs = []
            for neuron in layer:
                # Calculate neeuron activation and append to next output vector
                result = neuron['bias']
                for i in range(len(neuron['weights'])):
                    result += neuron['weights'][i] * aux[i]
                neuron['output'] = 1 / (1 + math.exp(-result))  # Using sigmoid
                outputs.append(neuron['output'])
        return outputs

    def train(self, data, batch_size):
        # Calculate number of batches
        n_batches = math.ceil(len(data) / batch_size)

        for n_batch in range(n_batches):
            # Create data batch
            if n_batch != n_batches - 1:
                batch = data[n_batch * batch_size:(n_batch + 1) * batch_size]
            else:
                batch = data[n_batch * batch_size:len(data)]

            # Predict (foward-propagation) and back-propagation to update deltas
            # machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python
            for row in batch:
                outputs = [0 for _ in range(len(self.network[-1]))]
                outputs[row[-1]] = 1
                self.predict(row[:-1])

                # From end to begining
                for i in reversed(range(len(self.network))):
                    layer = self.network[i]
                    for j in range(len(layer)):
                        if i == len(self.network) - 1:  # Output layer
                            neuron = layer[j]
                            err = outputs[j] - neuron['output']
                        else:  # Hidden layer
                            err = 0
                            for neuron in self.network[i + 1]:
                                err += (neuron['weights'][j] * neuron['delta'])
                            neuron = layer[j]
                        # Transfer derivative
                        neuron['delta'] = err * neuron['output'] * (1.0 - neuron['output'])

            # Update weights and bias
            for row in batch:
                for i in range(len(self.network)):
                    inputs = row[:-1]
                    if i != 0:
                        inputs = [neuron['output'] for neuron in self.network[i - 1]]
                    for neuron in self.network[i]:
                        for j in range(len(inputs)):
                            neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
                        neuron['bias'] += self.learning_rate * neuron['delta']

    # Calculate accuracy and error in reference to a dataset
    def get_stats(self, data):
        sum_error = 0
        correct = 0
        for row in data:
            output = self.predict(row[:-1])
            expected = [0 for _ in range(len(self.network[-1]))]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - output[i]) ** 2 for i in range(len(expected))])
            if output.index(max(output)) == row[-1]:
                correct += 1
        return correct / len(data), sum_error

    def save(self, name):
        with open(name + '.pickle', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(name):
        with open(name + '.pickle', 'rb') as file:
            return pickle.load(file)
