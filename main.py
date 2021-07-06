import perceptron as ai
import random


def train():
    # META PARAMETERS
    layers = [64, 32, 10]
    ratio = 0.1
    batch_size = 10
    epochs = 50

    per = ai.Perceptron(layers, ratio)
    dataset = ai.load_dataset('dataset.csv', ";", False, True)
    for epoch in range(epochs):
        random.shuffle(dataset)  # Better results when dataset is shuffled
        per.train(dataset, batch_size)
        acc, error = per.get_stats(dataset)
        print("[ EPOCH: %d || Accuracy: %.3f %% || Error: %.3f ]" % (epoch+1, acc * 100, error))
    per.save("digitReader")


def test():
    per = ai.Perceptron.load("digitReader")
    dataset = ai.load_dataset('dataset.csv', ";", False, True)
    acc, error = per.get_stats(dataset)
    print("[ Accuracy: %.3f %% || Error: %.3f ]" % (acc * 100, error))


if __name__ == '__main__':
    test()

