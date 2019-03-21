import mnist
import random
import PIL
import matplotlib.pyplot as plt

from learnProblem import Data_set
import numpy
class Data_set_NN(Data_set):

    def __init__(self, train_images, train_label, test_images, test_label):
        self.train = train_images.tolist()
        self.train_label = train_label.tolist()
        self.test = test_images.tolist()
        self.test_label = test_label.tolist()
        for i,image in enumerate(self.train):
            image.append(self.train_label[i])
        self.create_features()
        self.display(1,"Tuples read. \nTraining set", len(self.train),
                    "examples. Number of columns:",{len(e) for e in self.train},
                    "\nTest set", len(self.test),
                    "examples. Number of columns:",{len(e) for e in self.test}
                    )

    def create_features(self):
        self.input_features = []
        for row in range(28):
            for col in range(28):
                def feat(e, index_row = row, index_col = col):
                    return e[index_row][index_col]
                feat.__doc__ = "Pixel[" + str(row) + "][" +str(col) + "]"
                self.input_features.append(feat)

        def target(g, index = 28):
            goal = [0]*10
            goal[g[index]] = 1
            return goal
        self.target = target

from learnNN import NN, Linear_complete_layer, Sigmoid_layer

if __name__ == "__main__":
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    import time
    start_time = time.perf_counter()
    data = Data_set_NN(train_images,train_labels,test_images,test_labels)
    nn1 = NN(data)
#    nn1.add_layer(Linear_complete_layer(nn1, 392))
#    nn1.add_layer(Sigmoid_layer(nn1))
    nn1.add_layer(Linear_complete_layer(nn1, 10))
    nn1.add_layer(Sigmoid_layer(nn1))
    nn1.learn(100)

    true_case = 0
    for i,e in enumerate(data.test):
    	predict = nn1.predictor(e)
    	if predict.index(max(predict)) == data.test_label[i]:
    		true_case += 1

    accuracy = true_case/len(data.test)
    end_time = time.perf_counter()

    print("Accuracy:", accuracy)
    print("Time:", end_time - start_time)

