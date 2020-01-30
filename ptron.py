# Bounnoy Phanthavong (ID: 973081923)
# Homework 1
#
# This is a machine learning program that models a perceptron.
# We train 10 perceptrons to recognize digits from a bunch of images.
# The goal is to find a set of weights that give us minimal loss.
# This program was built in Python 3.

from pathlib import Path
import numpy as np
import csv
import pickle

class perceptron:
    def __init__(self, train, test, output):
        self.trainData = train
        self.testData = test
        self.output = output

        # Create a 2D array with the same rows as the output and cols of training data.
        # Weights are populated randomly between -0.5 to 0.5 for each cell.
        self.weights = np.random.rand(len(train[0]), self.output)*.1-.05


    # The train function takes in the ETA (learning rate) and iterations, then outputs results.
    def train(self, eta, iterations):
        accuracy = np.zeros(iterations)         # Training accuracy.
        accuracyTest = np.zeros(iterations)     # Test accuracy.
        rowsTrain = len(self.trainData)         # Rows in the training data.
        rowsTest = len(self.testData)           # Rows in the test data.

        # Shuffle training/testing data.
        np.random.shuffle(self.trainData)
        np.random.shuffle(self.testData)

        tTrain = self.trainData[:,0]        # Set training target to first column of training data.
        tTrain = np.vstack(tTrain)          # Convert it to a vertical array.
        xTrain = self.trainData[:,1:]       # Set inputs as everything after first column.
        xTrain = xTrain/255                 # Divide all cells to keep calculation small. (0-1)

        # Do the same as above for testing set.
        tTest = self.testData[:,0]
        tTest = np.vstack(tTest)
        xTest = self.testData[:,1:]
        xTest = xTest/255

        # Replace first column with the bias.
        xTrain = np.concatenate( (np.ones((rowsTrain, 1)), xTrain), axis=1 )
        xTest = np.concatenate( (np.ones((rowsTest, 1)), xTest), axis=1 )

        print("Learning Rate = ", eta)
        with open('results.csv', 'a') as csvFile:
            w = csv.writer(csvFile)
            w.writerow(["Learning Rate"] + [eta])
            w.writerow(["Epoch"] + ["Training Accuracy"] + ["Test Accuracy"])

        # Start the learning algorithm.
        for i in range(iterations):
            trainAccuracy = 0
            testAccuracy = 0

            for j in range(rowsTrain):

                # Get index of highest output.
                outputs = np.dot(xTrain[j], self.weights)
                prediction = np.argmax(outputs)

                if prediction != tTrain[j]:

                    # Set y to 1 if output greater than 0, others are 0.
                    # Set t to 0, except for the index of our target output.
                    y = np.where(outputs > 0, 1, 0)
                    t = np.zeros(self.output)
                    t[ int(tTrain[j]) ] = 1

                    # Update the weights.
                    # This is from the formula: w_i' = n(t^k - y^k)x_i^k
                    self.weights -= eta * np.transpose(
                        np.dot(
                            np.vstack(y - t),
                            np.asmatrix(xTrain[j])
                        )
                    )
                else:
                    trainAccuracy += 1

            # At this point, we should have a good set of weights from training our
            # perceptrons. Now we use the weights to guess the digits from the test images.
            for k in range(rowsTest):
                outputs = np.dot(xTest[k], self.weights)
                prediction = np.argmax(outputs)
                if prediction == tTest[k]:
                    testAccuracy += 1

            # Compute accuracy.
            accuracy[i] = (float(trainAccuracy) / float(rowsTrain)) * 100
            accuracyTest[i] = (float(testAccuracy) / float(rowsTest)) * 100

            print("Epoch ", i, ": Training Accuracy = ", int(accuracy[i]), "% / Test Accuracy = ", int(accuracyTest[i]), "%")
            with open('results.csv', 'a') as csvFile:
                w = csv.writer(csvFile)
                w.writerow([i] + [accuracy[i]] + [accuracyTest[i]])

            if accuracy[int(i-1)] > (accuracy[i] + 1):
                break

        return

    # Build the confusion matrix.
    def confusion(self):
        if (len(self.testData[0]) != len(self.trainData[0])):
            print("Error: Training and test data structure does not match.")
            return

        rowsTest = len(self.testData)
        np.random.shuffle(self.testData)    # Shuffle test data.
        t = self.testData[:,0]              # Set test target to first column of test data.
        t = np.vstack(t)                    # Convert it to a vertical array.
        x = self.testData[:,1:]             # Set inputs as everything after the first column.
        x = x/255                           # Divide all cells to keep calculation small. (0-1)

        # Replace first column with the bias.
        x = np.concatenate( (np.ones((rowsTest, 1)), x), axis=1 )

        matrix = np.zeros((self.output, self.output)) # Build our matrix.
        testAccuracy = 0

        for i in range(rowsTest):
            # Get index of highest output.
            outputs = np.dot(x[i], self.weights)
            prediction = np.argmax(outputs)

            # Check if our prediction is correct.
            if prediction == t[i]:
                testAccuracy += 1

            # Plot our data in the table if correct prediction.
            matrix[int(prediction)][int(t[i])] += 1

        # Calculate test accuracy.
        accuracy = int( (float(testAccuracy)/float(rowsTest)) * 100)

        print("Final Accuracy = ", accuracy, "%")

        np.set_printoptions(suppress = True)
        print("\nConfusion Matrix")
        print(matrix, "\n")

        with open('results.csv', 'a') as csvFile:
            w = csv.writer(csvFile)
            w.writerow([])
            w.writerow(["Confusion Matrix"])
            for j in range(self.output):
                w.writerow(matrix[j,:])
            w.writerow(["Final Accuracy"] + [accuracy])
            w.writerow([])

        return


if __name__ == '__main__':

    pklTrain = Path("mnist_train.pkl")
    pklTest = Path("mnist_test.pkl")
    fileTrain = Path("mnist_train.csv")
    fileTest = Path("mnist_test.csv")

    if not fileTrain.exists():
        sys.exit("mnist_train.csv not found")

    if not fileTest.exists():
        sys.exit("mnist_test.csv not found")

    if not pklTrain.exists():
        f = np.genfromtxt("mnist_train.csv", delimiter=",")
        csv = open("mnist_train.pkl", 'wb')
        pickle.dump(f, csv)
        csv.close()

    if not pklTest.exists():
        f = np.genfromtxt("mnist_test.csv", delimiter=",")
        csv = open("mnist_test.pkl", 'wb')
        pickle.dump(f, csv)
        csv.close()

    file = open("mnist_train.pkl", "rb")
    train = pickle.load(file)
    file.close()

    file = open("mnist_test.pkl", "rb")
    test = pickle.load(file)
    file.close()

    output = 10

    p = perceptron(train, test, output)
    p.train(0.001, 50)
    p.confusion()
    print("\n")

    p2 = perceptron(train, test, output)
    p2.train(0.01, 50)
    p2.confusion()
    print("\n")

    p3 = perceptron(train, test, output)
    p3.train(0.1, 50)
    p3.confusion()
