# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from report.evaluator import Evaluator

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, 
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

    def train(self, verbose=True):
        """
        trains the perceptron with the training set
        :param verbose: prints accuracy information if true
        :return: trained perceptron ready to classify 7's
        """
        for i in range(self.epochs):
            if verbose:
                evaluator = Evaluator()
                evaluator.printAccuracy(self.validationSet, self.evaluate(self.validationSet.input))
            for label,input in zip(self.trainingSet.label,self.trainingSet.input):
                error = label - int(self.fire(input))
                self.updateWeights(input, error)

    def classify(self, testInstance):
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """
        Evaluate a whole dataset
        :param test: test Input to be evaluated
        :return: returns the  classified test Input
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        """
        updates the weights for the perceptron
        :param input: is the new Input data
        :param error: the error value calculated
        :return: updates the perceptrons weight
        """
        self.weight += self.learningRate*error*input
         
    def fire(self, input):
        """
        Fire the output of the perceptron corresponding to the input
        :param input: Input Data for the perceptron
        :return: returns a 1 for classifying as a 7 and 0 if it doesn't recognize it as a 7
        """
        return Activation.sign(np.dot(np.array(input), self.weight))
