//
//  main.swift
//  NeuroSwift
//
//  Created by Яков Карпов on 10.02.2021.
//

import Foundation


let showResult = true
let inputVector: [Double] = [6.0]
let inputVectorLength = inputVector.count
let dataset = Datasets.Animals.weights
let outputs = Datasets.Animals.classification


let baseNeuron = Perceptron.neuron(inputsCount: inputVectorLength)
let trainedNeuronOnSet = Perceptron.trainNeuronFully(inputMatrix: dataset, targetOutputs: outputs, neuron: baseNeuron)

print(Perceptron.classifyWithSigmoid(neuron: trainedNeuronOnSet, inputVector: inputVector, showSigmoidResult: showResult) == 1.0 ? "Dog" : "Cat")
