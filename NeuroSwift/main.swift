//
//  main.swift
//  NeuroSwift
//
//  Created by Яков Карпов on 10.02.2021.
//

import Foundation

import Foundation

struct Perceptron {
    typealias Neuron = [Double]
    
    //MARK: - Private Functions
    
    static private func zipWith<A, B, C>(_ xs1: [A], _ xs2: [B], fn: (A, B) -> C) -> [C] {
        zip(xs1, xs2).map { fn($0, $1) }
    }
    
    static private func getWeightedAndBiasedSum(for neuron: Neuron, inputVector: [Double], bias: Double) -> Double {
        let weightedVector: [Double] = zipWith(neuron, inputVector, fn: *)
        return weightedVector.reduce(0, +) + bias
    }
    
    static private func calculateSigmoid(for sum: Double) -> Double {
        1 / (1 + pow(M_E, -sum))
    }

    static private func calculateSwish(for sum: Double) -> Double {
        sum * calculateSigmoid(for: sum)
    }
    
    static private func backpropagateWeight(with trainingFactor: Double, with resultDifference: Double, inputWeight: Double, inputElement: Double) -> Double {
        inputWeight - trainingFactor * resultDifference * inputElement
    }
    
    static func trainOne(trainingFactor: Double, inputVector: [Double], targetOutput: Double, neuron: Neuron, bias: Double) -> Neuron {
        let resulDifference = classifyWithSigmoid(neuron: neuron, inputVector: inputVector, withBias: bias) - targetOutput
        return zipWith(neuron, inputVector) { weight, element in
            backpropagateWeight(with: trainingFactor, with: resulDifference, inputWeight: weight, inputElement: element)
        }
    }
    
    static func trainSet(trainingFactor: Double, inputMatrix: [[Double]], targetOutputs: [Double], neuron: Neuron, withBias: Double) -> Neuron {
        let trainedNeuron = trainOne(trainingFactor: trainingFactor, inputVector: inputMatrix[0], targetOutput: targetOutputs[0], neuron: neuron, bias: withBias)
        if inputMatrix.count == 1 {
            return trainedNeuron
        }
        return trainSet(trainingFactor: trainingFactor, inputMatrix: Array(inputMatrix[1...]), targetOutputs: Array(targetOutputs[1...]), neuron: trainedNeuron, withBias: withBias)
    }
    
    //MARK: - Public Functions
    
    static func neuron(inputsCount: Int) -> Neuron {
        Array(repeating: Double.random(in: 0.01...0.1), count: inputsCount)
    }
    
    static func classifyWithSigmoid(neuron: Neuron, inputVector: [Double], withBias: Double, showSigmoidResult: Bool = false) -> Double {
        let sum = getWeightedAndBiasedSum(for: neuron, inputVector: inputVector, bias: withBias)
        let sigmoidResult = calculateSigmoid(for: sum)
        if showSigmoidResult { print(sigmoidResult) }
        return round(sigmoidResult)
    }
    
    static func trainNeuron(n: Int, trainingFactor: Double, inputMatrix: [[Double]], targetOutputs: [Double], neuron: Neuron, withBias: Double) -> Neuron {
        let trainedNeuron = trainSet(trainingFactor: trainingFactor, inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: neuron, withBias: withBias)
        if n == 1 {
            return trainedNeuron
        }
        return trainNeuron(n: n-1, trainingFactor: trainingFactor, inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: trainedNeuron, withBias: withBias)
    }
}

// f(x) = 5x - 25
let trainingDataset: [[Double]] = [[-40], [1], [3], [9]]
let targetOutputs = [0.0, 0.0, 0.0, 1.0]

let defaultBasis: Double = 0
let showResult = true
let n = 10

let baseNeuron = Perceptron.neuron(inputsCount: 1)

let trainedNeuronSet = Perceptron.trainNeuron(n: n, trainingFactor: 0.01, inputMatrix: trainingDataset, targetOutputs: targetOutputs, neuron: baseNeuron, withBias: defaultBasis)

print(Perceptron.classifyWithSigmoid(neuron: trainedNeuronSet, inputVector: [-22.0], withBias: defaultBasis, showSigmoidResult: showResult)) // 0
print(Perceptron.classifyWithSigmoid(neuron: trainedNeuronSet, inputVector: [6.0], withBias: defaultBasis, showSigmoidResult: showResult)) // 1

