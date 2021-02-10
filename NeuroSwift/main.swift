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
    
    //MARK: - Helper functions
    
    static private func replicateM<T>(n: Int, fn: () -> T) -> [T] { (0..<n).map { _ in fn() } }
    
    static private func zipWith<A, B, C>(_ xs1: [A], _ xs2: [B], fn: (A, B) -> C) -> [C] {
        zip(xs1, xs2).map { fn($0, $1) }
    }
    
    //MARK: - Private Functions
    
    static private func getWeightedAndBiasedSum(for neuron: Neuron, inputVector: [Double], bias: Double) -> Double {
        let weightedVector: [Double] = zipWith(neuron, [-1]+inputVector, fn: *)
        return weightedVector.reduce(0, +) + bias
    }
    
    static private func calculateSigmoid(for sum: Double) -> Double {
        1 / (1 + pow(M_E, -sum))
    }
    
    static private func backpropagateWeight(with trainingFactor: Double, with resultDifference: Double, inputWeight: Double, inputElement: Double) -> Double {
        inputWeight - trainingFactor * resultDifference * inputElement
    }
    
    static func trainVector(trainingFactor: Double, inputVector: [Double], targetOutput: Double, neuron: Neuron, bias: Double) -> Neuron {
        let resulDifference = classifyWithSigmoid(neuron: neuron, inputVector: inputVector, withBias: bias) - targetOutput
        return zipWith(neuron, [-1]+inputVector) { weight, element in
            backpropagateWeight(with: trainingFactor, with: resulDifference, inputWeight: weight, inputElement: element)
        }
    }
    
    static func trainSet(trainingFactor: Double, inputMatrix: [[Double]], targetOutputs: [Double], neuron: Neuron, withBias: Double) -> Neuron {
        let trainedNeuron = trainVector(trainingFactor: trainingFactor, inputVector: inputMatrix[0], targetOutput: targetOutputs[0], neuron: neuron, bias: withBias)
        if inputMatrix.count == 1 {
            return trainedNeuron
        }
        return trainSet(trainingFactor: trainingFactor, inputMatrix: Array(inputMatrix[1...]), targetOutputs: Array(targetOutputs[1...]), neuron: trainedNeuron, withBias: withBias)
    }
    
    //MARK: - Public Functions
    
    static func neuron(inputsCount: Int) -> Neuron {
        replicateM(n: inputsCount + 1) { Double.random(in: 1.0...3.0) }
    }
    
    static func classifyWithSigmoid(neuron: Neuron, inputVector: [Double], withBias: Double = 0, showSigmoidResult: Bool = false) -> Double {
        let sum = getWeightedAndBiasedSum(for: neuron, inputVector: inputVector, bias: withBias)
        let sigmoidResult = calculateSigmoid(for: sum)
        if showSigmoidResult { print(sigmoidResult) }
        return round(sigmoidResult)
    }
    
    static func trainNeuron(n: Int, trainingFactor: Double = 0.01, inputMatrix: [[Double]], targetOutputs: [Double], neuron: Neuron, withBias: Double = 0) -> Neuron {
        let trainedNeuron = trainSet(trainingFactor: trainingFactor, inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: neuron, withBias: withBias)
        if n == 1 {
            return trainedNeuron
        }
        return trainNeuron(n: n-1, trainingFactor: trainingFactor, inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: trainedNeuron, withBias: withBias)
    }
    
    static func trainNeuron(trainingFactor: Double = 0.01, inputMatrix: [[Double]], targetOutputs: [Double], neuron: Neuron, withBias: Double = 0) -> Neuron {
        let trainedNeuron = trainSet(trainingFactor: trainingFactor, inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: neuron, withBias: withBias)
        if trainedNeuron == neuron {
            return trainedNeuron
        }
        return trainNeuron(inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: trainedNeuron)
    }
}

// f(x) = 5x - 25
let trainingDataset: [[Double]] = [
    [9.373732108560468],
    [5.0050368789476],
    [7.916654314280631],
    [-2.688571619237498],
    [-4.5603931893948735],
    [4.183874968512825],
    [2.2236788068344815],
    [-8.845027818939643],
    [6.222914504486923],
    [0.21267396148589413],
    [4.926321335166078],
    [2.7716234280410887],
    [-4.196253980184523],
    [-0.982138345427515],
    [-1.918898741818687],
    [-7.85858071214284],
    [5.766480557275477],
    [-2.272182433308161],
    [5.256799946754004],
    [-5.470143485626853],
    [3.9151610906830054],
    [-4.062467826400313],
    [8.702913587450855],
    [-2.8841158374106683],
    [3.178764858918214],
    [5.55767660270047],
    [0.5312109489150636],
    [6.553883281624305],
    [-4.619035056815044],
    [4.083056444477586],
    [2.586297526352837],
    [4.757765166129847],
    [-4.1273549754045185],
    [1.8222479228635038],
    [3.368691253316234],
    [-5.04668912632686],
    [-2.9183439224471064],
    [-3.6137669849854603],
    [-9.44368411701214],
    [6.03276413735156],
]

let targetOutputs = [
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0
]

let showResult = true
let n = 1000
let inputVectorLength = 2

let baseNeuron = Perceptron.neuron(inputsCount: inputVectorLength)

let trainedNeuronSet = Perceptron.trainNeuron(inputMatrix: trainingDataset, targetOutputs: targetOutputs, neuron: baseNeuron)

print(Perceptron.classifyWithSigmoid(neuron: trainedNeuronSet, inputVector: [-22.0], showSigmoidResult: showResult)) // 0
print(Perceptron.classifyWithSigmoid(neuron: trainedNeuronSet, inputVector: [5.01], showSigmoidResult: showResult)) // 1
