//
//  Perceptron.swift
//  NeuroSwift
//
//  Created by Яков Карпов on 10.02.2021.
//

import Foundation

struct Perceptron {
    
    typealias Neuron = [Double]
    
    //MARK: - Helper functions
    
    static private func replicateM<T>(n: Int, fn: () -> T) -> [T] { (0..<n).map { _ in fn() } }
    
    static private func zipWith<A, B, C>(_ xs1: [A], _ xs2: [B], _ fn: (A, B) -> C) -> [C] { zip(xs1, xs2).map { fn($0, $1) } }
    
    //MARK: - Private Functions
    
    static private func calculateDotProduct(for neuron: Neuron, inputVector: [Double]) -> Double {
        zipWith(neuron, [-1]+inputVector, *).reduce(0, +)
    }
    
    static private func calculateSigmoid(for sum: Double) -> Double { 1 / (1 + pow(M_E, -sum)) }
    
    static private func backpropagateWeight(with trainingFactor: Double, with resultDifference: Double, inputWeight: Double, inputElement: Double) -> Double {
        inputWeight - trainingFactor * resultDifference * inputElement
    }
    
    static func trainVector(trainingFactor: Double, inputVector: [Double], targetOutput: Double, neuron: Neuron) -> Neuron {
        let resulDifference = classifyWithSigmoid(neuron: neuron, inputVector: inputVector) - targetOutput
        return zipWith(neuron, [-1]+inputVector) { weight, element in
            backpropagateWeight(with: trainingFactor, with: resulDifference, inputWeight: weight, inputElement: element)
        }
    }
    
    static func trainSet(trainingFactor: Double, inputMatrix: [[Double]], targetOutputs: [Double], neuron: Neuron) -> Neuron {
        let trainedNeuron = trainVector(trainingFactor: trainingFactor, inputVector: inputMatrix[0], targetOutput: targetOutputs[0], neuron: neuron)
        if inputMatrix.count == 1 { return trainedNeuron }
        return trainSet(trainingFactor: trainingFactor, inputMatrix: Array(inputMatrix[1...]), targetOutputs: Array(targetOutputs[1...]), neuron: trainedNeuron)
    }
    
    //MARK: - Public Functions
    
    static func neuron(inputsCount: Int) -> Neuron { replicateM(n: inputsCount + 1) { Double.random(in: 1.0...3.0) } }
    
    static func classifyWithSigmoid(neuron: Neuron, inputVector: [Double], withBias: Double = 0, showSigmoidResult: Bool = false) -> Double {
        let sigmoidResult = calculateSigmoid(for: calculateDotProduct(for: neuron, inputVector: inputVector))
        if showSigmoidResult {
            let confidence = sigmoidResult < 0.5 ? 100 - (sigmoidResult * 200): (sigmoidResult * 200) - 100
            print(String(format: "%.1f",confidence) + "% confidence")
        }
        return round(sigmoidResult)
    }
    
    static func trainNeuron(n: Int, trainingFactor: Double = 0.01, inputMatrix: [[Double]], targetOutputs: [Double], neuron: Neuron, withBias: Double = 0) -> Neuron {
        let trainedNeuron = trainSet(trainingFactor: trainingFactor, inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: neuron)
        if n == 1 { return trainedNeuron }
        return trainNeuron(n: n-1, trainingFactor: trainingFactor, inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: trainedNeuron, withBias: withBias)
    }
    
    static func trainNeuronFully(counter: Int = 0, trainingFactor: Double = 0.01, inputMatrix: [[Double]], targetOutputs: [Double], neuron: Neuron) -> Neuron {
        let trainedNeuron = trainSet(trainingFactor: trainingFactor, inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: neuron)
        if trainedNeuron == neuron { return trainedNeuron }
        return trainNeuronFully(counter: counter + 1, inputMatrix: inputMatrix, targetOutputs: targetOutputs, neuron: trainedNeuron)
    }
}
