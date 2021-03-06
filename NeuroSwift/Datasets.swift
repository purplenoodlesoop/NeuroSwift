//
//  Datasets.swift
//  NeuroSwift
//
//  Created by Яков Карпов on 11.02.2021.
//

import Foundation

enum Datasets {
    
    enum Animals {
        // One - dog, zero - cat
        static let weights: [[Double]] = [[18.9], [12.3], [6.8], [2.2], [1.5], [4.1]]
        static let classification: [Double] = [1, 1, 1, 0, 0, 0]
    }
    
    enum LinearFunction {
        // f(x) = 5x - 25
        // One - plus, zero - minus
        static let xs: [[Double]] = [
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
        static let ys: [Double] = [
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
    }
}

