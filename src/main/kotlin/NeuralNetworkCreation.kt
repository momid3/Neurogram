package com.momid

import kotlin.math.sqrt
import kotlin.random.Random

val random = java.util.Random()

fun Layer.denseTo(neurons: Layer): Layer {
    this.forward.add(neurons)
    this.weights.forEach {
        it.add(
            DoubleArray(neurons.preActivation.size)
        )
    }
    neurons.backward.add(this)

    return neurons
}

fun NeuralNetwork.weightInitialization() {
    val weightsHashMap = hashMapOf<Layer, DoubleArray>()
    val weightsIndexHashMap = hashMapOf<Layer, Int>()
    for (layer in this.neurons) {
        val weights = weightInitialization(
            layer.backward.sumOf {
                it.preActivation.size
            }, layer.preActivation.size, if (layer.activationFunction !is Activation.Softmax) {
                InitializationType.HE
            } else {
                InitializationType.XAVIER
            }
        )

        weightsHashMap[layer] = weights
        weightsIndexHashMap[layer] = 0
    }

    for (layer in this.neurons) {
        for (neuronIndex in layer.preActivation.indices) {
            for (connectionIndex in layer.weights[neuronIndex].indices) {
                for (nextNeuronIndex in layer.weights[neuronIndex][connectionIndex].indices) {
                    layer.weights[neuronIndex][connectionIndex][nextNeuronIndex] = weightsHashMap[layer.forward[connectionIndex]]!![weightsIndexHashMap[layer.forward[connectionIndex]]!!]
                    weightsIndexHashMap[layer.forward[connectionIndex]] = weightsIndexHashMap[layer.forward[connectionIndex]]!! + 1
                }
            }
        }
    }
}

//fun weightInitialization(inputSize: Int, size: Int, isSoftmax: Boolean = false): DoubleArray {
//    return DoubleArray(size) {
//        val range = if (!isSoftmax) {
//            2.toDouble() * 2 / inputSize
//        } else {
//            2.toDouble() * 2 / (inputSize * size)
//        }
//        (random.nextDouble() * range) - (range / 2)
//    }
//}

enum class InitializationType {
    XAVIER, HE
}

enum class DistributionType {
    UNIFORM, NORMAL
}

fun weightInitialization(
    inputSize: Int,
    size: Int,
    initializationType: InitializationType = InitializationType.HE,
    distributionType: DistributionType = DistributionType.UNIFORM
): DoubleArray {
    return when (initializationType) {
        InitializationType.XAVIER -> {
            when (distributionType) {
                DistributionType.UNIFORM -> {
                    val range = sqrt(6.0 / (inputSize + size))
                    DoubleArray(size * inputSize) { (Random.nextDouble() * 2 * range) - range }
                }
                DistributionType.NORMAL -> {
                    val stdDev = sqrt(2.0 / (inputSize + size))
                    DoubleArray(size * inputSize) { random.nextGaussian() * stdDev }
                }
            }
        }
        InitializationType.HE -> {
            when (distributionType) {
                DistributionType.UNIFORM -> {
                    val range = sqrt(6.0 / inputSize)
                    DoubleArray(size * inputSize) { (Random.nextDouble() * 2 * range) - range }
                }
                DistributionType.NORMAL -> {
                    val stdDev = sqrt(2.0 / inputSize)
                    DoubleArray(size * inputSize) { random.nextGaussian() * stdDev }
                }
            }
        }
    }
}
