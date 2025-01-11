package com.momid

import kotlin.math.sqrt
import kotlin.random.Random

val random = java.util.Random()

fun Neurons.denseTo(neurons: Neurons, isOutput: Boolean = false): Neurons {
    this.connectedTo.add(neurons)
    this.weights.forEach {
        it.add(
            weightInitialization(
                this.neurons.size, neurons.neurons.size, if (!isOutput) {
                    InitializationType.HE
                } else {
                    InitializationType.XAVIER
                }
            )
        )
    }
    neurons.connectionsFrom.add(this)

    return neurons
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
                    DoubleArray(size) { (Random.nextDouble() * 2 * range) - range }
                }
                DistributionType.NORMAL -> {
                    val stdDev = sqrt(2.0 / (inputSize + size))
                    DoubleArray(size) { random.nextGaussian() * stdDev }
                }
            }
        }
        InitializationType.HE -> {
            when (distributionType) {
                DistributionType.UNIFORM -> {
                    val range = sqrt(6.0 / inputSize)
                    DoubleArray(size) { (Random.nextDouble() * 2 * range) - range }
                }
                DistributionType.NORMAL -> {
                    val stdDev = sqrt(2.0 / inputSize)
                    DoubleArray(size) { random.nextGaussian() * stdDev }
                }
            }
        }
    }
}
