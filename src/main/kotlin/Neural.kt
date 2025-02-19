package com.momid

class Layer(
    val postActivation: DoubleArray,
    val forward: ArrayList<Layer>,
    val weights: ArrayList<ArrayList<DoubleArray>>,
    val biases: DoubleArray,
    val backward: ArrayList<Layer>,
    val preActivation: DoubleArray,
    val gradientsPreActivation: DoubleArray,
    val gradientsPostActivation: DoubleArray,
    val activationFunction: Activation
) {
    var weightsUpdates: List<List<List<DoubleArray>>>? = null
    var biasesUpdates: List<DoubleArray>? = null

    fun initForBackpropagation(batchSize: Int) {
        weightsUpdates = List(batchSize) {
            List(this.weights.size) {
                List((this.weights.firstOrNull() ?: throw (Throwable("should be connected to something"))).size) {
                    DoubleArray((this.weights.first().firstOrNull() ?: throw (Throwable("should have a non zero size"))).size)
                }
            }
        }

        biasesUpdates = List(batchSize) {
            DoubleArray((this.biases.size))
        }
    }
}

class NeuralNetwork(
    val neurons: ArrayList<Layer>,
    val inputLayer: Layer,
    val outputLayer: Layer,
    val lossFunction: LossFunction,
    val learningRate: Double
) {
    var currentLoss = 0.0
    var currentBatchIndex = 0
}

fun NeuralNetwork.forward(input: DoubleArray) {
    this.neurons.forEach { currentNeurons ->
        for (index in currentNeurons.postActivation.indices) {
            currentNeurons.postActivation[index] = 0.0
        }

        for (index in currentNeurons.preActivation.indices) {
            currentNeurons.preActivation[index] = 0.0
        }

        for (index in currentNeurons.gradientsPreActivation.indices) {
            currentNeurons.gradientsPreActivation[index] = 0.0
        }

        for (index in currentNeurons.gradientsPreActivation.indices) {
            currentNeurons.gradientsPostActivation[index] = 0.0
        }
    }

    input.copyInto(this.inputLayer.postActivation)
    forward(arrayListOf(this.inputLayer))
}

internal fun NeuralNetwork.forward(currentNeurons: ArrayList<Layer>) {
    val next = ArrayList<Layer>()

    currentNeurons.forEach {
        it.forward.forEach {
            if (!next.contains(it)) {
                next.add(it)
            }
        }
    }

    currentNeurons.forEach { layer ->
        layer.forward.forEachIndexed { nextLayerIndex, nextLayer ->
            for (nextLayerNeuronsIndex in nextLayer.preActivation.indices) {
                var sum = 0.0
                for (layerNeuronIndex in layer.preActivation.indices) {
                    sum += layer.postActivation[layerNeuronIndex] * layer.weights[layerNeuronIndex][nextLayerIndex][nextLayerNeuronsIndex]
                }
                nextLayer.preActivation[nextLayerNeuronsIndex] += sum
            }
        }
    }

    next.forEach { nextLayer ->
        when (nextLayer.activationFunction) {
            is ActivationFunction -> {
                val preActivationVariableList = VariableList(variableValues = nextLayer.preActivation)
                preActivationVariableList.isInRespectTo = true
                val activation = nextLayer.activationFunction.function(preActivationVariableList)
                for (nextLayerNeuronsIndex in nextLayer.preActivation.indices) {
                    nextLayer.preActivation[nextLayerNeuronsIndex] += nextLayer.biases[nextLayerNeuronsIndex]

                    preActivationVariableList.currentVariableIndex = nextLayerNeuronsIndex
                    nextLayer.postActivation[nextLayerNeuronsIndex] = activation.eval()
                }
            }

            is Activation.Relu -> {
                for (nextLayerNeuronsIndex in nextLayer.preActivation.indices) {
                    nextLayer.preActivation[nextLayerNeuronsIndex] += nextLayer.biases[nextLayerNeuronsIndex]

                    nextLayer.postActivation[nextLayerNeuronsIndex] =
                        relu(nextLayer.preActivation[nextLayerNeuronsIndex])
                }
            }

            is Activation.Softmax -> {
                val softmax = softmax(nextLayer.preActivation)
                for (nextLayerNeuronsIndex in nextLayer.preActivation.indices) {
                    nextLayer.preActivation[nextLayerNeuronsIndex] += nextLayer.biases[nextLayerNeuronsIndex]

                    nextLayer.postActivation[nextLayerNeuronsIndex] =
                        softmax[nextLayerNeuronsIndex]
                }
            }

            else -> {

            }
        }
    }

    if (next.isNotEmpty()) {
        forward(next)
    }
}

fun layer(numberOfNeurons: Int, activationFunction: Activation): Layer {
    val weights = ArrayList<ArrayList<DoubleArray>>()
    for (index in 0 until numberOfNeurons) {
        weights.add(ArrayList())
    }
    return Layer(
        DoubleArray(numberOfNeurons),
        arrayListOf(),
        weights,
        DoubleArray(numberOfNeurons) {
            0.0
        },
        arrayListOf(),
        DoubleArray(numberOfNeurons),
        DoubleArray(numberOfNeurons),
        DoubleArray(numberOfNeurons),
        activationFunction
    )
}

sealed class Activation {
    data object Relu: Activation()
    data object Sigmoid: Activation()
    data object Softmax: Activation()
}

class LossFunction(val function: (output: VariableList, predictedOutput: VariableList) -> Function)

class ActivationFunction(val function: (output: VariableList) -> Function): Activation()

fun relu(value: Double): Double {
    if (value < 0.0) {
        return 0.0
    } else {
        return value
    }
}

fun reluDerivative(value: Double): Double {
    if (value >= 0.0) {
        return 1.0
    } else {
        return 0.0
    }
}
