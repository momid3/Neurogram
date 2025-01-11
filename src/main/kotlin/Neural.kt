package com.momid

class Neurons(
    val neurons: DoubleArray,
    val connectedTo: ArrayList<Neurons>,
    val weights: ArrayList<ArrayList<DoubleArray>>,
    val biases: DoubleArray,
    val connectionsFrom: ArrayList<Neurons>,
    val neuronsBeforeActivation: DoubleArray,
    val derivativesBeforeActivation: DoubleArray,
    val derivativesAfterActivation: DoubleArray,
    val activationFunction: ActivationFunction
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
    val neurons: ArrayList<Neurons>,
    val inputNeurons: Neurons,
    val outputNeurons: Neurons,
    val lossFunction: LossFunction,
) {
    var currentLoss = 0.0
    var currentBatchIndex = 0
}

fun NeuralNetwork.forward(input: DoubleArray) {
    this.neurons.forEach { currentNeurons ->
        for (index in currentNeurons.neurons.indices) {
            currentNeurons.neurons[index] = 0.0
        }

        for (index in currentNeurons.neuronsBeforeActivation.indices) {
            currentNeurons.neuronsBeforeActivation[index] = 0.0
        }

        for (index in currentNeurons.derivativesBeforeActivation.indices) {
            currentNeurons.derivativesBeforeActivation[index] = 0.0
        }

        for (index in currentNeurons.derivativesBeforeActivation.indices) {
            currentNeurons.derivativesAfterActivation[index] = 0.0
        }
    }

    input.copyInto(this.inputNeurons.neurons)
    forward(arrayListOf(this.inputNeurons))
}

internal fun NeuralNetwork.forward(currentNeurons: ArrayList<Neurons>) {
    val next = ArrayList<Neurons>()

    currentNeurons.forEach {
        it.connectedTo.forEach {
            if (!next.contains(it)) {
                next.add(it)
            }
        }
    }

    currentNeurons.forEach { layer ->
        layer.connectedTo.forEachIndexed { nextLayerIndex, nextLayer ->
            for (nextLayerNeuronsIndex in nextLayer.neuronsBeforeActivation.indices) {
                var sum = 0.0
                for (layerNeuronIndex in layer.neuronsBeforeActivation.indices) {
                    sum += layer.neurons[layerNeuronIndex] * layer.weights[layerNeuronIndex][nextLayerIndex][nextLayerNeuronsIndex]
                }
                nextLayer.neuronsBeforeActivation[nextLayerNeuronsIndex] += sum
            }
        }
    }

    next.forEach { nextLayer ->
        val neuronsBeforeActivationVariableList = VariableList(variableValues = nextLayer.neuronsBeforeActivation)
        neuronsBeforeActivationVariableList.isInRespectTo = true
        val activation = nextLayer.activationFunction.function(neuronsBeforeActivationVariableList)
        for (nextLayerNeuronsIndex in nextLayer.neuronsBeforeActivation.indices) {
            nextLayer.neuronsBeforeActivation[nextLayerNeuronsIndex] += nextLayer.biases[nextLayerNeuronsIndex]

            neuronsBeforeActivationVariableList.currentVariableIndex = nextLayerNeuronsIndex
            nextLayer.neurons[nextLayerNeuronsIndex] = activation.eval()
        }
    }

    if (next.isNotEmpty()) {
        forward(next)
    }
}

fun layer(numberOfNeurons: Int, activationFunction: ActivationFunction): Neurons {
    val weights = ArrayList<ArrayList<DoubleArray>>()
    for (index in 0 until numberOfNeurons) {
        weights.add(ArrayList())
    }
    return Neurons(
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

class LossFunction(val function: (output: VariableList, predictedOutput: VariableList) -> Function)

class ActivationFunction(val function: (output: VariableList) -> Function)
