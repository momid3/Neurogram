package com.momid

fun NeuralNetwork.backPropagate(expectedOutput: DoubleArray) {
    val outputNeuronsVariableList = VariableList(variableValues = this.outputNeurons.neurons)
    val expectedOutputNeuronsVariableList = VariableList(variableValues = expectedOutput)
    val lossFunctionForNeurons = this.lossFunction.function(expectedOutputNeuronsVariableList, outputNeuronsVariableList)
    outputNeuronsVariableList.isInRespectTo = true
    val loss = lossFunctionForNeurons.eval()
    this.currentLoss = loss
    val derivative = lossFunctionForNeurons.derivative()

    for (outputNeuronsIndex in this.outputNeurons.neurons.indices) {
        outputNeuronsVariableList.currentVariableIndex = outputNeuronsIndex
        var derivativeValue = derivative.eval()
        outputNeurons.derivativesAfterActivation[outputNeuronsIndex] = derivativeValue
    }

    backPropagate(listOf(outputNeurons))
}

internal fun NeuralNetwork.backPropagate(currentNeuronsList: List<Neurons>) {
    for (currentNeurons in currentNeuronsList) {
        val currentDerivativesBeforeActivationVariableList = VariableList(variableValues = currentNeurons.neuronsBeforeActivation)
        currentDerivativesBeforeActivationVariableList.isInRespectTo = true
        val activation = currentNeurons.activationFunction.function(currentDerivativesBeforeActivationVariableList)

        for (currentDerivativesIndex in currentNeurons.derivativesAfterActivation.indices) {
            if (currentNeurons != this.outputNeurons) {
                currentDerivativesBeforeActivationVariableList.currentVariableIndex = currentDerivativesIndex
                val activationDerivative = activation.derivative()
                var activationDerivativeValue = activationDerivative.eval()
                currentNeurons.derivativesBeforeActivation[currentDerivativesIndex] =
                    activationDerivativeValue *
                            currentNeurons.derivativesAfterActivation[currentDerivativesIndex]
            } else {
                for (otherDerivativeIndex in currentNeurons.derivativesAfterActivation.indices) {
                    if (currentDerivativesIndex == otherDerivativeIndex) {
                        currentDerivativesBeforeActivationVariableList.currentVariableIndex = currentDerivativesIndex
                        val activationDerivativeValue = softmaxDerivative(currentNeurons.neuronsBeforeActivation, currentDerivativesIndex, currentDerivativesIndex)
                        currentNeurons.derivativesBeforeActivation[currentDerivativesIndex] +=
                            activationDerivativeValue *
                                    currentNeurons.derivativesAfterActivation[currentDerivativesIndex]
                    } else {
                        val activationDerivativeValue = softmaxDerivative(currentNeurons.neuronsBeforeActivation, otherDerivativeIndex, currentDerivativesIndex)

                        currentNeurons.derivativesBeforeActivation[currentDerivativesIndex] +=
                            activationDerivativeValue *
                                    currentNeurons.derivativesAfterActivation[otherDerivativeIndex]
                    }
                }
            }
        }

        for (currentNeuronsIndex in currentNeurons.neurons.indices) {
            var biasDerivativeValue = currentNeurons.derivativesBeforeActivation[currentNeuronsIndex]
            currentNeurons.biasesUpdates!![this.currentBatchIndex][currentNeuronsIndex] = - biasDerivativeValue * 0.01
        }
    }

    val previousNeuronsList = ArrayList<Neurons>()
    for (currentNeurons in currentNeuronsList) {
        val previousNeuronsListFromCurrentNeurons = currentNeurons.connectionsFrom
        for (previousNeuronsFromCurrentNeurons in previousNeuronsListFromCurrentNeurons) {
            if (!previousNeuronsList.contains(previousNeuronsFromCurrentNeurons)) {
                previousNeuronsList.add(previousNeuronsFromCurrentNeurons)
            }
        }
    }

    previousNeuronsList.forEach { previousNeurons ->
        for (previousNeuronsIndex in previousNeurons.neurons.indices) {
            for (connectionIndex in previousNeurons.weights[previousNeuronsIndex].indices) {
                for (nextNeuronsIndex in previousNeurons.weights[previousNeuronsIndex][connectionIndex].indices) {
                    var weightDerivativeValue =
                        previousNeurons.neurons[previousNeuronsIndex] *
                                previousNeurons.connectedTo[connectionIndex].derivativesBeforeActivation[nextNeuronsIndex]
                    previousNeurons.weightsUpdates!![this.currentBatchIndex][previousNeuronsIndex][connectionIndex][nextNeuronsIndex] = - weightDerivativeValue * 0.01

                    val neuronDerivativeValue = previousNeurons.weights[previousNeuronsIndex][connectionIndex][nextNeuronsIndex]
                    previousNeurons.derivativesAfterActivation[previousNeuronsIndex] +=
                        neuronDerivativeValue *
                                previousNeurons.connectedTo[connectionIndex].derivativesBeforeActivation[nextNeuronsIndex]
                }
            }
        }
    }

    if (previousNeuronsList.isNotEmpty()) {
        backPropagate(previousNeuronsList)
    }
}
