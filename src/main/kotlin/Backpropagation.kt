package com.momid

fun NeuralNetwork.backPropagate(expectedOutput: DoubleArray) {
    val outputLayerVariableList = VariableList(variableValues = this.outputLayer.postActivation)
    val expectedOutputNeuronsVariableList = VariableList(variableValues = expectedOutput)
    val lossFunctionForNeurons = this.lossFunction.function(expectedOutputNeuronsVariableList, outputLayerVariableList)
    outputLayerVariableList.isInRespectTo = true
    val loss = lossFunctionForNeurons.eval()
    this.currentLoss = loss
    val derivative = lossFunctionForNeurons.derivative()

    for (outputNeuronsIndex in this.outputLayer.postActivation.indices) {
        outputLayerVariableList.currentVariableIndex = outputNeuronsIndex
        var derivativeValue = derivative.eval()
        outputLayer.gradientsPostActivation[outputNeuronsIndex] = derivativeValue
    }

    backPropagate(listOf(outputLayer))
}

internal fun NeuralNetwork.backPropagate(currentLayers: List<Layer>) {
    for (layer in currentLayers) {
        val gradientsBeforeActivationVariableList = VariableList(variableValues = layer.preActivation)
        gradientsBeforeActivationVariableList.isInRespectTo = true
        when (layer.activationFunction) {
            is ActivationFunction -> {
                val activation = layer.activationFunction.function(gradientsBeforeActivationVariableList)

                for (gradientsIndex in layer.gradientsPostActivation.indices) {
                    gradientsBeforeActivationVariableList.currentVariableIndex = gradientsIndex
                    val activationDerivative = activation.derivative()
                    var activationDerivativeValue = activationDerivative.eval()
                    layer.gradientsPreActivation[gradientsIndex] =
                        activationDerivativeValue *
                                layer.gradientsPostActivation[gradientsIndex]
                }
            }

            is Activation.Softmax -> {
                for (gradientsIndex in layer.gradientsPostActivation.indices) {
                    for (otherDerivativeIndex in layer.gradientsPostActivation.indices) {
                        if (gradientsIndex == otherDerivativeIndex) {
                            gradientsBeforeActivationVariableList.currentVariableIndex = gradientsIndex
                            val activationDerivativeValue =
                                softmaxDerivative(layer.preActivation, gradientsIndex, gradientsIndex)
                            layer.gradientsPreActivation[gradientsIndex] +=
                                activationDerivativeValue *
                                        layer.gradientsPostActivation[gradientsIndex]
                        } else {
                            val activationDerivativeValue =
                                softmaxDerivative(layer.preActivation, otherDerivativeIndex, gradientsIndex)

                            layer.gradientsPreActivation[gradientsIndex] +=
                                activationDerivativeValue *
                                        layer.gradientsPostActivation[otherDerivativeIndex]
                        }
                    }
                }
            }

            is Activation.Relu -> {
                for (gradientsIndex in layer.gradientsPostActivation.indices) {
                    val activationDerivativeValue = reluDerivative(layer.preActivation[gradientsIndex])
                    layer.gradientsPreActivation[gradientsIndex] =
                        activationDerivativeValue *
                                layer.gradientsPostActivation[gradientsIndex]
                }
            }

            else -> {

            }
        }

        for (layerIndex in layer.postActivation.indices) {
            var biasDerivativeValue = layer.gradientsPreActivation[layerIndex]
            layer.biasesUpdates!![this.currentBatchIndex][layerIndex] = - biasDerivativeValue * this.learningRate
        }
    }

    val previousLayers = ArrayList<Layer>()
    for (layer in currentLayers) {
        for (previousLayer in layer.backward) {
            if (!previousLayers.contains(previousLayer)) {
                previousLayers.add(previousLayer)
            }
        }
    }

    previousLayers.forEach { previousLayer ->
        for (previousLayerIndex in previousLayer.postActivation.indices) {
            for (connectionIndex in previousLayer.weights[previousLayerIndex].indices) {
                for (nextLayerIndex in previousLayer.weights[previousLayerIndex][connectionIndex].indices) {
                    var weightDerivativeValue =
                        previousLayer.postActivation[previousLayerIndex] *
                                previousLayer.forward[connectionIndex].gradientsPreActivation[nextLayerIndex]
                    previousLayer.weightsUpdates!![this.currentBatchIndex][previousLayerIndex][connectionIndex][nextLayerIndex] = - weightDerivativeValue * this.learningRate

                    val neuronDerivativeValue = previousLayer.weights[previousLayerIndex][connectionIndex][nextLayerIndex]
                    previousLayer.gradientsPostActivation[previousLayerIndex] +=
                        neuronDerivativeValue *
                                previousLayer.forward[connectionIndex].gradientsPreActivation[nextLayerIndex]
                }
            }
        }
    }

    if (previousLayers.isNotEmpty()) {
        backPropagate(previousLayers)
    }
}
