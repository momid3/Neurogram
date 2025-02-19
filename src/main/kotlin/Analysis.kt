package com.momid

import kotlin.math.absoluteValue
import kotlin.math.exp
import kotlin.math.ln

fun main() {
    val relu = ActivationFunction { output ->
        ReluFunction(output)
    }

    val softmax = ActivationFunction { output ->
        softmax(output)
    }

    val crossEntropyLoss = LossFunction { output, predictedOutput ->
        - SumOf(predictedOutput) { i ->
            output[i] * Log(predictedOutput[i])
        }
    }

    val inputLayer = layer(28 * 28, Activation.Relu)
    val outputLayer = layer(10, Activation.Softmax)
    val hidden0 = layer(100, Activation.Relu)
    val hidden1 = layer(37, Activation.Relu)
    inputLayer.denseTo(hidden0).denseTo(hidden1).denseTo(outputLayer)
    hidden0.denseTo(outputLayer)

    val neuralNetwork = NeuralNetwork(arrayListOf(inputLayer, hidden0, hidden1, outputLayer), inputLayer, outputLayer, crossEntropyLoss, 0.003)

    neuralNetwork.weightInitialization()

    val trainingData = decodeMNISTTraining()

    val dataset = Dataset(trainingData.images.map {
        flatten(it.pixels)
    }, trainingData.labels.map {
        oneHotEncode(it, 10)
    })

    var accuracy = 0.0
    var loss = 0.0

    neuralNetwork.iterate(dataset, 32, 10) { index ->
        this.forward()
        val outputs = neuralNetwork.outputLayer.postActivation
        var predicted = 0
        for (outputIndex in outputs.indices) {
            if (outputs[outputIndex] > outputs[predicted]) {
                predicted = outputIndex
            }
        }
        accuracy += if (predicted == dataset.expectedOutputs[this.currentDataIndex - 1].indexOfFirst {
                it == 1.0
            }) {
            1.0
        } else {
            0.0
        }

        loss += neuralNetwork.currentLoss

        if (index % 100 == 0) {
            println("data " + this.currentDataIndex + " epoch " + this.currentEpoch)
            println("loss: " + loss / 100)
            println("accuracy: " + accuracy / 100)
            accuracy = 0.0
            loss = 0.0
//                printLayerData(hidden0, "hidden 0")
//                printLayerData(hidden1, "hidden 1")
//                printLayerData(outputLayer, "output layer")
//                println("outputs " + outputLayer.neurons.joinToString())
            println("-------------------------------------------------------------------------------------------------------------------")
        }
//            println("outputs " + outputLayer.neurons.joinToString())
//            println("outputs before activation " + outputLayer.neuronsBeforeActivation.joinToString())
//            println("label " + trainingData.labels[index])
        this.backPropagate()
//        println("derivatives after activation " + outputLayer.derivativesAfterActivation.joinToString())
    }
}

fun oneHotEncode(value: Int, size: Int): DoubleArray {
    val values = DoubleArray(size)
    values[value] = 1.0
    return values
}

fun flatten(data: List<List<Double>>): DoubleArray {
    return data.flatten().toDoubleArray()
}

fun clip(values: DoubleArray) {
    for (index in values.indices) {
        if (values[index] > 1.0) {
            values[index] = 1.0
        }
        if (values[index] < - 1.0) {
            values[index] = - 1.0
        }
    }
}

fun clip(value: Double): Double {
    if (value >= 1.0) {
        return 1.0
    } else if (value < - 1.0) {
        return - 1.0
    } else {
        return value
    }
}

fun softMaxOther(thisParam: VariableList, otherParamIndex: Int): Function {
    return exp(thisParam.variableValues[otherParamIndex]) / SumOf(thisParam) { i ->
        Exp(thisParam[i])
    }
}

fun softMaxOtherDerivative(thisParam: VariableList, otherParamIndex: Int): Function {
    return ln(thisParam.variableValues[otherParamIndex]) / SumOf(thisParam) { i ->
        thisParam[i]
    }
}

fun printLayerData(neurons: Layer, name: String) {
    val propertyNames = listOf(
        "bias absolute average",
        "weights absolute average"
    )

    val biasAverage = neurons.biases.map {
        it.absoluteValue
    }.average()

    val weightsAverage = neurons.backward[0].weights.map {
        it[0]
    }[0].map {
        it.absoluteValue
    }.average()

    printTable(listOf(name to listOf(biasAverage, weightsAverage)), propertyNames)
}
