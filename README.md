A Kotlin library for building neural networks. 

# Example Usage 

Here's how to train a simple MNIST digits classifier. 


```kotlin
package com.momid

import kotlin.math.absoluteValue
import kotlin.math.exp
import kotlin.math.ln

fun main() {
    val relu = ActivationFunction { output ->
        Relu(output)
    }

    val softmax = ActivationFunction { output ->
        softmax(output)
    }

    val crossEntropyLoss = LossFunction { output, predictedOutput ->
        - SumOf(predictedOutput) { i ->
            output[i] * Log(predictedOutput[i])
        }
    }

    val inputLayer = layer(28 * 28, relu)
    val outputLayer = layer(10, softmax)
    val hidden0 = layer(100, relu)
    val hidden1 = layer(37, relu)
    inputLayer.denseTo(hidden0).denseTo(hidden1).denseTo(outputLayer, true)

    val neuralNetwork = NeuralNetwork(arrayListOf(inputLayer, hidden0, hidden1, outputLayer), inputLayer, outputLayer, crossEntropyLoss)

    val trainingData = decodeMNISTTraining()

    val dataset = Dataset(trainingData.images.map {
        flatten(it.pixels)
    }, trainingData.labels.map {
        oneHotEncode(it, 10)
    })

    var accuracy = 0.0

    neuralNetwork.iterate(dataset, 32, 10) { index ->
        this.forward()
        val outputs = neuralNetwork.outputNeurons.neurons
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
        if (index % 100 == 0) {
            println("data " + this.currentDataIndex + " epoch " + this.currentEpoch)
            println("loss: " + neuralNetwork.currentLoss)
            println("accuracy: " + accuracy / 100)
            accuracy = 0.0
            println("--------------------------------------------------------")
        }
        this.backPropagate()
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
```

# Notes 
- Neurogram is designed to allow computation graphs and connecting gruops of neurons in custom structures. but for now, only one input layer and one output layer is allowed. 
- class names may change and some temporary and unconventional names are used right now.
- optimizations and of vector operations have not been implemented yet and it may be slow for larger network right now. 
