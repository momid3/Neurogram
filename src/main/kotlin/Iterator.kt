package com.momid

import com.momid.Iterator

class Dataset(val inputs: List<DoubleArray>, val expectedOutputs: List<DoubleArray>)

class Iterator(
    val neuralNetwork: NeuralNetwork,
    val dataset: Dataset,
    val batchSize: Int,
    val epochs: Int,
    val datasetSize: Int = dataset.inputs.size,
    var currentBatchIndex: Int = 0,
    val currentEpoch: Int = 0,
    var currentDataIndex: Int = 0
)

fun Iterator.forward() {
    if (this.currentDataIndex + this.batchSize <= this.datasetSize) {
        for (index in 0 until this.batchSize) {
            this.currentBatchIndex = index
            this.neuralNetwork.currentBatchIndex = this.currentBatchIndex
            this.neuralNetwork.forward(this.dataset.inputs[this.currentDataIndex + index])
            this.neuralNetwork.backPropagate(this.dataset.expectedOutputs[this.currentDataIndex + index])
        }
        this.currentDataIndex += this.batchSize
    }
}

fun Iterator.backPropagate() {
    this.neuralNetwork.neurons.forEach { layer ->
        for (neuronsIndex in layer.weights.indices) {
            for (connectionIndex in layer.weights[neuronsIndex].indices) {
                for (nextNeuronsIndex in layer.weights[neuronsIndex][connectionIndex].indices) {
                    layer.weights[neuronsIndex][connectionIndex][nextNeuronsIndex] += layer.weightsUpdates!!.map {
                        it[neuronsIndex][connectionIndex][nextNeuronsIndex]
                    }.average()
                }
            }

            layer.biases[neuronsIndex] += layer.biasesUpdates!!.map {
                it[neuronsIndex]
            }.average()
        }
    }
}

fun NeuralNetwork.iterate(dataset: Dataset, batchSize: Int, epochs: Int, block: Iterator.(iterationIndex: Int) -> Unit) {
    this.neurons.forEach { neurons ->
        neurons.initForBackpropagation(batchSize)
    }

    val iterator = Iterator(this, dataset, batchSize, epochs, dataset.inputs.size, 0, 0, 0)

    var iterationIndex = 0

    repeat(epochs) {
        iterator.currentDataIndex = 0
        iterator.currentBatchIndex = 0
        while (iterator.currentDataIndex + iterator.batchSize <= iterator.datasetSize) {
            iterator.block(iterationIndex)
            iterationIndex += 1
        }
    }
}
