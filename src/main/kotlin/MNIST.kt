package com.momid

import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.FileInputStream
import java.io.IOException

data class MnistData(val images: List<MnistImage>, val labels: List<Int>)
data class MnistImage(val pixels: List<List<Double>>)

fun decodeMnist(imagePath: String, labelPath: String): MnistData? {
    try {
        val labelInputStream = BufferedInputStream(FileInputStream(labelPath))
        val labelMagicNumber = labelInputStream.readInt()
        if (labelMagicNumber != 2049) {
            throw IOException("invalid MNIST label file")
        }
        val numLabels = labelInputStream.readInt()
        val labels = IntArray(numLabels)
        val labelBuffer = ByteArray(numLabels)
        labelInputStream.read(labelBuffer)
        for (i in 0 until numLabels) {
            labels[i] = labelBuffer[i].toInt() and 0xFF
        }
        labelInputStream.close()

        // Decode images
        val imageInputStream = BufferedInputStream(FileInputStream(imagePath))
        val imageMagicNumber = imageInputStream.readInt()
        if (imageMagicNumber != 2051) {
            throw IOException("invalid MNIST image file")
        }
        val numImages = imageInputStream.readInt()
        val numRows = imageInputStream.readInt()
        val numCols = imageInputStream.readInt()
        val images = mutableListOf<MnistImage>()
        val imageSize = numRows * numCols
        val imageBuffer = ByteArray(imageSize)

        for (i in 0 until numImages) {
            imageInputStream.read(imageBuffer)
            val pixels = List(numRows) { row ->
                List(numCols) { col ->
                    (imageBuffer[row * numCols + col].toInt() and 0xFF).toDouble() / 0xff
                }
            }
            images.add(MnistImage(pixels))
        }
        imageInputStream.close()

        if (numImages != numLabels) {
            throw IOException("Number of images and labels do not match!")
        }

        return MnistData(images, labels.toList())

    } catch (e: IOException) {
        println("Error decoding MNIST data: ${e.message}")
        return null
    }
}

private fun BufferedInputStream.readInt(): Int {
    return (read() shl 24) or (read() shl 16) or (read() shl 8) or read()
}

fun decodeMNISTTraining(): MnistData {
    return decodeMnist("C:\\Users\\moham\\Desktop\\MNIST_ORG\\train-images.idx3-ubyte", "C:\\Users\\moham\\Desktop\\MNIST_ORG\\train-labels.idx1-ubyte")!!
}

fun decodeMNISTTest(): MnistData {
    return decodeMnist("", "")!!
}
