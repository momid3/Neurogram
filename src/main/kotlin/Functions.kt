package com.momid

import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.assertEquals

class Condition(val atom: Atom, val param: (Atom) -> Atom) : Atom() {
    override fun derivative(): Atom {
        throw (Throwable("not derivativable"))
    }

    override fun eval(): Double {
        val paramValue = param(atom)
        return paramValue.eval()
    }
}

class Relu(val param: Atom) : Function() {
    override fun derivative(): Atom {
        return Condition(param) {
            val eval = it.eval()
            if (eval > 0) {
                it.derivative()
            } else {
                Constant(0.0)
            }
        }
    }

    override fun eval(): Double {
        val eval = param.eval()
        if (eval > 0) {
            return eval
        } else {
            return 0.0
        }
    }
}

class Exp(val param: Atom) : Function() {
    override fun derivative(): Atom {
        return Exp(param) * param.derivative()
    }

    override fun eval(): Double {
        var value = param.eval()
//        if (value > 17.0) {
//            value = 17.0
//        }
        return exp(value)
    }
}

class Log(val param: Atom) : Function() {
    override fun derivative(): Atom {
        return when {
            param.isConstantOrVariable() -> Constant(0.0)
            else -> param.derivative() / param
        }
    }

    override fun eval(): Double {
        var value = param.eval()
        if (value < 0.00000000001) {
            value = 0.00000000001
        }
        if (value <= 0.0) {
            throw ArithmeticException("logarithm of non-positive value")
        }
        return ln(value)
    }
}

fun softmax(param: VariableList): Function {
    return Exp(param) / SumOf(param) { i ->
        Exp(param[i])
    }
}

val CrossEntropyLoss = LossFunction() { output, predictedOutput ->
    - SumOf(predictedOutput) { i ->
        output[i] * Log(predictedOutput[i])
    }
}

fun softmaxDerivative(output: DoubleArray, o: Int, i: Int): Double {
    require(o in output.indices) { "Output index 'o' must be within the bounds of the output array" }
    require(i in output.indices) { "Input index 'i' must be within the bounds of the output array" }

    val s_o = softmax(output)[o] // Softmax output at index o

    return if (i == o) {
        s_o * (1 - s_o)
    } else {
        -s_o * softmax(output)[i]
    }
}

fun softmax(input: DoubleArray): DoubleArray {
    val expSum = input.sumOf { exp(it) }
    return input.map { exp(it) / expSum }.toDoubleArray()
}

fun main() {
    testSoftmax()

    // Test cases for basic operations
    testBasicOperations()

    // Test cases for Log, Exp, and Relu
    testLogExpRelu()

    // Test cases for combinations of functions
    testCombinedFunctions()

    // Test cases for VariableList and SumOf (if applicable)
    testVariableListAndSumOf()

    // Test cases for Softmax and CrossEntropyLoss
    testSoftmaxAndCrossEntropy()

    testSoftmaxAndCrossEntropyWithDerivatives()

    println("All tests passed!")
}

fun testBasicOperations() {
    withVariables { x, y ->
        val a = 2.0
        val b = 3.0

        val sum = x + y
        assertEquals(a + b, sum.eval(x to a, y to b), 1e-10, "Sum failed")

        val subtraction = x - y
        assertEquals(a - b, subtraction.eval(x to a, y to b), 1e-10, "Subtraction failed")

        val multiplication = x * y
        assertEquals(a * b, multiplication.eval(x to a, y to b), 1e-10, "Multiplication failed")

        val division = x / y
        assertEquals(a / b, division.eval(x to a, y to b), 1e-10, "Division failed")

        val divisionDerivative = division.withRespectTo(x).derivative()
        assertEquals(1.0 / b, divisionDerivative.eval(x to a, y to b), 1e-10, "Division derivative failed")
    }
}

fun testSoftmax() {
    withVariables { x ->
        val a = exp(2.0) + exp(3.0)
        x.isInRespectTo = true
        val function = Exp(x) / (Exp(x) + a)
        println(function.derivative().eval(x to 1.0))
    }
}

fun testLogExpRelu() {
    withVariables { x ->
        val a = 2.0

        val log = Log(x)
        assertEquals(ln(a), log.eval(x to a), 1e-10, "Log failed")

        val logDerivative = log.withRespectTo(x).derivative()
        assertEquals(1.0 / a, logDerivative.eval(x to a), 1e-10, "Log derivative failed")

        val exp = Exp(x)
        assertEquals(exp(a), exp.eval(x to a), 1e-10, "Exp failed")

        val expDerivative = exp.withRespectTo(x).derivative()
        assertEquals(exp(a), expDerivative.eval(x to a), 1e-10, "Exp derivative failed")

        val relu = Relu(x)
        assertEquals(a, relu.eval(x to a), 1e-10, "Relu failed (positive case)")
        assertEquals(0.0, relu.eval(x to -a), 1e-10, "Relu failed (negative case)")

        val reluDerivative = relu.withRespectTo(x).derivative()
        assertEquals(1.0, reluDerivative.eval(x to a), 1e-10, "Relu derivative failed (positive case)")
        assertEquals(0.0, reluDerivative.eval(x to -a), 1e-10, "Relu derivative failed (negative case)")
    }
}

fun testCombinedFunctions() {
    withVariables { x, y ->
        val a = 2.0
        val b = 3.0

        val combined1 = Exp(Log(x))
        assertEquals(a, combined1.eval(x to a), 1e-10, "Exp(Log(x)) failed")

        val combined2 = Log(Exp(x))
        assertEquals(a, combined2.eval(x to a), 1e-10, "Log(Exp(x)) failed")

        val combined3 = Relu(x * y - 5.0)
        assertEquals(1.0, combined3.eval(x to a, y to b), 1e-10, "Relu(x*y - 5) failed (positive case)")
        assertEquals(0.0, combined3.eval(x to 1.0, y to 2.0), 1e-10, "Relu(x*y - 5) failed (negative case)")

        val combined3Derivative = combined3.withRespectTo(x).derivative()
        assertEquals(b, combined3Derivative.eval(x to a, y to b), 1e-10, "Relu(x*y - 5) derivative failed (positive case)")
        assertEquals(0.0, combined3Derivative.eval(x to 1.0, y to 2.0), 1e-10, "Relu(x*y - 5) derivative failed (negative case)")
    }
}

fun testVariableListAndSumOf() {
    val values = doubleArrayOf(1.0, 2.0, 3.0)
    val variableList = VariableList(variableValues = values)
    val sumOf = variableList * SumOf(variableList) { i ->
        variableList[i]
    }
    variableList.isInRespectTo = true
    variableList.currentVariableIndex = 0

    assertEquals(7.0, sumOf.derivative().eval(), 1e-10, "SumOf failed")

}

fun testSoftmaxAndCrossEntropy() {
    val values = doubleArrayOf(1.0, 2.0, 3.0)
    val variableList = VariableList(variableValues = values)

    val softmax = softmax(variableList)
    val softmaxEval = softmax.eval()

    // Manual softmax calculation for comparison
    val expValues = values.map { exp(it) }
    val sumExp = expValues.sum()
    val manualSoftmax = expValues.map { it / sumExp }

    for (i in values.indices) {
        variableList.currentVariableIndex = i
        assertEquals(manualSoftmax[i], softmax.eval(), 1e-10, "Softmax failed for index $i")
    }

    val outputValues = doubleArrayOf(0.2, 0.3, 0.5)
    val predictedValues = doubleArrayOf(0.1, 0.6, 0.3)
    val output = VariableList(variableValues = outputValues)
    val predictedOutput = VariableList(variableValues = predictedValues)

    val crossEntropy = CrossEntropyLoss.function(output, predictedOutput)
    val crossEntropyEval = crossEntropy.eval()

    var manualCrossEntropy = 0.0
    for (i in outputValues.indices) {
        output.currentVariableIndex = i
        predictedOutput.currentVariableIndex = i
        manualCrossEntropy += outputValues[i] * ln(predictedValues[i])
    }
    manualCrossEntropy = - manualCrossEntropy

    assertEquals(manualCrossEntropy, crossEntropyEval, 1e-10, "CrossEntropy failed")
}

fun testSoftmaxAndCrossEntropyWithDerivatives() {
    val values = doubleArrayOf(1.0, 2.0, 3.0)
    val variableList = VariableList(variableValues = values)

    val softmax = softmax(variableList)
    val softmaxDerivative = softmax.derivative()

    val expValues = values.map { exp(it) }
    val sumExp = expValues.sum()
    val manualSoftmax = expValues.map { it / sumExp }

    for (i in values.indices) {
        variableList.currentVariableIndex = i
        variableList.isInRespectTo = false
        assertEquals(manualSoftmax[i], softmax.eval(), 1e-10, "Softmax failed for index $i")
    }

    val softmaxDerivatives = mutableListOf<Double>()
    for (i in values.indices) {
        variableList.currentVariableIndex = i
        variableList.isInRespectTo = true
        softmaxDerivatives.add(softmaxDerivative.eval())
        variableList.isInRespectTo = false
    }

    val manualJacobianDiagonal = manualSoftmax.map { it * (1 - it) }

    for (i in values.indices) {
        assertEquals(
            manualJacobianDiagonal[i],
            softmaxDerivatives[i],
            1e-10,
            "Softmax derivative failed for index $i"
        )
    }

    val outputValues = doubleArrayOf(0.2, 0.3, 0.5)
    val predictedValues = doubleArrayOf(0.1, 0.6, 0.3)
    val output = VariableList(variableValues = outputValues)
    val predictedOutput = VariableList(variableValues = predictedValues)

    val crossEntropy = CrossEntropyLoss.function(output, predictedOutput)

    var manualCrossEntropy = 0.0
    for (i in outputValues.indices) {
        manualCrossEntropy -= outputValues[i] * ln(predictedValues[i])
    }
    assertEquals(manualCrossEntropy, crossEntropy.eval(), 1e-10, "CrossEntropy failed")

    for (i in outputValues.indices) {
        predictedOutput.currentVariableIndex = i
        predictedOutput.isInRespectTo = true
        val crossEntropyDerivative = crossEntropy.derivative()

        val expectedDerivative = -outputValues[i] / predictedValues[i]
        assertEquals(
            expectedDerivative,
            crossEntropyDerivative.eval(),
            1e-10,
            "CrossEntropy derivative failed for index $i"
        )
        predictedOutput.isInRespectTo = false
    }
}
