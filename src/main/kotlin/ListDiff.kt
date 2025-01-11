package com.momid

abstract class ListAtom() : Atom() {
    open fun derivative(variableIndex: Int): Double {
        return 0.0
    }

    open fun eval(variableIndex: Int): Double {
        return 0.0
    }
}

//open class ListFunction(): ListAtom()

class VariableIndex(var value: Int) {
    fun clone(): VariableIndex {
        return VariableIndex(this.value)
    }
}

class VariableList(name: String = "", val variableValues: DoubleArray = doubleArrayOf(), var variableIndex: VariableIndex = VariableIndex(0), isInRespectTo: Boolean = false) : Variable(name, isInRespectTo) {
    var currentVariableIndex: Int
        get() {
            return variableIndex.value
        }
        set(value) {
            variableIndex.value = value
        }

    override fun derivative(): Atom {
        if (isInRespectTo) {
            return Constant(1.0)
        } else {
            return Constant(0.0)
        }
    }

    override fun eval(): Double {
        return variableValues[currentVariableIndex]
    }

    override fun clone(): VariableList {
        return VariableList(this.name, this.variableValues, this.variableIndex.clone(), this.isInRespectTo)
    }
}

class SumOf(val over: VariableList, val expression: (VariableIndex) -> Atom): Function() {
    override fun derivative(): Atom {
        val variableIndex = over.variableIndex
        return expression(variableIndex).derivative()
    }

    override fun eval(): Double {
        var sum = 0.0

        val currentVariableIndex = over.currentVariableIndex
        val variableIndex = over.variableIndex

        val expressionValue = expression(variableIndex)

        for (index in over.variableValues.indices) {
            variableIndex.value = index
            sum += expressionValue.eval()
        }

        over.currentVariableIndex = currentVariableIndex

        return sum
    }
}

operator fun VariableList.get(variableIndex: VariableIndex): VariableList {
    val variableList = this.clone()
    variableList.variableIndex = variableIndex
    return variableList
}
