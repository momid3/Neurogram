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

class Index(var variableIndex: VariableIndex = VariableIndex(0), val withRespectToIndex: VariableIndex = VariableIndex(- 3))

class VariableList(name: String = "", val variableValues: DoubleArray = doubleArrayOf(), var index: Index = Index(), isInRespectTo: Boolean = false) : Variable(name, isInRespectTo) {
    var currentVariableIndex: Int
        get() {
            return index.variableIndex.value
        }
        set(value) {
            index.variableIndex.value = value
        }

    var currentWithRespectToIndex: Int
        get() {
            return index.withRespectToIndex.value
        }
        set(value) {
            index.withRespectToIndex.value = value
        }

    override fun derivative(): Atom {
        return Condition(this) {
            if (this.index.withRespectToIndex.value == - 3) {
                if (isInRespectTo) {
                    Constant(1.0)
                } else {
                    Constant(0.0)
                }
            } else {
                if (isInRespectTo && this.index.variableIndex.value == this.index.withRespectToIndex.value) {
                    Constant(1.0)
                } else {
                    Constant(0.0)
                }
            }
        }
    }

    override fun eval(): Double {
        return variableValues[currentVariableIndex]
    }

    override fun clone(): VariableList {
        return VariableList(this.name, this.variableValues, Index(this.index.variableIndex.clone(), this.index.withRespectToIndex.clone()), this.isInRespectTo)
    }
}

class SumOf(val over: VariableList, val expression: (Index) -> Atom): Function() {
    override fun derivative(): Atom {
        if (over.index.withRespectToIndex.value == - 3) {
            val variableIndex = over.index
            return expression(variableIndex).derivative()
        } else {
            val variableIndex = Index(over.index.withRespectToIndex, over.index.withRespectToIndex)
            return expression(variableIndex).derivative()
        }
    }

    override fun eval(): Double {
        var sum = 0.0

        val currentVariableIndex = over.currentVariableIndex
        val variableIndex = over.index

        val expressionValue = expression(variableIndex)

        for (index in over.variableValues.indices) {
            variableIndex.variableIndex.value = index
            sum += expressionValue.eval()
        }

        over.currentVariableIndex = currentVariableIndex

        return sum
    }
}

operator fun VariableList.get(index: Index): VariableList {
    val variableList = this.clone()
    variableList.index = index
    return variableList
}
