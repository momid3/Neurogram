package com.momid

abstract class Atom() {
    open fun derivative(): Atom {
        return this
    }

    open fun eval(): Double {
        return 0.0
    }

    open fun clone(): Atom {
        if (this is Variable) {
            return this.clone()
        } else {
            return this
        }
    }
}

abstract class Function(val parameters: ArrayList<Atom> = ArrayList()) : Atom()

class Constant(val value: Double) : Atom() {
    override fun derivative(): Atom {
        return Constant(0.0)
    }

    override fun eval(): Double {
        return this.value
    }
}

open class Variable(val name: String = "", var isInRespectTo: Boolean = false, var value: Double? = null) : Atom() {
    override fun derivative(): Atom {
        if (isInRespectTo) {
            return Constant(1.0)
        } else {
            return Constant(0.0)
        }
    }

    override fun eval(): Double {
        return this.value ?: throw (Throwable("there should have been a substitution for this variable"))
    }

    override fun clone(): Variable {
        return Variable(this.name, this.isInRespectTo, this.value)
    }
}

class Sum(val expressions: ArrayList<Atom>): Function(expressions) {
    override fun derivative(): Atom {
        val addition = Sum(ArrayList())
        expressions.forEach {
            addition.expressions.add(it.derivative())
        }
        return addition
    }

    override fun eval(): Double {
        return expressions.sumOf {
            it.eval()
        }
    }
}

class Subtraction(val param0: Atom, val param1: Atom) : Function() {
    override fun derivative(): Atom {
        return param0.derivative() - param1.derivative()
    }

    override fun eval(): Double {
        return param0.eval() - param1.eval()
    }
}

operator fun Atom.minus(other: Atom): Subtraction {
    return Subtraction(this, other)
}

operator fun Double.minus(other: Atom): Subtraction {
    return Subtraction(Constant(this), other)
}

operator fun Atom.minus(other: Double): Subtraction {
    return Subtraction(this, Constant(other))
}

operator fun Atom.unaryMinus(): Subtraction {
    return Subtraction(Constant(0.0), this)
}

class Multiplication(val param0: Atom, val param1: Atom) : Function() {
    override fun derivative(): Atom {
        if (param0.isConstantOrVariable() && !param1.isConstantOrVariable()) {
            return param0 * param1.derivative()
        } else if (!param0.isConstantOrVariable() && param1.isConstantOrVariable()) {
            return param1 * param0.derivative()
        } else if (param0.isConstantOrVariable() && param1.isConstantOrVariable()) {
            return param0 * param1
        } else {
            return param0 * param1.derivative() + param0.derivative() * param1
        }
    }

    override fun eval(): Double {
        return param0.eval() * param1.eval()
    }
}

class Division(val param0: Atom, val param1: Atom) : Function() {
    override fun derivative(): Atom {
        return when {
            param0.isConstantOrVariable() && param1.isConstantOrVariable() -> Constant(0.0)
            param0.isConstantOrVariable() -> -((param0 * param1.derivative()) / (param1 * param1))
            param1.isConstantOrVariable() -> param0.derivative() / param1
            else -> (param0.derivative() * param1 - param0 * param1.derivative()) / (param1 * param1)
        }
    }

    override fun eval(): Double {
        var denominator = param1.eval()
//        if (denominator == 0.0) {
//            throw ArithmeticException("Division by zero")
//        }
        if (denominator >= 0 && denominator < 0.00000001) {
            denominator = 0.00000001
        }

        if (denominator < 0 && denominator > - 0.00000001) {
            denominator = -0.00000001
        }

        return param0.eval() / denominator
    }
}

operator fun Atom.div(other: Atom): Division {
    return Division(this, other)
}

operator fun Double.div(other: Atom): Division {
    return Division(Constant(this), other)
}

operator fun Atom.div(other: Double): Division {
    return Division(this, Constant(other))
}

fun Atom.isConstantOrVariable(): Boolean {
    return this is Constant || (this is Variable && !this.isInRespectTo)
}

operator fun Atom.plus(other: Atom): Sum {
    return Sum(arrayListOf(this, other))
}

operator fun Sum.plus(other: Atom): Sum {
    this.expressions.add(other)
    return this
}

operator fun Sum.plus(other: Sum): Sum {
    this.expressions.addAll(other.expressions)
    return this
}

operator fun Atom.times(other: Atom): Multiplication {
    return Multiplication(this, other)
}

operator fun Double.times(other: Atom): Multiplication {
    return Multiplication(Constant(this), other)
}

operator fun Double.plus(other: Atom): Sum {
    return Sum(arrayListOf(Constant(this), other))
}

operator fun Atom.plus(other: Double): Sum {
    return Sum(arrayListOf(this, Constant(other)))
}

fun Atom.eval(values: List<Pair<Variable, Double>>): Double {
    values.forEach {
        val (variable, value) = it
        variable.value = value
    }

    return this.eval()
}

fun Atom.eval(vararg values: Pair<Variable, Double>): Double {
    values.forEach {
        val (variable, value) = it
        variable.value = value
    }

    return this.eval()
}

fun withVariables(variables: (variable: Variable) -> Unit) {
    val variable = Variable()
    variable.isInRespectTo = true
    variables(variable)
}

fun withVariables(variables: (variable0: Variable, variable1: Variable) -> Unit) {
    val variable0 = Variable()
    val variable1 = Variable()
    variables(variable0, variable1)
}

fun withVariables(variables: (variable0: Variable, variable1: Variable, variable2: Variable) -> Unit) {
    val variable0 = Variable()
    val variable1 = Variable()
    val variable2 = Variable()
    variables(variable0, variable1, variable2)
}

fun Atom.withRespectTo(variable: Variable): Atom {
    variable.isInRespectTo = true
    return this
}

fun main() {
//    val x = Variable("x")
//    val function = Constant(3.0) * x + Constant(3.0)
//    println(function.derivative().eval(
//        x to 0.0
//    ))

    withVariables { x ->
        val function = x * x * x
        val derivative = function.derivative()

        println("" + derivative.eval(x to 3.0))
        println("" + derivative.eval(x to 0.0))
        println("" + derivative.eval(x to 7.0))
    }

    withVariables { x, y, z ->
        val function = 3.0 * x * y * z + 3.0
        val derivative = function.withRespectTo(x).derivative()
            .eval(
                x to 0.0,
                y to 3.0,
                z to 3.0
            )

        println("" + derivative)
    }
}
