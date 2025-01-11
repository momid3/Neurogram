package com.momid

fun printTable(data: List<Pair<String, Double>>) {
    if (data.isEmpty()) {
        println("no data to display")
        return
    }

    val maxKeyWidth = data.maxOf { it.first.length }
    val maxValueWidth = data.maxOf { "%d".format(it.second).length }

    println("+${"-".repeat(maxKeyWidth + 2)}+${"-".repeat(maxValueWidth + 2)}+")
    println("| ${"Name".padEnd(maxKeyWidth)} | ${"Value".padEnd(maxValueWidth)} |")
    println("+${"-".repeat(maxKeyWidth + 2)}+${"-".repeat(maxValueWidth + 2)}+")

    for ((key, value) in data) {
        println("| ${key.padEnd(maxKeyWidth)} | ${"%d".format(value).padStart(maxValueWidth)} |")
    }

    println("+${"-".repeat(maxKeyWidth + 2)}+${"-".repeat(maxValueWidth + 2)}+")
}

fun printTable(data: List<Pair<String, List<Double>>>, propertyNames: List<String>) {
    if (data.isEmpty() || propertyNames.isEmpty()) {
        println("data or property names are empty.")
        return
    }

    val columnWidths = mutableListOf<Int>()

    columnWidths.add(data.maxOf { it.first.length })

    for (i in propertyNames.indices) {
        val headerWidth = propertyNames[i].length
        val dataWidth = data.maxOf {
            if (i < it.second.size) {
                it.second[i].toString().length
            } else {
                0
            }
        }
        columnWidths.add(maxOf(headerWidth, dataWidth))
    }

    print("|")
    for (width in columnWidths) {
        print("-".repeat(width + 2)) // +2 for padding spaces
        print("|")
    }

    println()

    print("| ")
    print("".padEnd(columnWidths[0]))
    print(" | ")
    for (i in propertyNames.indices) {
        print(propertyNames[i].padEnd(columnWidths[i + 1]))
        print(" | ")
    }
    println()

    print("|")
    for (width in columnWidths) {
        print("-".repeat(width + 2))
        print("|")
    }
    println()

    for ((label, values) in data) {
        print("| ")
        print(label.padEnd(columnWidths[0]))
        print(" | ")
        for (i in propertyNames.indices) {
            val value = if (i < values.size) values[i].toString() else ""
            print(value.padEnd(columnWidths[i + 1]))
            print(" | ")
        }
        println()
    }

    print("|")
    for (width in columnWidths) {
        print("-".repeat(width + 2))
        print("|")
    }

    println()
}
