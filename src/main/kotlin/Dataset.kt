package org.example

data class Dataset(
    val text: String,
    val classification: Classification
) {
    /** Tokenizes and lowercases text in a single pass */
    fun tokenize(): List<String> =
        text.lowercase()
            .split("\\s+".toRegex())
            .filter { it.isNotBlank() }

    /** Generates a Term Frequency vector against a master [vocabulary] */
    fun toFrequencyVector(vocabulary: List<String>): List<Int> {
        val wordCounts = tokenize().groupingBy { it }.eachCount()
        return vocabulary.map { wordCounts[it] ?: 0 }
    }
}