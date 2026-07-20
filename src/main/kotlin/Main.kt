package org.example

import kotlin.math.ln

enum class Classification {
    PLUS,
    MINUS
}

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

/**
 * Enterprise-grade Naive Bayes Classifier using Laplace Smoothing
 * and Log-Likelihood to prevent underflow.
 */
class NaiveBayesClassifier private constructor(
    val vocabulary: List<String>,
    private val classPriors: Map<Classification, Double>,
    private val wordLikelihoods: Map<Classification, Map<String, Double>>
) {
    companion object {
        fun train(dataset: List<Dataset>): NaiveBayesClassifier {
            require(dataset.isNotEmpty()) { "Training dataset cannot be empty." }

            // 1. Build unified, sorted vocabulary |V|
            val vocabulary = dataset
                .flatMap { it.tokenize() }
                .distinct()
                .sorted()

            val vocabularySize = vocabulary.size
            val totalDocuments = dataset.size.toDouble()

            // Group datasets by class once: O(N)
            val groupedByClass = dataset.groupBy { it.classification }

            // 2. Calculate Class Priors: P(C)
            val classPriors = Classification.entries.associateWith { label ->
                val classCount = groupedByClass[label]?.size ?: 0
                classCount / totalDocuments
            }

            // 3. Calculate Word Likelihoods: P(w|C) with Laplace (+1) Smoothing
            val wordLikelihoods = Classification.entries.associateWith { label ->
                val classDocs = groupedByClass[label].orEmpty()
                val allClassWords = classDocs.flatMap { it.tokenize() }

                val totalWordsInClass = allClassWords.size
                val wordFrequencies = allClassWords.groupingBy { it }.eachCount()

                // P(w | C) = (count(w, C) + 1) / (totalWordsInClass + |V|)
                vocabulary.associateWith { word ->
                    val wordCount = wordFrequencies[word] ?: 0
                    (wordCount + 1.0) / (totalWordsInClass + vocabularySize)
                }
            }

            return NaiveBayesClassifier(vocabulary, classPriors, wordLikelihoods)
        }
    }

    /**
     * Predicts the [Classification] for an unseen string using log-probabilities
     * to prevent floating-point underflow.
     */
    fun predict(text: String): Classification {
        val tokens = text.lowercase().split("\\s+".toRegex()).filter { it in vocabulary }

        return Classification.entries.maxByOrNull { label ->
            val prior = classPriors.getValue(label)
            if (prior == 0.0) Double.NEGATIVE_INFINITY
            else {
                // log(P(C)) + sum(log(P(w_i | C)))
                val logLikelihood = tokens.sumOf { token ->
                    ln(wordLikelihoods.getValue(label).getValue(token))
                }
                ln(prior) + logLikelihood
            }
        } ?: Classification.PLUS
    }

    /** Formatted output of learned likelihood probabilities */
    fun printLikelihoodTable() {
        println("%-15s | %-20s | %-20s".format("Word", "P(w | PLUS)", "P(w | MINUS)"))
        println("-".repeat(60))
        vocabulary.forEach { word ->
            val pPlus = wordLikelihoods.getValue(Classification.PLUS).getValue(word)
            val pMinus = wordLikelihoods.getValue(Classification.MINUS).getValue(word)
            println("%-15s | %-20.4f | %-20.4f".format(word, pPlus, pMinus))
        }
    }
}

fun main() {
    val datasetList = listOf(
        Dataset("I loved the movie", Classification.PLUS),
        Dataset("I hated the movie", Classification.MINUS),
        Dataset("A great movie good movie", Classification.PLUS),
        Dataset("poor acting", Classification.MINUS),
        Dataset("great acting a good movie", Classification.PLUS)
    )

    // 1. Train Classifier
    val model = NaiveBayesClassifier.train(datasetList)

    println("=== Master Vocabulary (${model.vocabulary.size} words) ===")
    println(model.vocabulary)
    println()

    // 2. Print Document Vectors
    println("=== Document Feature Vectors ===")
    datasetList.forEach { dataset ->
        val vector = dataset.toFrequencyVector(model.vocabulary)
        println("Text  : \"${dataset.text}\"")
        println("Vector: $vector")
        println("Class : ${dataset.classification}\n")
    }

    // 3. Print Learned Likelihoods P(w | C)
    println("=== Learned Conditional Probabilities ===")
    model.printLikelihoodTable()
    println()

    // 4. Test Predictions on Unseen Text
    println("=== Inference Tests ===")
    val testSamples = listOf(
        "a good movie",
        "hated acting",
        "poor movie",
        "I hate the poor acting"
    )

    testSamples.forEach { sample ->
        val prediction = model.predict(sample)
        println("Input : \"$sample\" --> Predicted: $prediction")
    }
}