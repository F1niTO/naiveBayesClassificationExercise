package org.example

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