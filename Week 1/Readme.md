Let's start with the most basic question from which the whole notion of Generative AI and LLM's came into picture.
# What are Transformers?

Transformers are a type of neural network architecture that are really good at understanding and generating text. They became really popular because they can process a lot of data at once and learn patterns from it. One of the important parts of transformers is something called self-attention, which helps the model focus on different parts of the text and understand the relationships between words.

# What is self-attention and multi-headed self-attention that made transformer's architecture state  of the art?

Self-attention is a key concept in transformer networks. It allows the model to focus on different parts of the input text and understand the relationships between words.

To understand self-attention, let's imagine we have a sentence: "The cat sat on the mat." In self-attention, each word in the sentence is represented as a vector. These vectors are used to calculate attention scores between words.

The attention scores determine how much each word should pay attention to other words in the sentence. For example, in the sentence "The cat sat on the mat," the word "cat" might have a high attention score with "mat" because they are related in the context of the sentence.

Once the attention scores are calculated, they are used to weight the importance of each word's vector. The words with higher attention scores will have more influence on the final representation of the sentence.

By using self-attention, the transformer model can capture the dependencies and relationships between words in a sentence. It allows the model to understand the context and meaning of the text, which is crucial for tasks like language translation, text generation, and sentiment analysis.

Multi-headed self-attention is an extension of the concept of self-attention in transformer networks. It allows the model to capture different types of relationships and dependencies between words by using multiple sets of attention weights.

In a transformer model, self-attention is performed by calculating attention scores between each pair of words in a sentence. These attention scores determine how much each word should pay attention to other words. However, using a single set of attention weights may not capture all the different types of relationships in the text.

That's where multi-headed self-attention comes in. Instead of using a single set of attention weights, the model uses multiple sets of attention weights, known as attention heads. Each attention head focuses on capturing a different aspect or pattern in the text.

For example, one attention head might focus on capturing syntactic relationships between words, while another attention head might focus on capturing semantic relationships. By having multiple attention heads, the model can capture different types of information simultaneously.

After calculating the attention scores for each attention head, the results are combined to create a final representation of the input text. This final representation incorporates the different perspectives and patterns captured by each attention head.

By using multi-headed self-attention, transformer models can capture a richer and more nuanced understanding of the relationships between words in a sentence. This allows the model to perform better on various natural language processing tasks, such as machine translation, text summarization, and sentiment analysis.

In summary, self-attention in transformer networks enables the model to focus on different parts of the input text and understand the relationships between words, helping it to better understand and generate text. Multi-headed self-attention in transformer networks involves using multiple sets of attention weights to capture different types of relationships and dependencies between words. It enhances the model's ability to understand and process text by incorporating multiple perspectives and patterns.

# How are the attention scores calculated in the self-attention?

In self-attention, the attention scores are calculated using three sets of vectors: the query vectors, the key vectors, and the value vectors. These vectors are derived from the input text and are used to determine the importance or relevance of each word to other words in the sequence.

Here's a step-by-step explanation of how the attention scores are calculated:

    Query Vectors: Each word in the input sequence is transformed into a query vector. These query vectors represent the words' current state and are used to determine how much attention each word should pay to other words.

    Key Vectors: Similarly, each word is transformed into a key vector. These key vectors represent the words' context and are used to measure the similarity or compatibility between words.

    Attention Scores: To calculate the attention scores, the dot product is taken between each query vector and the corresponding key vectors. The dot product measures the similarity between the query and key vectors.

    Scaling: To prevent the attention scores from becoming too large, they are scaled by dividing them by the square root of the dimension of the key vectors.

    Softmax: The scaled attention scores are then passed through a softmax function, which normalizes the scores and ensures that they sum up to 1. This step determines the importance or weight of each word in relation to other words.

    Weighted Sum: Finally, the softmax-normalized attention scores are used to weight the value vectors. The value vectors represent the words' content or information. The weighted sum of the value vectors gives the final representation of each word, taking into account the attention scores.

Let's go through the step-by-step calculation of attention scores using the example sentence "The cat sat on the mat" in self-attention.

    Query Vectors: Each word in the input sequence is transformed into a query vector. Let's assume the query vectors for the words are as follows:
        "The": [0.2, 0.3, 0.4]
        "cat": [0.1, 0.5, 0.2]
        "sat": [0.3, 0.2, 0.1]
        "on": [0.4, 0.1, 0.3]
        "the": [0.2, 0.4, 0.1]
        "mat": [0.5, 0.3, 0.2]

    Key Vectors: Similarly, each word is transformed into a key vector. Let's assume the key vectors for the words are as follows:
        "The": [0.3, 0.2, 0.1]
        "cat": [0.4, 0.1, 0.3]
        "sat": [0.2, 0.3, 0.4]
        "on": [0.1, 0.5, 0.2]
        "the": [0.3, 0.2, 0.4]
        "mat": [0.2, 0.4, 0.1]

    Attention Scores: To calculate the attention scores, we take the dot product between each query vector and the corresponding key vectors. Let's calculate the attention scores for each word:

        "The":
            Attention score with "The": (0.2 * 0.3) + (0.3 * 0.2) + (0.4 * 0.1) = 0.13
            Attention score with "cat": (0.2 * 0.4) + (0.3 * 0.1) + (0.4 * 0.3) = 0.25
            Attention score with "sat": (0.2 * 0.2) + (0.3 * 0.3) + (0.4 * 0.4) = 0.29
            Attention score with "on": (0.2 * 0.1) + (0.3 * 0.5) + (0.4 * 0.2) = 0.26
            Attention score with "the": (0.2 * 0.3) + (0.3 * 0.2) + (0.4 * 0.4) = 0.23
            Attention score with "mat": (0.2 * 0.2) + (0.3 * 0.4) + (0.4 * 0.1) = 0.19

        "cat":
            Attention score with "The": (0.1 * 0.3) + (0.5 * 0.2) + (0.2 * 0.1) = 0.14
            Attention score with "cat": (0.1 * 0.4) + (0.5 * 0.1) + (0.2 * 0.3) = 0.16
            Attention score with "sat": (0.1 * 0.2) + (0.5 * 0.3) + (0.2 * 0.4) = 0.23
            Attention score with "on": (0.1 * 0.1) + (0.5 * 0.5) + (0.2 * 0.2) = 0.3
            Attention score with "the": (0.1 * 0.3) + (0.5 * 0.2) + (0.2 * 0.4) = 0.19
            Attention score with "mat": (0.1 * 0.2) + (0.5 * 0.4) + (0.2 * 0.1) = 0.21

        Similarly, we calculate the attention scores for the remaining words.

    Scaling: To prevent the attention scores from becoming too large, we scale them by dividing them by the square root of the dimension of the key vectors. Let's assume the dimension of the key vectors is 3. So, we divide each attention score by sqrt(3) ≈ 1.732.

    Softmax: After scaling, we apply the softmax function to normalize the attention scores. The softmax function ensures that the scores sum up to 1 and represent the weights or importance of each word. Let's assume the softmax-normalized attention scores for each word are as follows:
        "The": 0.13 / 1.732 ≈ 0.075
        "cat": 0.16 / 1.732 ≈ 0.092
        "sat": 0.23 / 1.732 ≈ 0.133
        "on": 0.3 / 1.732 ≈ 0.173
        "the": 0.19 / 1.732 ≈ 0.11
        "mat": 0.21 / 1.732 ≈ 0.121

    Weighted Sum: Finally, we use the softmax-normalized attention scores to weight the value vectors. The value vectors represent the words' content or information. Let's assume the value vectors for the words are as follows:
        "The": [0.7, 0.5, 0.3]
        "cat": [0.4, 0.6, 0.2]
        "sat": [0.2, 0.3, 0.5]
        "on": [0.1, 0.8, 0.4]
        "the": [0.6, 0.4, 0.2]
        "mat": [0.3, 0.2, 0.1]

    We calculate the weighted sum of the value vectors using the softmax-normalized attention scores:

        Weighted sum for "The": (0.075 * [0.7, 0.5, 0.3]) + (0.092 * [0.4, 0.6, 0.2]) + (0.133 * [0.2, 0.3, 0.5]) + (0.173 * [0.1, 0.8, 0.4]) + (0.11 * [0.6, 0.4, 0.2]) + (0.121 * [0.3, 0.2, 0.1]) = [0.365, 0.437, 0.244]

        Similarly, we calculate the weighted sum for the remaining words.

By following these steps, we calculate the attention scores in self-attention. The resulting weighted sums represent the final representation of each word, taking into account the attention scores and the content of the words. This allows the model to understand the context and dependencies within the text.

By calculating attention scores in this way, self-attention allows the model to focus on different parts of the input sequence and capture the relationships between words. It enables the model to understand the context and dependencies within the text, which is crucial for various natural language processing tasks.

In summary, attention scores in self-attention are calculated by taking the dot product between query vectors and key vectors, scaling the scores, applying softmax to normalize them, and using the resulting attention scores to weight the value vectors. This process allows the model to determine the importance and relevance of each word in relation to other words in the sequence.

# Why does the query vector and key vector for the word "The" and "the" differ even when both the words are same? Does the vector values change by the upper case or lower-case of a letter? Does this really bring change in the how the model understands the context of the sentence?

The query vector and key vector for the words "The" and "the" may differ because the model treats them as different tokens. In natural language processing, words are often tokenized, which means they are split into individual units for processing. In this case, "The" and "the" are considered as separate tokens because they have different capitalization.

The vector values for each token are learned during the training process and are not directly influenced by the case of the letters. However, the model can learn different representations for words with different capitalization based on the patterns it observes in the training data.

The distinction between "The" and "the" can be important in some contexts. For example, "The" is typically used as the beginning of a sentence, while "the" is used within a sentence. The model can learn to capture these patterns and use them to understand the context of a sentence.

Overall, the model's ability to understand the context of a sentence is not solely dependent on the case of individual letters, but rather on the patterns and relationships it learns from the training data.