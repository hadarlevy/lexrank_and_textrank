# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
import matplotlib.pyplot as plt
import networkx as nx


def similarity(sentence1, sentence2):
    vectorizer = CountVectorizer().fit_transform([sentence1, sentence2])
    similarity = cosine_similarity(vectorizer[0], vectorizer[1])
    return similarity[0][0]


def pagerank(similarities):
    n = similarities.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            if similarities[i, j] != 0:
                G.add_edge(i, j)
    scores = nx.pagerank(G)

    return list(scores.values())


def textrank(document):
    # Convert the document into a list of sentences
    sentences = document.split(".")
    n = len(sentences)

    # Create a matrix of similarities between sentences
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                similarities[i, j] = similarity(sentences[i], sentences[j])

    # Create a matrix of similarities between sentences
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                similarities[i, j] = similarity(sentences[i], sentences[j])

    # Apply the PageRank algorithm to the similarity matrix
    pagerank_scores = np.array(pagerank(similarities))

    # Sort the sentences by PageRank score
    ranked_sentences = sorted(zip(range(n), pagerank_scores), key=lambda x: x[1], reverse=True)

    # Return the top-ranked sentences
    return [sentences[i] for i, score in ranked_sentences[:3]]


def show_textrank_graph(document):
    sentences = document.split(".")
    n = len(sentences)-1

    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                similarities[i, j] = similarity(sentences[i], sentences[j])
    pagerank_scores = np.array(pagerank(similarities))
    ranked_sentences = sorted(zip(range(n), pagerank_scores), key=lambda x: x[1], reverse=True)
    G = nx.Graph()
    for i, score in ranked_sentences:
        G.add_node(i, weight=score)
    for i in range(n):
        for j in range(n):
            if similarities[i, j] != 0:
                G.add_edge(i, j)
    pos = nx.spring_layout(G)
    labels = {i: f'{sentences[i]} - {pagerank_scores[i]}' for i in range(n)}
    nx.draw(G, pos, with_labels=False, node_size=[G.nodes[i]['weight'] * 10000 for i in G.nodes()], alpha=0.8)
    nx.draw(G, pos, with_labels=False, node_size=[G.nodes[i]['weight'] * 10000 for i in G.nodes()], alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    plt.show()


def lexrank(sentences, summary_length):
    # Step 1: Preprocessing
    # Convert the set of sentences into a list of words
    word_list = [sentence.split() for sentence in sentences]

    # Flatten the list of words into a single list of strings
    words = list(chain.from_iterable(word_list))

    # Step 2: Construction of the graph
    # Use a CountVectorizer to convert the list of words into a matrix of word counts
    vectorizer = CountVectorizer().fit_transform(words)

    # Compute the similarity matrix
    similarity_matrix = cosine_similarity(vectorizer)

    # Create the graph
    graph = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                graph[i][j] = similarity_matrix[i][j]

    # Step 3: Calculation of the LexRank score
    # Use the power method to compute the LexRank score for each sentence
    eigenvector = np.ones(len(sentences)) / len(sentences)
    while True:
        prev_eigenvector = eigenvector
        eigenvector = np.dot(graph, eigenvector)
        if np.linalg.norm(eigenvector) > 0:
            eigenvector = eigenvector / np.linalg.norm(eigenvector)
        else:
            eigenvector = np.ones(len(sentences)) / len(sentences)
        if np.abs(np.subtract(prev_eigenvector, eigenvector)).mean() < 0.0001:
            break

    # Step 4: Selection of the summary
    # Sort the sentences according to their LexRank score
    ranked_sentences = [(eigenvector[i], sentence) for i, sentence in enumerate(sentences)]
    ranked_sentences.sort(key=lambda x: x[0], reverse=True)

    # Select the top-ranked sentences to form the summary
    summary = [ranked_sentences[i][1] for i in range(summary_length)]

    return summary,ranked_sentences


def visualize_lexrank(ranked_sentences):
    # Create a graph object
    G = nx.Graph()

    # Add nodes to the graph
    for i, (score, sentence) in enumerate(ranked_sentences):
        G.add_node(i, label=sentence, score=score)

    # Add edges to the graph
    for i in range(len(ranked_sentences)):
        for j in range(i+1, len(ranked_sentences)):
            G.add_edge(i, j)

    # Draw the graph
    pos = nx.spring_layout(G)
    labels = {i: f'{data["label"]} ({data["score"]:.4f})' for i, data in G.nodes(data=True)}
    nx.draw_networkx_nodes(G, pos, node_size=[data['score']*1000 for i, data in G.nodes(data=True)], node_color='red')
    nx.draw_networkx_edges(G, pos, width=[0.5 for u,v in G.edges()])
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='blue')
    plt.title('LexRank')
    plt.show()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sentences = [
        "Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective deep in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the structured part."
    ]
    summary_length =1  # Number of sentences to include in the summary
    summary, ranked_sentences = lexrank(sentences, summary_length)
    print(summary)
    visualize_lexrank(ranked_sentences)
    # Compute the LexRank scores
    # scores = [ranked_sentences[i][0] for i in range(len(sentences))]
    # print(ranked_sentences)


    document ="Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective deep in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the structured part."
    show_textrank_graph(document)
    print(textrank(document))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
