import string
from collections import Counter
from underthesea import word_tokenize
from scipy.sparse import csr_matrix

class VietnameseCountVectorizer:
    def __init__(self, n_gram_range=(1, 1)):
        """
        Initialize Vietnamese CountVectorizer Vectorizer.

        Attributes:
        - vocabulary_: A dictionary mapping words to their indices in the vocabulary.
        - document_counts: A Counter object storing the document frequency of each word.
        """
        self.vocabulary_ = {}
        self.document_counts = {}
        self.n_gram_range = n_gram_range
        
    def fit_transform(self, corpus):
        """
        Fit the vectorizer on the given corpus and transform it into a Count matrix.

        Parameters:
        - corpus: List of text documents.

        Returns:
        - count_matrix: Sparse Count matrix representing the input corpus.
        """
        self.fit(corpus)
        return self.transform(corpus)

    def fit(self, corpus):
        """
        Fit the vectorizer on the given corpus.

        Parameters:
        - corpus: List of text documents.
        """
        # Tokenize each document in the corpus and count word frequencies
        tokenized_corpus = [self.tokenize(text) for text in corpus]
        self.document_counts = dict(Counter(word for tokens in tokenized_corpus for word in tokens))

        # Create a vocabulary mapping words to indices
        set_vocab = set(self.document_counts.keys())
        self.vocabulary_ = {word: idx for idx, word in enumerate(set_vocab)}

    def transform(self, corpus):
        """
        Transform the given corpus into a Count matrix.

        Parameters:
        - corpus: List of text documents.

        Returns:
        - count_matrix: Sparse Count matrix representing the input corpus.
        """
        n_vocab = len(self.vocabulary_)
        n_sents = len(corpus)
        rows = []
        cols = []
        data = []

        # Iterate through each document in the corpus
        for i in range(n_sents):
            tokens = self.tokenize(corpus[i])
            
            # Convert tokens to indices using the vocabulary and count frequencies
            for token in tokens:
                if token in self.vocabulary_:
                    rows.append(i)
                    cols.append(self.vocabulary_[token])
                    data.append(self.document_counts[token])

        # Create a sparse count matrix using the collected data
        count_matrix = csr_matrix((data, (rows, cols)), shape=(n_sents, n_vocab))
        return count_matrix

    def tokenize(self, text):
        """
        Tokenize a given text and generate n-grams.

        Parameters:
        - text: Input text.

        Returns:
        - n_grams: List of tokens and n-grams.
        """
        tokens = word_tokenize(self.remove_punctuation(text))

        # Generate n-grams
        n_grams = []
        min_n, max_n = self.n_gram_range
        for n in range(min_n, max_n + 1):
            if n == 1:
                n_grams.extend(tokens)  # Unigrams
            else:
                n_grams.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))  # N-grams

        return n_grams
    
    def remove_punctuation(self, input_string):
        """
        Remove punctuation from a given string.

        Parameters:
        - input_string: Input string.

        Returns:
        - result_string: String with punctuation removed.
        """
        translation_table = str.maketrans("", "", string.punctuation)
        result_string = input_string.translate(translation_table)

        return result_string