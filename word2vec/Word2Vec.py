from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import numpy as np

class studentFeedbackWord2Vec:
    def __init__(self, sentences=None, model_type='skipgram', vector_size=200, window=10, min_count=5, workers=4, epochs = 10):
        """
        Initialize Word2Vec model.

        Parameters:
        - model_type: 'skipgram' or 'cbow' to choose between Skip-Gram and CBOW.
        - size: Dimensionality of the word vectors.
        - window: Maximum distance between the current and predicted word within a sentence.
        - min_count: Ignores all words with a total frequency lower than this.
        - workers: Number of CPU cores to use when training the model.
        """
        
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model_type = model_type
        self.epochs = epochs
        
        if model_type == 'skipgram':
            self.model = Word2Vec(sentences, sg = 1, vector_size=self.vector_size, window=window, min_count=min_count, workers=workers, epochs = self.epochs)
        elif model_type == 'cbow':
            self.model = Word2Vec(sentences, sg = 0, vector_size=self.vector_size, window=window, min_count=min_count, workers=workers, epochs = self.epochs)
        elif model_type == 'fasttext':
            self.model = FastText(sentences, sg = 1, vector_size=self.vector_size, window=window, min_count=min_count, workers=workers, epochs = self.epochs)
            
    def save(self, filepath):
        """
        Save the trained model to a file.

        Parameters:
        - filepath: Path to the file where the model will be saved.
        """
        self.model.save(filepath)
        
    def load(self, filepath):
        """
        Load a pre-trained model from a file.

        Parameters:
        - filepath: Path to the file containing the pre-trained model.
        """
        self.model = Word2Vec.load(filepath)
    
    def most_similar(self, positive=None, negative=None, topn=10):
        """
        Find the top-N most similar words.

        Parameters:
        - positive: List of words that contribute positively.
        - negative: List of words that contribute negatively.
        - topn: Number of similar words to return.

        Returns:
        - List of (word, similarity) tuples.
        """
        return self.model.wv.most_similar(positive=positive, negative=negative, topn=topn)

    def get_vector(self, word):
        """
        Get the vector representation of a word.

        Parameters:
        - word: The word to get the vector for.

        Returns:
        - The vector representation of the word.
        """
        return self.model.wv[word]
    
    def get_vectors(self):
        """
        Get the vectors representation of all words.

        Returns:
        - The vectors representation of the words.
        """
        
        return self.model.wv.vectors
    
    def get_vocab(self):
        """
        Get the vocabulary of the model.

        Returns:
        - List of words in the vocabulary.
        """
        return list(self.model.wv.index_to_key)
    
    def sentence_vector(self, sentence):
        """
        Calculate the vector representation of a sentence by averaging the vectors of its words.

        Parameters:
        - sentence: List of words in the sentence.

        Returns:
        - The vector representation of the sentence.
        """
        word_vectors = [self.model.wv[word] for word in sentence if word in self.model.wv]
        if not word_vectors:
            # Handle the case when none of the words in the sentence is in the vocabulary
            return np.zeros(self.vector_size)
        return np.mean(word_vectors, axis=0)
    
    def index_to_key(self):
        return self.model.wv.index_to_key