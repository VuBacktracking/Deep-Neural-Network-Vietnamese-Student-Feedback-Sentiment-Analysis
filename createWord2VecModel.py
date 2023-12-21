from utils.preprocessing import readData, tokenizeWords
from word2vec.Word2Vec import studentFeedbackWord2Vec

def createW2VModel(models = ["skipgram"]):
    models = models
    sentences = readData("_UIT-VSFC/Corpus.txt")
    tokenizedWords = tokenizeWords(sentences)
    
    # Generate Word2Vec Model
    for model in models:
        w2v_model = studentFeedbackWord2Vec(tokenizedWords, model_type=model)
        w2v_model.save(f'word2vec/{model}_model.bin')
    
    print("Succesfully")

if __name__ == "__main__":
    createW2VModel(models = ["skipgram", "cbow"])