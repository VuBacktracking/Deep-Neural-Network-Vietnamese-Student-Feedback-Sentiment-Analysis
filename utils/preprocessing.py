from underthesea import sent_tokenize, word_tokenize
import re
import string

# Get stopwords
with open("vietnamese_stopwords/vietnamese-stopwords.txt", "r", encoding="utf-8") as file:
    stopwords = file.read().split("\n")
stopwords = set(stopwords)

# def preprocessing(text, redundantSet):
#     text = text.lower() 
#     words = word_tokenize(text)
#     for i in range(0,len(words)):
#         if words[i].count('_') == 0 and (words[i] in redundantSet or words[i].isdigit()):
#             words[i] = ''
#         else:
#             sub_words = words[i].split('_')
#             if any(w in redundantSet or w.isdigit() for w in sub_words):
#                 words[i] = ''
#     words = [w for w in words if w != '']
#     words = ' '.join(words)
#     return words

def remove_punctuation(input_string):
    # Create a translation table that maps each punctuation character to None
    translation_table = str.maketrans("", "", string.punctuation)

    # Use translate to remove punctuation
    result_string = input_string.translate(translation_table)

    return result_string

def remove_digit(input_string):
    result_string = re.sub(r"\d+", "",input_string)
    return result_string

def readData(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    sentences = sent_tokenize(text)
    return sentences

def tokenizeWords(sentences):
    words = [word_tokenize(remove_punctuation(sentence)) for sentence in sentences]
    return words