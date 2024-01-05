from random import shuffle
import time
import re
import requests
from bs4 import BeautifulSoup
from statistics import mean
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Gather all headlines from CNN Business
url = "https://www.cnn.com/business/economy"
response = requests.get(url)
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')
selected_elements = soup.select('span[class="container__headline-text"]')
headlines = [element.text for element in selected_elements]
print(f'Headlines gathered: {len(headlines)}')


# Sentiment Analyzer Class
class HeadlineAnalyzer:
    def __init__(self):
        # VADER Analyzer
        print('Initializing Sentiment Analyzer...')
        start_time = time.time()
        self.sia = SentimentIntensityAnalyzer()
        print(f'{time.time() - start_time:.2f}s\n')

        # Filter words
        print('Creating filter list...')
        start_time = time.time()
        self.unwanted = nltk.corpus.stopwords.words("english")
        self.unwanted.extend([word.lower() for word in nltk.corpus.names.words()])
        print(f'{time.time() - start_time:.2f}s\n')

        # Positive and negative words (negative only needed to filter shared words)
        print('Creating positive word list...')
        start_time = time.time()
        positive_words = [word for word in open('positive_words.txt', 'r').read().split('\n') if re.match("^[A-Za-z]+$", word)]
        print(f'{time.time() - start_time:.2f}s\n')

        print('Creating negative word list...')
        start_time = time.time()

        negative_words = [word for word in open('negative_words.txt', 'r').read().split('\n') if re.match("^[A-Za-z]+$", word)]
        print(f'{time.time() - start_time:.2f}s\n')
        
        # FDs and Top 100 positive words
        print('Creating frequency distributions...')
        start_time = time.time()
        positive_fd = nltk.FreqDist(positive_words)
        negative_fd = nltk.FreqDist(negative_words)
        common_set = set(positive_fd).intersection(negative_fd)
        for word in common_set:
            del positive_fd[word]
            del negative_fd[word]
        self.top_100_positive = [word for word, _ in positive_fd.most_common(100)]
        print(f'{time.time() - start_time:.2f}s\n')

        # Build features list
        print('Building features list...')
        start_time = time.time()
        self.features = [
            (self.extract_features(nltk.corpus.movie_reviews.raw(review)), "pos") for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])
        ]
        print('positive features done...')
        self.features.extend([
            (self.extract_features(nltk.corpus.movie_reviews.raw(review)), "neg") for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])
        ])
        print('negative features done...')
        self.features.extend([
            (self.extract_features(nltk.corpus.pros_cons.raw(pro)), "pos") for pro in nltk.corpus.pros_cons.fileids(categories=["pros"])
        ])
        print('pros features done...')
        self.features.extend([
            (self.extract_features(nltk.corpus.pros_cons.raw(con)), "neg") for con in nltk.corpus.pros_cons.fileids(categories=["cons"])
        ])
        print('cons features done...')
        print(f'{time.time() - start_time:.2f}s\n')

        # Train the model with the features list
        print('Training model...')
        start_time = time.time()
        shuffle(self.features)
        print(self.features[:5])
        train_count = len(self.features)
        self.classifier = nltk.classify.SklearnClassifier(MLPClassifier(max_iter=1000))
        self.classifier.train(self.features[:train_count])
        print(f'{time.time() - start_time:.2f}s\n')
        # accuracy = nltk.classify.accuracy(classifier, self.features[train_count:])
        # print(f'Accuracy: {accuracy:.2%}')

    def skip_unwanted(self, pos_tuple):
        word, tag = pos_tuple
        if not word.isalpha() or word in self.unwanted:
            return False
        if tag.startswith("NN"):
            return False
        return True

    def extract_features(self, text):
        features = dict()
        wordcount = 0
        compound_scores = list()
        positive_scores = list()

        for sentence in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sentence):
                if word.lower() in self.top_100_positive:
                    wordcount += 1
            compound_scores.append(self.sia.polarity_scores(sentence)["compound"])
            positive_scores.append(self.sia.polarity_scores(sentence)["pos"])

        features["mean_compound"] = mean(compound_scores) + 1
        features["mean_positive"] = mean(positive_scores)
        features["wordcount"] = wordcount

        return features
    
    def analyze(self, headline):
        features = self.extract_features(headline)
        return self.classifier.classify(features)
    

if __name__ == "__main__":
    analyzer = HeadlineAnalyzer()
    print(analyzer.analyze("This is a good headline"))
    print(analyzer.analyze("This is a bad headline"))
    print('\n')
    for headline in headlines:
        print(f'{headline} - {analyzer.analyze(headline)}')