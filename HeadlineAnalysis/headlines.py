import nltk
import time
import requests
from statistics import mean
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import re


def gather_headlines():
    # Gather all headlines from CNN Business
    url = "https://www.cnn.com/business/economy"
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    selected_elements = soup.select('section[class^="layout__wrapper"]')
    headline_elements = selected_elements[0].select('span[class="container__headline-text"]')
    return list(set([element.text for element in headline_elements]))

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

        self.headlines = gather_headlines()
        print(f'Gathered {len(self.headlines)} headlines')
        with open('HeadlineAnalysis/headlines_text', 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        with open('HeadlineAnalysis/headlines_text', 'a') as f:
            for headline in self.headlines:
                for char in re.findall(r'[^a-zA-Z0-9,$?.-:\' ]', headline):
                    print(f'Found {char} in {headline} - Unicode: {ord(char)}')
                if headline not in lines:
                    f.write(f'{headline}\n')

    def skip_unwanted(self, scores_tag_tuple):
        word, tag = scores_tag_tuple
        if not word.isalpha() or word in self.unwanted:
            return False
        if tag.startswith("NN"):
            return False
        return True

    def extract_features(self, text):
        features = dict()
        compound_scores = list()
        positive_scores = list()

        for sentence in nltk.sent_tokenize(text):
            compound_scores.append(self.sia.polarity_scores(sentence)["compound"])
            positive_scores.append(self.sia.polarity_scores(sentence)["pos"])

        features["mean_compound"] = mean(compound_scores) + 1
        features["mean_positive"] = mean(positive_scores)

        return features
    
    def categorize_headlines_manual(self, headlines):
        for headline in headlines:
            print(f'\n{headline}\n{self.sia.polarity_scores(headline)}\n')
            choice = input('Is this headline good, bad, or neutral? (g, b, n)\n')
            while choice not in ['g', 'b', 'n']:
                choice = input('Invalid choice. Please enter g, b, or n.')
            if choice == 'g':
                tag = 'pos'
            elif choice == 'b':
                tag = 'neg'
            else:
                tag = 'neu'
            with open('HeadlineAnalysis/headlines_lexicon.txt', 'a') as f:
                score = self.sia.polarity_scores(headline)
                if f'{score}\t{tag}' not in lines:
                    f.write(f'{score}\t{tag}\n')
    

if __name__ == "__main__":
    analyzer = HeadlineAnalyzer()
    # analyzer.categorize_headlines_manual(gather_headlines())