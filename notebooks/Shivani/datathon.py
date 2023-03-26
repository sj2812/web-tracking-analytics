from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

def get_content_page(url):

    
    driver = webdriver.Chrome()
    driver.delete_all_cookies()
    
    driver.get(url)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words_german = stopwords.words('german')+stopwords.words('english')+['2022','02','open','2023','2021']
# input paragraph

for url in ['https://www.wirfuervielfalt.de/','https://www.wirfuervielfalt.de/de/angebote/werde-volunteer-special-olympics-world-games-berlin-2023',
            'https://www.wirfuervielfalt.de/vinedig','https://www.wirfuervielfalt.de/de/angebote/stattwerkstatt/']:
    paragraph = get_content_page(url)
    # tokenize the paragraph
    vectorizer = TfidfVectorizer(stop_words=stop_words_german)
    X = vectorizer.fit_transform([paragraph])

    # get the top 5 most important keywords
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()[0]
    keywords = [feature_names[i] for i in tfidf_scores.argsort()[-50:][::-1]]

    print(keywords)


#print(get_content_page('https://www.wirfuervielfalt.de/'))