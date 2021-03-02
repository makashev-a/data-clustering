import re
import string
from urllib.request import urlopen

import matplotlib.pyplot as plt
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud

# url = "https://zakon.kz/"
url = "https://tengrinews.kz/"
html = urlopen(url).read()
soup = BeautifulSoup(html, features="html.parser")

for script in soup(["script", "style"]):
    script.extract()

text = soup.get_text()
lines = (line.strip() for line in text.splitlines())
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
text = '\n'.join(chunk for chunk in chunks if chunk)

text = text.lower()
spec_chars = string.punctuation + '\n\xa0«»\t—…'
text = "".join([ch for ch in text if ch not in spec_chars])
text = re.sub('\n', '', text)


def remove_chars_from_text(text, chars):
    return "".join([ch for ch in text if ch not in chars])


text = remove_chars_from_text(text, spec_chars)
text = remove_chars_from_text(text, string.digits)

text_tokens = word_tokenize(text)

russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['тыс'])
text_tokens = [token.strip() for token in text_tokens if token not in russian_stopwords]
text = nltk.Text(text_tokens)
fdist_sw = FreqDist(text)
print(fdist_sw.most_common(10))

text_raw = " ".join(text)

wordcloud = WordCloud(width=1920, height=1080).generate(text_raw)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordscloud.pdf', dpi=300)
plt.show()
