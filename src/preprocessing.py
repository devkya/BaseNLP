# Tokenization
from abc import *
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter


class Tokenizer(ABC):
    @abstractmethod
    def ko_tokenize(self, text: str) -> list:
        pass

    def en_tokenize(self, text: str) -> list:
        pass

    def __call__(self, text: str) -> list:
        return self.en_tokenize(text)


class WordTokenizer(Tokenizer):
    def ko_tokenize(self, text: str) -> list:
        return text.split()

    def en_tokenize(self, text: str) -> list:
        return word_tokenize(text)


class SentenceTokenizer(Tokenizer):
    def ko_tokenize(self, text: str) -> list:
        import kss
        return kss.split_sentences(text)

    def en_tokenize(self, text: str) -> list:
        return sent_tokenize(text)


class MorphemeTokenizer(Tokenizer):
    def ko_tokenize(self, text: str) -> list:
        from konlpy.tag import Mecab
        mecab = Mecab()
        return mecab.morphs(text)


if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('punkt')
    ko_text = "위의 예제는 Okt 형태소 분석기로 토큰화를 시도해본 예제입니다. 각각의 메소드는 아래와 같은 기능을 갖고 있습니다."
    en_text = "The above example is an example of tokenization attempted with the Okt morpheme analyzer. Each method has the following functions."

    sentence_tokenizer = SentenceTokenizer()
    token = sentence_tokenizer(en_text)

    word_tokenizer = WordTokenizer()
    token = word_tokenizer(en_text)
    print(token)
    vocab = Counter(token)
    print(vocab)
