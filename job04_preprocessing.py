import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv('./crawling_data/naver_news_title_20231012.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
# print(labeled_y[:3])
label = encoder.classes_
# print(label)

with open('./models/new_token.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_y = to_categorical(labeled_y)
# print(onehot_y)

okt = Okt()     # 형태소
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem = True )
# print(X[0])

stopwords = pd.read_csv('./stopwords.csv', index_col = 0)
for j in range(len(X)):     # 뉴스 타이플 갯수 만큼 for state
    words = []
    for i in range(len(X[j])):  # 뉴스 제목 길이 만큼 for state
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])

    X[j] = ' '.join(words)  # 한글자 이상이고, 불용어가 아닌 애들로 문장 됨
# print(X[0])

token = Tokenizer()                 # labeling  모든 형태소에 번호 부여
token.fit_on_texts(X)               # 토큰이 가진 형태소, 라벨 세트
tokened_x = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
print(tokened_x[0:3])
print(wordsize)

with open('./models/new_token.pickle', 'wb') as f:
    pickle.dump(token, f)

max = 0
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])
print(max)

x_pad = pad_sequences(tokened_x, max)
print(x_pad[:3])

X_train, X_test, Y_train, Y_test = train_test_split(
    x_pad, onehot_y, test_size=0.2
)
print(X_train.shape, Y_train.shape)
print(X_train.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./crawling_data/news_data_max_{}_wordsize_{}'.format(max, wordsize),xy)



