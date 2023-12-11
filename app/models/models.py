import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

profanity = pd.read_csv('app/models/clean_data.csv')

profanity.head(10)

profanity.isnull().sum()

profanity.dropna(subset=['text'], inplace = True)

profanity.isnull().sum()

profanity["is_offensive"].value_counts()

x = np.array(profanity["text"])
y = np.array(profanity["is_offensive"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size = 0.3,
                                                   random_state = 42)

model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)



import pickle
pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict(["i love you"]))

