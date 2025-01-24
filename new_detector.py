import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report
import joblib
data=pd.read_csv("C:/Users/HP/Downloads/Compressed/fake_and_real_news.csv")
vector=TfidfVectorizer()
data['Text']=data['Text'].str.lower()
data['Text']=data['Text'].apply(lambda x:re.sub(r'[^a-zA-Z0-9\s]','',x))
change=vector.fit_transform(data['Text'])
X=change
y=data['label']
model=MultinomialNB()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(X_train,y_train)
predict=model.predict(X_test)
accuracy=accuracy_score(y_test,predict)
classification=classification_report(y_test,predict)
save=joblib.dump(model,'Email_detection.joblib')
save_vector=joblib.dump(vector,"Email_detecion_vector.joblib")

