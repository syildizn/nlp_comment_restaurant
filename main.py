import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB # Naive Bayes classifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC  # SVM classifier
from sklearn.ensemble import RandomForestClassifier # Random Forest classifier
from sklearn.linear_model import LogisticRegression # Logistic Regression classifier
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boosting classifier



# Veri setini yükleme
#data_path = '/Users/syildizn/Desktop/nlp_restaurant/Restaurant_Reviews.tsv'
data_path = 'C:\GitHub\denemetemplate\enlp_comment_restaurant\Restaurant_Reviews.tsv'
reviews_df = pd.read_csv(data_path, delimiter='\t', quoting=3)  # TSV dosyası olduğu için delimiter olarak '\t' kullanılıyor.

# İlk beş satırı gösterme
reviews_df.head()
print(reviews_df.head())
print("---------------------------------")

def preprocess_text(text):
    # Noktalama işaretlerini kaldırma ve küçük harfe dönüştürme
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)

    # Durak sözcükleri çıkarma
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    # Kök bulma (stemming)
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(word) for word in text.split())

    return text



# Ön işleme fonksiyonunu tüm yorumlara uygulama
reviews_df['Processed_Review'] = reviews_df['Review'].apply(preprocess_text)

# İşlenmiş verinin ilk beş satırını gösterme
print(reviews_df.head())

# Özellik çıkarımı
tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(reviews_df['Processed_Review']).toarray()
y = reviews_df['Liked'].values

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitme naivebayes
model = MultinomialNB()
model.fit(X_train, y_train)
# Model oluşturma ve eğitme SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
# Model oluşturma ve eğitme Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Model oluşturma ve eğitme Logistic Regression
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
# Model oluşturma ve eğitme Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Modeli test etme ve performansı değerlendirme naivebayes
predictions = model.predict(X_test)
print("-----------------Naive Bayes------------")
print(classification_report(y_test, predictions))

# Modeli test etme ve performansı değerlendirme SVM
svm_predictions = svm_model.predict(X_test)
print("----------------- SVM ------------")
print(classification_report(y_test, svm_predictions))

# Modeli test etme ve performansı değerlendirme RandomForest
rf_predictions = rf_model.predict(X_test)
print("----------------- Random Forest ------------")
print(classification_report(y_test, rf_predictions))

# Modeli test etme ve performansı değerlendirme Logistic Regression
logreg_predictions = logreg_model.predict(X_test)
print("----------------- Logistic Regression ------------")
print(classification_report(y_test, logreg_predictions))

# Modeli test etme ve performansı değerlendirme Gradient Boosting
gb_predictions = gb_model.predict(X_test)
print("----------------- Gradient Boosting ------------")
print(classification_report(y_test, gb_predictions))

print("---------------------------------------")

def custom_input_prediction(input_text):
    # Girdiyi işleme
    processed_input = preprocess_text(input_text)

    # Girdiyi TF-IDF vektörüne dönüştürme
    tfidf_input = tfidf.transform([processed_input]).toarray()

    # Model ile tahmin yapma Naive Bayes
    prediction_nb = model.predict(tfidf_input)
    # return "Olumlu" if prediction[0] == 1 else "Olumsuz"
    # Model ile tahmin yapma SVM
    prediction_svm = svm_model.predict(tfidf_input)
    # return "Olumlu" if predictionSvm[0] == 1 else "Olumsuz"
    # Model ile tahmin yapma Random Forest
    prediction_rf = rf_model.predict(tfidf_input)
    # Model ile tahmin yapma Logistic Regression
    prediction_logreg = logreg_model.predict(tfidf_input)
    # Model ile tahmin yapma Gradient Boosting
    prediction_gb = gb_model.predict(tfidf_input)

    return {
        "Naive Bayes": "Olumlu" if prediction_nb[0] == 1 else "Olumsuz",
        "SVM": "Olumlu" if prediction_svm[0] == 1 else "Olumsuz",
        "Random Forest": "Olumlu" if prediction_rf[0] == 1 else "Olumsuz",
        "Logistic Regression": "Olumlu" if prediction_logreg[0] == 1 else "Olumsuz",
        "Gradient Boosting": "Olumlu" if prediction_gb[0] == 1 else "Olumsuz"
    }

# Kullanıcıdan girdi alınması ve tahmin yapılması
for i in range(3):
    custom_review = input("Bir yorum girin: ")
    """
    nb_prediction, svm_prediction = custom_input_prediction(custom_review)
    #print("Tahmin: ", custom_input_prediction(custom_review))
    print("Naive Bayes Tahmin: ", nb_prediction)
    print("SVM Tahmin: ", svm_prediction)
    """
    predictions = custom_input_prediction(custom_review)
    for model_name, prediction in predictions.items():
        print(f"{model_name} Tahmin: {prediction}")