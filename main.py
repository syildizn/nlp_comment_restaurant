
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# Veri setini yükleme
data_path = '/Users/syildizn/Desktop/nlp_restaurant/Restaurant_Reviews.tsv'
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

# Model oluşturma ve eğitme
model = MultinomialNB()
model.fit(X_train, y_train)

# Modeli test etme ve performansı değerlendirme
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
