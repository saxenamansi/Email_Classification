import nltk
nltk.download('punkt')
import csv

#Gathering the Training Data (5 normal emails and 5 spam emails with labels 0 and 1, respectively)
emails = []
with open('emaildataset.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        emails.append((row[0].strip(), row[1].strip()))
        
#Tokenisation: split each document into words. Also, convert to lower case.
from nltk.tokenize import word_tokenize
tokens = []
for email in emails:
    word_list = word_tokenize(email[0])
    for word in word_list:
        word = word.lower()
        tokens.append(word)

#Remove punctuation
cleaned_tokens = [word for word in tokens if word.isalpha()]

#Remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
filtered_tokens = [w for w in cleaned_tokens if not w in stop_words]

#Apply stemming to convert filtered tokens that are not nouns to their base form. 
from nltk.stem.porter import PorterStemmer
nltk.download('averaged_perceptron_tagger')
stemmed_words = []
porter = PorterStemmer()
pos_tags = nltk.pos_tag(filtered_tokens)
for pos_tag in pos_tags:
    word = pos_tag[0]
    tag = pos_tag[1]
    if (tag!= 'NN' or tag!='NNS' or tag!= 'NNP' or tag!='NNPS'):
        stemmed_words.append(porter.stem(word))
    else:
        stemmed_words.append(word)

#Finding unique words
unique_words = set(stemmed_words)

#extracting features for an emil string
def extract_features(email_string):
    tokens = word_tokenize(email_string)
    tokens = [w.lower() for w in tokens]
    cleaned_tokens = [word for word in tokens if word.isalpha()]
    filtered_tokens = [w for w in cleaned_tokens if not w in stop_words]
    stemmed_words = []
    porter = PorterStemmer()
    pos_tags = nltk.pos_tag(filtered_tokens)
    for pos_tag in pos_tags:
        word = pos_tag[0]
        tag = pos_tag[1]
        if (tag!= 'NN' or tag!='NNS' or tag!= 'NNP' or tag!='NNPS'):
            stemmed_words.append(porter.stem(word))
        else:
            stemmed_words.append(word)
    features = {}
    for word in unique_words:
        features[word] = word in stemmed_words
    return features
    
#extracting feature vectors for each document with labels 
train_data = []
for email in emails:
    train_feature = extract_features(email[0])
    pair = (train_feature, email[1]) 
    train_data.append(pair)

#Applying the Naive Bayes Classifier to our trained data
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

#testing the model
def testing(email_string):
    test_features = extract_features(email_string)
    print("Words from test email that are unique words from trained data: ")
    for key in test_features:
        if test_features[key] == True:
            print(key, sep=" ") #printing those words that are in our set of unique words from trained data
    output = classifier.classify(test_features)
    print("Output of first email --> ", output, sep = " ") #predicting the output of our test string
    return output
    
#First testing example - it is not spam
test_email1 = "Heyy mansi its been so long, please contact me asap we need to catch upp!"
testing(test_email1)
print("Expected output --> negative\n\n")
   
#Second testing example - it is not spam     
test_email2 = "Mansi please register at PAT soon today it the last date, don't forget!"
testing(test_email2)
print("Expected output --> negative\n\n")

#Third testing example - it is spam     
test_email3 = "don't miss out on these myntra offers, check them out!"
testing(test_email3)
print("Expected output --> positive\n\n")
