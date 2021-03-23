import csv

#TASK 1

#Gathering the Training Data (5 normal emails and 5 spam emails with labels 0 and 1, respectively)
emails = []
with open('emaildataset.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        emails.append((row[0].strip(), row[1].strip()))

#Functions defined for stop word removal, punctuation removal, stemming and lemmatization
def punctuation_removal(data_string):
    """ 
        This function takes a string as input and returns a string without punctuations 
        (mainly comma, fullstop and question mark).
        This is implemented by the string function: .replace() 
        More punctuations can be added to the list, or a regex of punctuations can be used.
    """
    punctuations = [",", ".", "?", "!", "'", "+", "(", ")"]
    for punc in punctuations:
        data_string = data_string.replace(punc, "")
    return data_string

def stopword_removal(tokens):
    """
        This function takes a list of words, or tokens as input and returns only those words that are not stopwords. 
        It removes the commonly used words in english that are used to make a sentence grammatically correct, 
        without adding much meaning to a sentence. 
        More stopwords can be added to the list. 
    """
   
    stopwords = ['of','on', 'i', 'am', 'this', 'is', 'a', 'was', 'it', 'the', 'do', 'you', 'by',
    'if','have', 'you', 'our', 'in', 'for', 'an', 'to', 'youve']
    filtered_tokens = []
    for token in tokens:
        if token not in stopwords:
            filtered_tokens.append(token)
    return filtered_tokens
    
def stemming(filtered_tokens):
    """
        This function takes a list of filtered tokens as input, and returns the base form or root word of the tokens. 
        This helps in normalising the words in our data/corpus.
    """
    root_to_token = {'you have':['youve'],
                    'select':['selected', 'selection'],
                    'it is':['its'],
                    'move':['moving'],
                    'photo':['photos'],
                    'success':['successfully', 'successful'],
                    'publish':['published'],
                    'achieve':['achieved'],
                    'focus':['focused'],
                    'prepare':['preparation'],
                    'wake':['woke'],
                    'regard':['regarding'],
                    'apply':['application'],
                    'reflect':['reflects'],
                    'demon':['demons'],
                    'revolution':['revolutions'],
                    'campaigns':['campaign'],
                    'book':['booking'],
                    'enroll':['enrolled'],
                    'receive':['received'],
                    'thank':['thanks'],
                    'love':['loved'],
                    'send':['sending'],
                    'note':['notes'],
                    'book':['books']
                    
    }
    
    base_form_tokens = []
    for token in filtered_tokens:
        for base_form, token_list in root_to_token.items():
            if token in token_list:
                base_form_tokens.append(base_form)
            else:
                base_form_tokens.append(token)
    return base_form_tokens

#Finding unique words
unique_words = []
tokens = []
for email in emails:
    email = email[0].lower().split()
    for word in email:
        clean_word = punctuation_removal(word)
        tokens.append(clean_word)
tokens = set(tokens)
filtered_tokens = stopword_removal(tokens)
base_form_tokens = stemming(filtered_tokens)
unique_words = set(base_form_tokens)

#TASK 2

#extracting feature vectors for each document with labels 
train_data = []
for email in emails:
    tokens = []
    word_list = email[0].lower().split()
    for word in word_list:
        clean_word = punctuation_removal(word)
        tokens.append(clean_word)
    filtered_tokens = stopword_removal(tokens)
    base_form_tokens = stemming(filtered_tokens)
    feature_vec = {}
    for word in unique_words:
        feature_vec[word] = word in base_form_tokens
    pair = (feature_vec, email[1]) 
    train_data.append(pair)

#Applying the Naive Bayes Classifier to our trained data
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

def testing(email_str):
    tokens = []
    word_list = email_str.lower().split()
    for word in word_list:
        clean_word = punctuation_removal(word)
        tokens.append(clean_word)
    filtered_tokens = stopword_removal(tokens)
    base_form_tokens = stemming(filtered_tokens)
    test_features = {}
    for word in unique_words:
        test_features[word] = word in base_form_tokens
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

#TASK 3
"""
JUSTIFICATION:
Here we see that the first two test samples are not spam emails and they have been rightly classified. 
However, the third is actually a spam mail and it has been wrongly classified as non spam.
This can be because of the small size of our training data, due to which the classifier is not completely
able to understand the difference in a spam and non spam email. This is evident when we print the words in the
test emails that are also prsent in our list of unique words from the training data. Here, we can clearly see 
that most of the words are new to the classifier. Infact, in the third test, the classifier ha seen none of words.
If it was aware that myntra is a brand, it will be able to classify it as spam.

So, it fails to understand the context and meaning of such words. Yet, we see two examples getting predicted correctly. 

So, Naive Bayes is a suitable classifier for our problem statement of classifying spam and non spam emails.
However, it cannot be very reliable if we do not provide it with sufficient training data. 
"""
