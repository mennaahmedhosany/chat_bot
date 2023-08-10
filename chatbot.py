
import random
import json
import pickle
import numpy as np
#the natural language tool kit
import  nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words=[]
classes =[]
documents=[]
ignore_letters=['?','!','.',',']

for intent in intents['intents'] :
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        #take the item and append to the list
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes :
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize( word) for word in words if word not in ignore_letters]

words =sorted(set(words))
classes=sorted(set(words))


pickle.dump(words,open('words.pkl' ,'wb'))
pickle.dump(classes,open('classes.pkl','wb'))


#prepare the data to feed to neuaralnetwork

dataset = []
#output_empty work ast he number of zeros will be the same as of the element in classes

template = [0]*len(classes)

for document in documents:
	bag = []
	word_patterns = document[0]
	word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

	for word in words:
		bag.append(1) if word in word_patterns else bag.append(0)

	output_row = list(template)
	output_row[classes.index(document[1])] = 1
	dataset.append([bag, output_row])

random.shuffle(dataset)
dataset = np.array(dataset)

train_x = list(dataset[:, 0])
train_y = list(dataset[:, 1])




