import gzip
import gensim 
import logging
import pickle
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.DEBUG)

def tokenize_list(document):
	for line in document:
		yield gensim.utils.simple_preprocess (line)

def validate_type(type):
	if type == 'pos':
		return 1
	else:
		return 0

print("Loading CSV File.")

documents = pd.read_csv('imdb_master.csv',names=['id','type','review','label','file'],encoding = "ISO-8859-1")

reviews_vocab = []
training_vocab = []

label_list = documents.label.tolist()
type_list = documents.type.tolist()

for i, line in enumerate(documents.review.tolist()):
	if label_list[i] == 'unsup':
		reviews_vocab.append(line)
	else:
		training_vocab.append(line)

reviews_vocab = list(tokenize_list(reviews_vocab))
training_vocab = list(tokenize_list(training_vocab))

del label_list[0]
del type_list[0]

#review_train_size = int(len(review_list)/4)
review_train_size = int(len(documents.review.tolist())/4)
#review_test_size = int(review_train_size/2)
review_test_size = review_train_size
review_validation_size = int(review_train_size/2)
#review_base_size = (review_train_size + review_test_size + review_validation_size)
review_base_size = (review_train_size + review_test_size)


print ("Done reading data file")
N = 50

total_features = np.matrix([[]])
train_features = np.matrix([[]])
test_features = np.matrix([[]])
validation_features = np.matrix([[]])

del documents

train_features = [0] * review_train_size
test_features = [0] * review_test_size
total_features = [0] * review_base_size
validation_features = [0] * review_validation_size

for i in range(review_train_size):
	train_features[i] = [0] * (N + 1)

for i in range(review_test_size):
	test_features[i] = [0] * (N + 1)

for i in range(review_validation_size):
	validation_features[i] = [0] * (N + 1)

for i in range(review_base_size):
	total_features[i] = [0] * (N + 1)

print("Training Vocabulary from Reviews")
model = gensim.models.Word2Vec (reviews_vocab, size=N, window=5, min_count=2, workers=10)
model.train(reviews_vocab,total_examples=len(reviews_vocab),epochs=10)

wordcount = 0
for i in range(review_base_size):
	if i % 10000 == 0:
		logging.info ("Extracting Features from {0} reviews".format (i))
	wordcount = 0
	for word in training_vocab[i]:
		if word in model.wv.vocab:
			n_feature = 0
			wordcount+=1
			for feature in model[word]:
				total_features[i][n_feature] += feature
				n_feature+=1
	for j in range(N):
		total_features[i][j]/=wordcount
	total_features[i][N] = validate_type(label_list[i])

logging.info ("Creating Train, Test and Validation Bases")



for i in range(review_train_size):
		train_features[i] = total_features[i]
		test_features[i % (review_test_size)] = total_features[i+review_test_size]

np.random.shuffle(test_features)
test_features_final = np.matrix([[]])


test_features_final = [0] * review_validation_size

for i in range(review_validation_size):
	test_features_final[i] = [0] * (N + 1)

for i in range(int(review_test_size/2)):
		test_features_final[i] = test_features[i]
		validation_features[i % int(review_test_size/2)] = test_features[i+int(review_test_size/2)]


print ("Bases created. Saving files...")
np.save("reviewsTrainBase50.npy",train_features)
np.savetxt("reviewsTrainBase50.txt",train_features)
np.save("reviewsTestBase50.npy",test_features_final)
np.savetxt("reviewsTestBase50.txt",test_features_final)
np.save("reviewsValidationBase50.npy",validation_features)
np.savetxt("reviewsValidationBase50.txt",validation_features)
#np.save("reviewsValidationBase.npy",validation_features)
#np.savetxt("reviewsValidationBase.txt",validation_features)
