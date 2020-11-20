from flask import Flask, render_template, request
import pickle
import numpy as np
from preprocess import normalize_text
from utils import find_most_probable_word, NormalizeText

###========###
###LOAD MODEL###

naive_bayes = pickle.load(open('tfidf_NB.pkl','rb'))
svm = pickle.load(open('tfidf_svm.pkl','rb'))
decision_tree = pickle.load(open('countvectorizer_dectree.pkl','rb'))
sgd = pickle.load(open('tfidf_sgd.pkl','rb'))
#cnn = keras.load_model('fasttext_cnn_797.h5')
###DECLARE APP####
app = Flask(__name__)
app.debug=True

###ROUTING###
#index route
@app.route("/")
def index():
    result = ''
    sentence=''
    return render_template("main.html",method='SVM',result=result,sentence=sentence,len=len(sentence.split()),index=0)
@app.route('/',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      res = request.form
      
      sentence=res['comment']
      method = res['methods']
      print(method)
      if method == 'SVM':
        pipeline= svm
        result = pipeline.predict([sentence])
        
        print(result)
        sentiment = ['neutral','postive','negative']
        senti = sentiment[result[0]].upper()
        special_index = find_most_probable_word(pipeline,sentence)
      elif method == 'SGD':
        pipeline= sgd
        result = pipeline.predict([sentence])
        
        sentiment = {"NEU":"neutral","POS":"positive","NEG":"negative"}
        senti = sentiment[result[0]].upper()
        special_index = find_most_probable_word(pipeline,sentence)
      elif method == 'Decision Tree':
        pipeline = decision_tree
        result = pipeline.predict([sentence])
        special_index = find_most_probable_word(pipeline,sentence)
        sentiment = {"NEU":"neutral","POS":"positive","NEG":"negative"}
        senti = sentiment[result[0]].upper()
        special_index = find_most_probable_word(pipeline,sentence)
      else:
        pipeline = naive_bayes
        result = pipeline.predict([sentence])
        special_index = find_most_probable_word(pipeline,sentence)
        sentiment = {"NEU":"neutral","POS":"positive","NEG":"negative"}
        senti = sentiment[result[0]].upper()
        special_index = find_most_probable_word(pipeline,sentence)
        
      return render_template("main.html",method=method, result='this comment is '+senti, sentence=sentence, index=special_index,len=len(sentence.split()))
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

