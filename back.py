from flask import Flask, render_template, request
import pickle
import numpy as np
from pyvi import ViTokenizer
from preprocess import normalize_text
#print(ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội"))
naive_bayes = pickle.load(open('naive_bayes_sa_new.pkl','rb'))
svm = pickle.load(open('svm_model.plk','rb'))
decision_tree = pickle.load(open('decision_tree_sa.pkl','rb'))
sgd = pickle.load(open('sgd_sa_new.pkl','rb'))
def find_most_probable_word(pipeline,sentence):
  res = []
  idx = np.argmax(pipeline.predict_proba([sentence]))
  
  for word in sentence.split(' '):
    #print(clf.predict_proba(cv.transform([word])))
    word = ' '.join(word.split('_'))
    res.append(pipeline.predict_proba([word])[0][idx])
  print(res)
  index = np.argwhere(np.array(res)>0.7)
  return index
app = Flask(__name__)
app.debug=True
@app.route('/hel')
def hello_world():
   return 'Hello World'
@app.route("/")
def index():
    result = ''
    sentence=''
    return render_template("main.html",result=result,sentence=sentence,len=len(sentence.split()),index=0)
@app.route('/',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      res = request.form
      
      sentence=res['comment']
      method = res['methods']
      print(method)
      sentence= ViTokenizer.tokenize(sentence)
      if method == 'SVM':
        pipeline= svm
        sentence=normalize_text(sentence)
        sentence= ViTokenizer.tokenize(sentence)
        result = pipeline.predict([normalize_text(sentence)])
        
        print(result)
        sentiment = {"NEU":"neutral","POS":"positive","NEG":"negative"}
        senti = sentiment[result[0]].upper()
        special_index = find_most_probable_word(pipeline,sentence)
      elif method == 'SGD':
        pipeline= sgd
        sentence=normalize_text(sentence)
        sentence= ViTokenizer.tokenize(sentence)
        result = pipeline.predict([normalize_text(sentence)])
        
        print(result)
        sentiment = ['neutral','positive','negative']
        senti = sentiment[result[0]].upper()
        special_index = find_most_probable_word(pipeline,sentence)
      elif method == 'Decision Tree':
        pipeline = decision_tree
        sentence=normalize_text(sentence)
        sentence= ViTokenizer.tokenize(sentence)
        result = pipeline.predict([sentence])
        special_index = find_most_probable_word(pipeline,sentence)
        print(sentence)
        print(result)
        sentiment= ['neutral','positive','negative']
        senti = sentiment[int(result)].upper()
      else:
        pipeline = naive_bayes
        sentence=normalize_text(sentence)
        sentence= ViTokenizer.tokenize(sentence)
        result = pipeline.predict([sentence])
        special_index = find_most_probable_word(pipeline,sentence)
        print(sentence)
        print(result)
        sentiment= ['neutral','positive','negative']
        senti = sentiment[int(result)].upper()
        
      return render_template("main.html", result='this comment is '+senti, sentence=sentence, index=special_index,len=len(sentence.split()))
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

