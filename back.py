from flask import Flask, render_template, request
import pickle
import numpy as np
from pyvi import ViTokenizer

#print(ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội"))
pipeline = pickle.load(open('naive_bayes.pkl','rb'))
def find_most_probable_word(sentence):
  res = []
  idx = np.argmax(pipeline.predict_proba([sentence]))
  sentence= ViTokenizer.tokenize(sentence)
  for word in sentence.split(' '):
    #print(clf.predict_proba(cv.transform([word])))
    word = ' '.join(word.split('_'))
    res.append(pipeline.predict_proba([word])[0][idx])
  print(res)
  index = np.argmax(np.array(res))
  return sentence, index
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
      result = str(pipeline.predict([sentence]))
      sentence,special_index = find_most_probable_word(sentence)
      print(sentence.split()[0].split('_'))
      sentiment= ['neutral','positive','negative']
      print(result[1:-1])
      return render_template("main.html", result='this comment is '+sentiment[int(result[1:-1])].upper(), sentence=sentence, index=special_index,len=len(sentence.split()))
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

