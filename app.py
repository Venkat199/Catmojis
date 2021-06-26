from flask import Flask,render_template,request,redirect,url_for
from utils import sentences_to_indices, read_glove_vecs
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.keras.layers import Embedding
import h5py
import numpy as np
app = Flask(__name__)
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
model = tf.keras.models.load_model('emojify.h5')

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict/<result>')
def predict(result):
    out_image = result + '.gif'
    return render_template('result.html',user_image= out_image)


@app.route('/submit',methods=['POST','GET'])
def submit():
    total_score=0
    if request.method=='POST':
        val1=(request.form['val1'])
        input1 = np.array([val1])
        inp = sentences_to_indices(input1,word_to_index,10)
        pred = np.argmax(model.predict(inp))
        result_list = ['love','play','happy','sad','eat','disgust','anger']
        res = result_list[pred]

    return redirect(url_for('predict', result=res))

if __name__ == '__main__':
    app.run()
