from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("predict.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('predict.html',pred='Your Clutch is in Danger.\nProbability of clutch failure is {}'.format(output))
    else:
        return render_template('predict.html',pred='Your Clutch is safe.\nProbability of clutch failure is {}'.format(output))


if __name__ == '__main__':
    app.run()