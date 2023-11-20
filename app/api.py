import numpy as np
import logging
from flask import  Blueprint, request, render_template
import pickle

model = None  # Declare model as a global variable

api = Blueprint('api', __name__)

logger = logging.getLogger('werkzeug')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.FileHandler('test.log')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_model():
    global model
    if model is None:
        model = pickle.load(open('app/models/pkl/model.pkl', 'rb'))

@api.before_request
def before_request():
    load_model()

@api.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            logger.info('Data received: {}'.format(request.form.values()))

            string_features = [str(x) for x in request.form.values()]

            # Load the CountVectorizer along with the model
            cv = pickle.load(open('app/models/pkl/transform.pkl', 'rb'))
            input_text = cv.transform(string_features).toarray()

            # Make prediction
            prediction = model.predict(input_text)

            user_input = string_features[0]
            output = prediction[0]

            logger.info('Prediction: {}'.format(output))

            if output == 0:
                return render_template('result.html', user_input=user_input, prediction_text='This sentence is safe 😎', bg_color='#B8FFA6')
            elif output == 1:
                return render_template('result.html', user_input=user_input, prediction_text='This sentence contains profanity 🤬', bg_color='#FF8686')
            else:
                return render_template('error.html', error_message='An error occurred. Please try again.')

        except Exception as e:
            logger.error('An error occurred: {}'.format(str(e)))
            return render_template('error.html', error_message='An error occurred. Please try again.')