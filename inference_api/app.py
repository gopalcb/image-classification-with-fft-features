from flask import Flask, jsonify, request, render_template
from inference import *

app = Flask(__name__) 

@app.route('/', methods = ['GET', 'POST']) 
def home():
    return render_template('index.html')
  

@app.route('/make_inference/<file_name>', methods = ['POST']) 
def make_inference(file_name):
    path = f'data/{file_name}'
    result = extract_results_from_prediction(path)
    return {'res': result}

  
  
# driver function 
if __name__ == '__main__': 
    app.run(debug = True) 