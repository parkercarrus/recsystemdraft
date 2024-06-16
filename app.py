# app.py
from flask import Flask, request, render_template
import logic

app = Flask(__name__)

# custom Jinja2 filter to zip lists
@app.template_filter('zip')
def zip_lists(a, b):
    return zip(a, b)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_data', methods=['POST'])
def get_data_route():
    name = request.form['name']
    try:
        data = logic.get(name)
    except Exception as e:
        return render_template('error.html', data={'e': e})

    logic.postprocess(data, 'data/postdata.csv')

    return render_template('results.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)

