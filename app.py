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
    data = logic.get(name)
    logic.postprocess(data, 'data/postdata.csv')
    return render_template('results.html', data=data)

@app.route('/live_search')
def live_search():
    data = logic.get_all_customers_list()
    return render_template('live_search.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)

