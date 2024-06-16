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
    data.sort()
    return render_template('live_search.html', data=data)

@app.route('/create_customer')
def create_customer():
    return render_template('create_customer.html')
    
@app.route('/rank_users', methods=['GET', 'POST'])
def rank_users():
    if request.method == 'POST':
        item_name = request.form['item_name']
        amount_of_users = int(request.form['amount_of_users'])
        return render_template('rank_users_results.html', item_name=item_name, amount_of_users=amount_of_users, data=[])
    return render_template('rank_users.html')

@app.route('/rank_users_results', methods=['POST'])
def rank_users_results():
    item_name = request.form.get('item_name', '')
    amount_of_users = int(request.form.get('amount_of_users', 0))
    gender = request.form.get('gender', 'any')
    size = request.form.get('size', 'any')
    discount = request.form.get('discount', 'any')

    # If 'slider' is selected, use the value from the slider
    if size == 'slider':
        size = request.form.get('slider_size', 'any')

    # Get initial unfiltered data
    data = logic.get_user_rankings(item_name, amount_of_users)
    print("Initial data:", data)  # Debug: Check initial data
    
    # Apply filters if not in baseline conditions
    if gender != 'any' or size != 'any' or discount != 'any':
        data = logic.filter(data, gender, size, discount)
        print("Filtered data:", data)  # Debug: Check filtered data

    return render_template('rank_users_results.html', item_name=item_name, amount_of_users=amount_of_users, data=data)

@app.route('/download_csv', methods=['POST'])
def download_csv():
    item_name = request.form.get('item_name', '')
    amount_of_users = int(request.form.get('amount_of_users', 0))
    gender = request.form.get('gender', 'any')
    size = request.form.get('size', 'any')
    discount = request.form.get('discount', 'any')

    slider_size = request.form.get('slider_size', 'any')

    # Handle the 'any' option for shoe size
    if size == 'slider':
        size = slider_size
    elif size == 'any' or not size:
        size = None

    data = logic.get_user_rankings(item_name, amount_of_users)
    if gender != 'any' or size is not None or discount != 'any':
        data = logic.filter(data, gender, size, discount)

    import io
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['User ID', 'Name', 'Phone', 'Email', 'Score'])
    cw.writerows(data)

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=best_users_for_{item_name}.csv"
    output.headers["Content-type"] = "text/csv"
    return output
if __name__ == '__main__':
    app.run(debug=True)

