# Run Specialty Recommendation Service

## New Features to Implement

### 1. Refine Likelihood to Purchase Function
Currently, the likelihood for a customer to purchase an item is generated using a random number between 50 and 100 in the `results_page.html` JavaScript. We need to build a more sophisticated function to accurately calculate and return the likelihood of a customer purchasing a product.

### 2. Refine Average Time to Purchase a Product
Improve the calculation of the average time it takes for a customer to purchase a product. Refine an algorithm to provide a more accurate and reliable average time to purchase metric.

### 3. Clean up logic modules
Break up 'logic.py' so that not all of the computation is done in that file. Make it make more sense by creating more files: `helpers.py`, `make_recommendations.py`, `customer_metrics.py`, `postprocessing.py`. This should be put inside a `/logic/` folder. 

### 4. Sort out deployment
Heroku computation for internal website tool.


## Usage
```bash
pip install -r requirements.txt
python app.py
```

## Contributors
@parkercarrus
@DivijSasidhar

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Code Flow

**app.py**  
Initializes local web page

**import logic**  
This custom module contains all of the thinking/machine learning/recommendation system logic. Basically, when you import logic, it computes the similarity matrix between all of the customers in the dataset. When you call logic.get() it calls predict() which takes the top 'n' most similar users and checks what products they've purchased most recently.

**logic.get**  
This returns a list of all customer data, which includes product recommendations.

**logic.postprocess**  
This appends relevant customer data to a local CSV file for later processing.  
(In deployment, there would be a separate function to go through and evaluate the accuracy of the model's predictions throughout the sales day.)

## Example Dictionary

`logic.get(name)` will return a dictionary with this format:

<pre style="font-size: 12px;">
{
  'customer_id': '31404eab-b254-42c3-a9d0-d262caa9053d',
  'name': 'Parker Carrus',
  'email': 'holson@example.org',
  'phone': '(947)498-5440x5837',
  'address': '6982 Hunter Dale Apt. 700\nEast Mariaburgh, NJ 81337',
  'loyalty_points': 499,
  'discount': 'Student-Athlete',
  'insole': False,
  'mailing_list': False,
  'gender': 'Male',
  'shoe_size': 10.5,
  'credit_card': 'Visa',
  'preferred_category': 'neutral',
  'returner': True,
  'avg_time_between_purchases': 244.5,
  'purchase_history': [
    ['Pegasus 36', '2021-09-16', 140, 'purchase'],
    ['Cloudflow 4', '2024-11-01', 160, 'purchase'],
    ['860 10', '2020-11-10', -160, 'return'],
    ['860 10', '2020-10-26', 160, 'purchase'],
    ['860 12', '2022-11-13', 160, 'purchase'],
    ['Structure 23', '2023-06-16', 160, 'purchase'],
    ['1080 12', '2022-08-22', 160, 'purchase']
  ],
  'recommendations': [
    ('860 10', 1.3551160203240848),
    ('1080 12', 0.6159133643628538),
    ('Cumulus 23', 0.5036489610292063),
    ...
    ('Clifton 8', -0.1090382895739079),
    ('Kayano 28', -0.15897206777334807),
    ('Kayano 29', -0.17691815889969578),
    ('Nimbus 22', -0.1798296917238732),
    ('Clifton 7', -0.20694285944526816)
  ]
}
</pre>
