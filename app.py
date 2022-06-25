from email import message
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import requests
from io import StringIO
from sklearn.cluster import KMeans
from frontend.app import dashboardapp


app = Flask(__name__, static_folder='static', static_url_path='')

# frontend
app.register_blueprint(dashboardapp)


@app.route('/')
def index():
    return render_template("/index.html")

@app.route('/predict', methods = ["POST"])
def predict():
    # df_rfm = pd.read_csv("df_rfm.csv")[['customer', 'inactive_days', 'number_of_orders', 'total_payment']]
    url='https://drive.google.com/file/d/1ia8mm7tTb8HJRr29DBY7hyx0DbzreqkD/view?usp=sharing'
    file_id = url.split('/')[-2]
    df_rfm='https://drive.google.com/uc?id=' + file_id
    url = requests.get(df_rfm).text
    df_rfm = StringIO(url)
    df_rfm = pd.read_csv(df_rfm)[['customer', 'inactive_days', 'number_of_orders', 'total_payment']]

    features = [x for x in request.form.values()]
    customer = [float(x) for x in [features[1], features[2], features[3]]]
    customer.append(features[0])

    test = pd.DataFrame(columns=['inactive_days','number_of_orders','total_payment', 'customer'])
    test.loc[0] = customer
    test = test[['customer', 'inactive_days', 'number_of_orders', 'total_payment']]
    df_rfm = pd.concat([df_rfm, test], ignore_index=True)

    list_segment = []

    def segmentation(data, x):
        for i in range(len(data[x])):
            if data[x][i]=='333' or data[x][i]=='332' or data[x][i]=='323' or data[x][i]=='313' or data[x][i]=='312' :
                list_segment.append('Loyal')
            elif data[x][i]=='232' or data[x][i]=='231' or data[x][i]=='222' or data[x][i]=='233' or data[x][i]=='223' or data[x][i]=='213':
                list_segment.append('Casual')
            elif data[x][i]=='331' or data[x][i]=='321' or data[x][i]=='322' or data[x][i]=='311':
                list_segment.append('Need Attention')
            else:
                list_segment.append('About to Sleep - Lost')
        return list_segment
        
    def RFMmodel(data):

        # Recency Clusters
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(data[['inactive_days']])
        data['cluster_r'] = km.labels_

        # Renaming the clusters according to mean number of inactive_days
        recency_cluster = data.groupby('cluster_r')['inactive_days'].mean().reset_index()
        recency_cluster = recency_cluster.sort_values(by='inactive_days', ascending=False).reset_index(drop=True)
        recency_cluster['index'] = np.arange(1,4)
        recency_cluster.set_index('cluster_r', inplace=True)
        cluster_dict = recency_cluster['index'].to_dict()
        data['cluster_r'].replace(cluster_dict, inplace=True)

        # Frequency Clusters
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(data[['number_of_orders']])
        data['cluster_f'] = km.labels_

        # Renaming the clusters according the mean number_of_orders
        frequency_cluster = data.groupby('cluster_f')['number_of_orders'].mean().reset_index()
        frequency_cluster = frequency_cluster.sort_values(by='number_of_orders').reset_index(drop=True)
        frequency_cluster['index'] = np.arange(1,4)
        frequency_cluster.set_index('cluster_f', inplace=True)
        cluster_dict = frequency_cluster['index'].to_dict()
        data['cluster_f'].replace(cluster_dict, inplace=True)

        # Monetary Clusters
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(data[['total_payment']])
        data['cluster_m'] = km.labels_

        # Renaming the clusters according to mean number of total_payment
        monetary_cluster = data.groupby('cluster_m')['total_payment'].mean().reset_index()
        monetary_cluster = monetary_cluster.sort_values(by='total_payment').reset_index(drop=True)
        monetary_cluster['index'] = np.arange(1,4)
        monetary_cluster.set_index('cluster_m', inplace=True)
        cluster_dict = monetary_cluster['index'].to_dict()
        data['cluster_m'].replace(cluster_dict, inplace=True)

        # Final Segmentation
        data['rfm_segment'] = data['cluster_r'].astype(str) + data['cluster_f'].astype(str) + data['cluster_m'].astype(str)
        data = data.reset_index()


        data['segment_index'] = segmentation(data, 'rfm_segment')
        return data

    df_rfm = RFMmodel(df_rfm)
    segment = df_rfm[df_rfm['customer'] == test['customer'].loc[0]].reset_index(drop=True)
    message = 'New data customer ' + segment['customer'].loc[0] + ' is a ' + segment['segment_index'].loc[0] + ' Customer'


    dict_cust = {
        'customer':'Customer',
        'inactive_days': 'Inactive Days',
        'number_of_orders': 'Number of Orders',
        'total_payment': 'Total Payment',
        'cluster_r':'Recency Cluster',
        'cluster_f': 'Frequency Cluster',
        'cluster_m': 'Monetary Cluster',
        'segment_index': 'Customer Segment'
    }

    segment = segment.drop(columns=['index', 'rfm_segment'], axis=1).rename(columns=dict_cust)
    return render_template("rfm.html", prediction_text = "{}".format(message), data=segment.to_html(index=False, classes='mytable'))

@app.route('/recsys', methods = ["POST"])
def recsys():
    # result = pd.read_csv("df_eval_pred_act.csv")
     # customer_c_history = pd.read_csv("customer_c.csv")
    # df_product_dict = pd.read_csv("list_product_details.csv")
    # df_cleanAll = pd.read_csv("clean_all.csv")
    # df_stateDict = pd.read_csv("state_dict.csv")

    url='https://drive.google.com/file/d/1ia8mm7tTb8HJRr29DBY7hyx0DbzreqkD/view?usp=sharing'
    file_id = url.split('/')[-2]
    result='https://drive.google.com/uc?id=' + file_id
    url = requests.get(result).text
    result = StringIO(url)
    result = pd.read_csv(result)

    index = result[result[['customer', 'product']].duplicated()].index
    result.drop(index=[i for i in index], axis=0, inplace=True)

    url='https://drive.google.com/file/d/1N5VVzM1nqY2Rmr8rwKEnxsW7qc9VWBz6/view?usp=sharing'
    file_id = url.split('/')[-2]
    customer_c_history='https://drive.google.com/uc?id=' + file_id
    url = requests.get(customer_c_history).text
    customer_c_history = StringIO(url)
    customer_c_history = pd.read_csv(customer_c_history)

    url='https://drive.google.com/file/d/1WEaex52Zz-EAQpHSGCih-imdwiDWNgUF/view?usp=sharing'
    file_id = url.split('/')[-2]
    df_product_dict='https://drive.google.com/uc?id=' + file_id
    url = requests.get(df_product_dict).text
    df_product_dict = StringIO(url)
    df_product_dict = pd.read_csv(df_product_dict)

    url='https://drive.google.com/file/d/1_0aettfQKH8bs4QQ71v7816S06hjUpBD/view?usp=sharing'
    file_id = url.split('/')[-2]
    df_cleanAll='https://drive.google.com/uc?id=' + file_id
    url = requests.get(df_cleanAll).text
    df_cleanAll = StringIO(url)
    df_cleanAll = pd.read_csv(df_cleanAll)

    url='https://drive.google.com/file/d/1gdJbFKvECmY9C001v6uuz1TMJYV1x1Lp/view?usp=sharing'
    file_id = url.split('/')[-2]
    df_stateDict='https://drive.google.com/uc?id=' + file_id
    url = requests.get(df_stateDict).text
    df_stateDict = StringIO(url)
    df_stateDict = pd.read_csv(df_stateDict)

    index = result[result[['customer', 'product']].duplicated()].index
    result.drop(index=[i for i in index], axis=0, inplace=True)

    customer = [str(i) for i in request.form.values()]
    
    customer = customer[0]

    if customer not in customer_c_history['customer'].unique():
        message = customer + ' not in list of segment C Customer'
        prevBuy = ' '
        recommendation = pd.DataFrame()
        condition = True
    else:
        hasil = result[result['customer'] ==  customer].sort_values('Predicted_Review', ascending=False).head(3)
        recommendation = hasil.merge(df_product_dict[['product','product_category_name_english','price', 'seller', 'review_score']], on=['product', 'product_category_name_english'], how='left')
        recommendation = recommendation.merge(df_cleanAll[['seller', 'seller_state']].drop_duplicates(), how='left', on='seller')
        recommendation = recommendation.merge(df_stateDict, how='left', left_on='seller_state', right_on='state_id')[['product', 'product_category_name_english', 'seller', 'state_name', 'price', 'review_score']]

        dict_recsys = {
            'product':'Product',
            'product_category_name_english': 'Category',
            'seller': 'Seller',
            'state_name': 'Seller State',
            'price':'Price',
            'review_score': 'Average Review Score',
        }

        recommendation = recommendation.rename(columns=dict_recsys)

        def categoryDict(data):
            category = []
            for i in data:
                category.append(i.replace('_', ' ').title())
            return category
        
        recommendation['Category'] =  categoryDict(recommendation['Category'] )
        
        def previousBuy(customer):
            previous_product = customer_c_history[customer_c_history['customer'] == customer]['product_category_name_english'].unique()
            list_category = [i for i in previous_product]
            list_category = categoryDict(list_category)
            n = 0
            prevCategory = ''
            for i in list_category:
                if n < len(list_category):
                    prevCategory += str(i)
                    if n != len(list_category) - 1:
                        prevCategory += ', '
                    n += 1
            return prevCategory
        
        prevBuy = str(previousBuy(customer))
        message = 'List of Top 3 Recommendation Product for customer ' + customer 
        condition=False

    return render_template("recsys.html", showMessage = "{}".format(message), condition=condition, resultPrev = "{}".format(prevBuy), cust=customer, data=recommendation.to_html(index=False, classes='mytable'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=1234)