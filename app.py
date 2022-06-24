from email import message
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from frontend.app import dashboardapp


app = Flask(__name__, static_folder='static', static_url_path='')

# frontend
app.register_blueprint(dashboardapp)


@app.route('/')
def index():
    return render_template("/index.html")

@app.route('/rfm-segmentation', methods = ["POST"])
def predict():
    df_rfm = pd.read_csv("df_rfm.csv")[['customer', 'inactive_days', 'number_of_orders', 'total_payment']]

    features = [x for x in request.form.values()]
    customer = [float(x) for x in [features[1], features[2], features[3]]]
    customer.append(features[0])

    test = pd.DataFrame(columns=['inactive_days','number_of_orders','total_payment', 'customer'])
    test.loc[0] = customer
    test = test[['customer', 'inactive_days', 'number_of_orders', 'total_payment']]
    df_rfm = df_rfm.append(test, ignore_index=True)

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
    message = 'Customer ' + segment['customer'].loc[0] + ' is a ' + segment['segment_index'].loc[0] + ' Customer'


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

    segment = segment.drop(columns=['index'], axis=1).rename(columns=dict_cust)
    return render_template("rfm.html", prediction_text = "{}".format(message), data=segment.to_html(index=False, classes='mytable'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8888)