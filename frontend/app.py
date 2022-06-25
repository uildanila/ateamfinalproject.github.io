from flask import Blueprint, render_template

dashboardapp = Blueprint('dashboardapp', __name__)

@dashboardapp.route('/home')
def home():
    return render_template("index.html")

@dashboardapp.route('/rfm-segmentation')
def rfm():
    return render_template("rfm.html")

@dashboardapp.route('/product-recommendation')
def recsys():
    return render_template("recsys.html")

@dashboardapp.route('/visualizations')
def visualizations():
    return render_template("visualizations.html")

