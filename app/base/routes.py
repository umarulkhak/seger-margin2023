from flask import Flask,render_template, redirect, request, url_for
from flask_login import (
    current_user,
    login_user,
    logout_user
)

import numpy as np # linear algebra
import pandas as pd # data processing
import os
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsRegressor as KNN_Reg
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import joblib

from app import db, login_manager
from app.base import blueprint
from app.base.forms import LoginForm, CreateAccountForm
from app.base.models import User


from app.base.util import verify_pass


@blueprint.route('/')
def route_default():
    return redirect(url_for('base_blueprint.login'))


@blueprint.route('/error-<error>')
def route_errors(error):
    return render_template('errors/{}.html'.format(error))

## Login & Registration


@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:

        # read form data
        username = request.form['username']
        password = request.form['password']

        # Locate user
        user = User.query.filter_by(username=username).first()

        # Check the password
        if user and verify_pass(password, user.password):

            login_user(user)
            return redirect(url_for('base_blueprint.route_default'))

        # Something (user or pass) is not ok
        return render_template('login/login.html', msg='Wrong user or password', form=login_form)

    if not current_user.is_authenticated:
        return render_template('login/login.html', form=login_form)
    return redirect(url_for('home_blueprint.index'))


@blueprint.route('/create_user', methods=['GET', 'POST'])
def create_user():
    login_form = LoginForm(request.form)
    create_account_form = CreateAccountForm(request.form)
    if 'register' in request.form:

        username = request.form['username']
        email = request.form['email']

        user = User.query.filter_by(username=username).first()
        if user:
            return render_template('login/register.html', msg='Username already registered', form=create_account_form)

        user = User.query.filter_by(email=email).first()
        if user:
            return render_template('login/register.html', msg='Email already registered', form=create_account_form)

        # else we can create the user
        user = User(**request.form)
        db.session.add(user)
        db.session.commit()

        return render_template('login/register.html', msg='User created please <a href="/login">login</a>', form=create_account_form)

    else:
        return render_template('login/register.html', form=create_account_form)
## Load Forder Image
app = Flask(__name__)
IMAGE_FOLDER = os.path.join('static', 'img_pool')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

## Proses rekomendasi
@blueprint.route('/rekomendasi', methods=['GET', 'POST'])
def rekomendasi():

    if request.method == 'POST':
        # Ambil value inputan
        area = float(request.form['area'])
        sc_price = float(request.form['sc_price'])
        qty_kirim = float(request.form['qty_kirim'])
        avg_hpp = float( request.form['avg_hpp'])
        avg_oa = float(request.form['avg_oa'])

        rekomendasi = np.array([[area,sc_price,qty_kirim,avg_hpp,avg_oa]])

        # Load the model from the file
        knn_from_margin = joblib.load('app/base/model_knn_margin.pkl')
        knn_from_asal = joblib.load('app/base/model_knn_asal.pkl')
        knn_from_transport = joblib.load('app/base/model_knn_transport.pkl')

        # Use the loaded model to rekomendasi
        margin = knn_from_margin.predict(rekomendasi)
        asal = knn_from_asal.predict(rekomendasi)
        transport = knn_from_transport.predict(rekomendasi)

        # Clean margin
        margin = float(margin)

        # Mengambil input dari pengguna untuk area dalam bentuk float
        area = float(area)

        # Mencocokkan nilai area dan mencetak string yang sesuai
        if area == 1.0:
            area = "JAWA TIMUR"
        elif area == 2.0:
            area = "JAWA TENGAH"
        elif area == 3.0:
            area = "JAKARTA"
        elif area == 4.0:
            area = "KALIMANTAN"
        else:
            area = "Nilai tidak valid"


        # Mengambil input dari pengguna untuk asal dalam bentuk float
        nilai = float(asal)

        # Mencocokkan nilai asal dan mencetak string yang sesuai
        if nilai >= 1.0 and nilai < 2.0:
            asal = "Calabai"
        elif nilai >= 2.0 and nilai < 3.0:
            asal = "Makassar"
        elif nilai >= 3.0 and nilai < 4.0:
            asal = "Kopo"
        elif nilai >= 4.0 and nilai < 5.0:
            asal = "Dompu"
        elif nilai >= 5.0 and nilai < 6.0:
            asal = "Pulubala"
        elif nilai >= 6.0 and nilai < 7.0:
            asal = "Palu"
        elif nilai >= 7.0 and nilai < 8.0:
            asal = "Paguat"
        elif nilai >= 8.0 and nilai < 9.0:
            asal = "Cepu"
        elif nilai >= 9.0 and nilai < 10.0:
            asal = "Sumengko"
        elif nilai >= 10.0 and nilai < 11.0:
            asal = "Kupang"
        elif nilai >= 11.0 and nilai < 12.0:
            asal = "Sumbawa"
        elif nilai >= 12.0 and nilai < 13.0:
            asal = "Ampana"
        elif nilai >= 13.0 and nilai < 14.0:
            asal = "Balung"
        elif nilai >= 14.0 and nilai < 15.0:
            asal = "Nganjuk"
        elif nilai >= 15.0 and nilai < 16.0:
            asal = "Tuban"
        elif nilai >= 16.0:
            asal = "Lampung"
        else:
            asal = "Nilai tidak valid"



        # Mengambil input dari pengguna untuk asal dalam bentuk float
        nilai01 = float(transport)

        # Inisialisasi variabel transport dengan nilai default
        transport = "Nilai tidak valid"

        # Mencocokkan nilai asal dan mengubah nilai variabel transport sesuai
        if nilai01 >= 5.0:
            transport = "Kapal"
        elif nilai01 >= 4.0:
            transport = "KM. NEW GLORY"
        elif nilai01 >= 3.0:
            transport = "GALATIA 05"
        elif nilai01 >= 2.0:
            transport = "Container"
        elif nilai01 >= 1.0:
            transport = "Trucking"


        if transport == 'Kapal':
            img_filename = os.path.join(
            app.config['UPLOAD_FOLDER'], 'kapal.png')
        elif transport == 'KM. NEW GLORY':
            img_filename = os.path.join(
            app.config['UPLOAD_FOLDER'], 'kapal.png')
        elif transport == 'GALATIA 05':
            img_filename = os.path.join(
            app.config['UPLOAD_FOLDER'], 'kapal.png')
        elif transport == 'Container':
            img_filename = os.path.join(
            app.config['UPLOAD_FOLDER'], 'container.png')
        elif transport == 'Trucking':
            img_filename = os.path.join(
            app.config['UPLOAD_FOLDER'], 'truk.png')
        else:
            img_filename = "NaN"

    return render_template('Home.html', area=area, sc_price=sc_price, qty_kirim=qty_kirim, avg_hpp=avg_hpp, avg_oa=avg_oa, margin=margin, asal=asal, transport=transport, image=img_filename)



@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('base_blueprint.login'))


@blueprint.route('/shutdown')
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'


# Errors


@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('errors/403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('errors/403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500
