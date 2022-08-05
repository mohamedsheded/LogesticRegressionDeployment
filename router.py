# Import the Libraries
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request
import joblib
import os
# the function I craeted to process the data in utils.py
from utils_regression_housing import preprocess_new


# Intialize the Flask APP
app = Flask(__name__)

# Loading the Model
model = joblib.load('model_LR.pkl')

# Route for Home page


@app.route('/')
def home():
    return render_template('index.html')

# Route for Predict page
'''
'LoanCurrentDaysDelinquent'

LP_CustomerPrincipalPayments

LP_GrossPrincipalLoss

LP_CustomerPayments (No)

LP_InterestandFees

LP_ServiceFees

MonthlyLoanPayment

AvailableBankcardCredit

RevolvingCreditBalance
'''
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # while prediction
        LoanCurrentDaysDelinquent = float(request.form['LoanCurrentDaysDelinquent'])
        LP_CustomerPrincipalPayments = float(request.form['LP_CustomerPrincipalPayments'])
        LP_GrossPrincipalLoss = float(request.form['LP_GrossPrincipalLoss'])
        LP_CustomerPayments = float(request.form['LP_CustomerPayments'])
        LP_InterestandFees = float(request.form['LP_InterestandFees'])
        LP_ServiceFees = float(request.form['LP_ServiceFees'])
        MonthlyLoanPayment = float(request.form['MonthlyLoanPayment'])
        AvailableBankcardCredit = float(request.form['AvailableBankcardCredit'])
        RevolvingCreditBalance = request.form['RevolvingCreditBalance']

        # Concatenate all Inputs
        X_new = pd.DataFrame({'LoanCurrentDaysDelinquent': [LoanCurrentDaysDelinquent], 'LP_CustomerPrincipalPayments': [LP_CustomerPrincipalPayments], 'LP_GrossPrincipalLoss': [LP_GrossPrincipalLoss], 'LP_CustomerPayments': [LP_CustomerPayments],
                              'LP_InterestandFees': [LP_InterestandFees], 'LP_ServiceFees': [LP_ServiceFees], 'MonthlyLoanPayment': [MonthlyLoanPayment], 'AvailableBankcardCredit': [AvailableBankcardCredit],
                              'RevolvingCreditBalance': [RevolvingCreditBalance]
                              })

        # Call the Function and Preprocess the New Instances
        X_processed = preprocess_new(X_new)

        # call the Model and predict
        y_pred_new = model.predict(X_processed)
        y_pred_new = '{:.4f}'.format(y_pred_new[0])

        return render_template('predict.html', pred_val=y_pred_new)
    else:
        return render_template('predict.html')


# Route for About page
@app.route('/about')
def about():
    return render_template('about.html')


# Run the App from the Terminal
if __name__ == '__main__':
    app.run(debug=True)