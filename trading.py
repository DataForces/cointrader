#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 01:19:30 2018

@author: cc
"""
import numpy as np
import pandas as pd
import time
import schedule
import os
import glob
import json
import python_bitbankcc as bitbankcc
import argparse
from loging import access_info
from sklearn.svm import SVC
from sklearn.externals import joblib


def checkout():
    # check condition of current account
    # cancel all active orders in trading pair
    # return buying power
    active_orders = refApi.get_active_orders(pair)['orders']
    if active_orders:
        # cancel active orders
        active_orders = pd.DataFrame(active_orders)
        for idx, active_order in active_orders.iterrows():
            tradeApi.cancel_order(pair, str(active_order["order_id"]))

    assets = refApi.get_asset()["assets"]
    assets = pd.DataFrame(assets)
    assets[['free_amount', 'locked_amount','onhand_amount']] = assets[['free_amount', 'locked_amount','onhand_amount']].astype(np.float32)
    coin_name, currency = pair.split("_")
    cash = assets[assets['asset']==currency]['free_amount'].values[0]
    coins = assets[assets['asset']==coin_name]['free_amount'].values[0]
    print("[info] Cash: {:.2f} jpy; Coins: {:.2f} .".format(cash, coins))
    return cash, coins


def make_decison(infos):
    # make decison on 30 min trading records
    # decison ->[trade_or_not, pair, volumn]
    orders = []
    if os.path.exists("./models/trend_{}_clf.pkl".format(pair)):
        clf = joblib.load("./models/trend_{}_clf.pkl".format(pair))
        with open('./models/{}_quality.json'.format(pair), 'r') as fp:
            quality = json.load(fp)
    else:
        clf, quality = train_model()

    if quality['training_acc'] < 0.8:
        print("[info] Skip making decison due to poor model accuracy.")
        train_model()
        return orders
    else:
        record = infos[price_related].values
        record = record - record[0, :]
        record = record.reshape(1, -1)[:, 5:]
        predit = clf.predict(record)
        pred_proba = clf.predict_proba(record)
        trend = trends[np.argmax(pred_proba)]
        confidence = np.max(pred_proba)
        print("[info] Prediction of is {}, confidence is {:.02f}".
              format(trend, confidence))
        if trend == "asce" and confidence > 0.5:
            # generate order
            recommend_price = max(infos['bids_max_ref_val'].values)+1
            recommend_amount = max(infos['asks_min_ref_vol'].values)/2
            cash_amount, nb_coins = checkout()
            buy_capacity = cash_amount / order_price
            order_amount = min(buy_capacity/2, recommend_amount)
            if order_amount < 0.1:
                orders = []
            else:
                order = {'pair': pair,
                         'price': recommend_price,
                         'amount': order_amount,
                         'side': 'buy',
                         'type': 'limit'}
                orders.append(order)
                return orders
        elif trend == "desc" and confidence > 0.5:
            # sell out when very confidence for descreasing
            # sell out at market price
            recommend_price = min(infos['asks_min_ref_val'].values)-1
            recommend_amount = min(infos['bids_max_ref_vol'].values)/2
            cash_amount, nb_coins = checkout()
            if recommend_amount < nb_coins:
                order_1 = {'pair': pair,
                           'price': recommend_price,
                           'amount': recommend_amount,
                           'side': 'sell',
                           'type': 'limit'}
                order_2 = {'pair': pair,
                           'price': recommend_price,
                           'amount': nb_coins-recommend_amount,
                           'side': 'sell',
                           'type': 'market'}
                orders.append(order_1)
                orders.append(order_2)
                return orders
            else:
                order_1 = {'pair': pair,
                           'price': recommend_price,
                           'amount': nb_coins/2,
                           'side': 'sell',
                           'type': 'limit'}
                order_2 = {'pair': pair,
                           'price': recommend_price,
                           'amount': nb_coins/2,
                           'side': 'sell',
                           'type': 'market'}
                orders.append(order_1)
                orders.append(order_2)
                return orders
        elif trend == "desc" and 0.5 >= confidence > 0.3:
            # sell out when very confidence for descreasing
            # sell out at market price
            recommend_price = min(infos['asks_min_ref_val'].values)-1
            recommend_amount = min(infos['bids_max_ref_vol'].values)/2
            cash_amount, nb_coins = checkout()
            order_amount = max(nb_coins/2, recommend_amount)
            if order_amount < 0.1:
                orders = []
                return orders
            else:
                order = {'pair': pair,
                         'price': recommend_price,
                         'amount': order_amount,
                         'side': 'sell',
                         'type': 'limit'}
                orders.append(order)
                return orders
        else:
            orders = []
            return orders


def send_orders(orders):
    # make orders base on the decison
    for order in orders:
        tradeApi.order(order['pair'],
                       ':.2f'.format(order['price']),
                       ':.2f'.format(order['amount']),
                       order['side'],
                       order['type'])
        time.sleep(20)
    return 0


def train_model():
    # train models for making prediction
    cur_date, cur_time = time.strftime("%Y%m%d,%H-%M").split(',')
    X_train_cur, y_train_cur = np.array([]), np.array([])
    if os.path.exists("./log/{}/{}/".format(pair, cur_date)):
        X_train_cur, y_train_cur = generate_training_data(cur_date)
    prev_date = str(int(cur_date)-1)
    X_train_prev, y_train_prev = generate_training_data(prev_date)

    if y_train_cur.shape[0] == 0:
        X_train, y_train = X_train_prev, y_train_prev
        dataset = prev_date
    else:
        X_train, y_train = np.concatenate(X_train_cur, X_train_prev), np.concatenate(y_train_cur, y_train_prev)
        dataset = '{}-{}'.format(prev_date, cur_date)
    if os.path.exists("./models/trend_{}_clf.pkl".format(pair)):
        clf = joblib.load("./models/trend_{}_clf.pkl".format(pair))
    else:
        clf = SVC(probability=True)
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    print("[info] Training model at {} , {} with dataset of {}".format(cur_date, cur_time, dataset))
    print("[info] Training accuracy: %.03f" % score)
    # save model
    joblib.dump(clf, "./models/trend_{}_clf.pkl".format(pair))
    quality = {"training_acc":score}
    with open('./models/{}_quality.json'.format(pair), 'w') as fp:
        json.dump(quality, fp)
    return clf, quality


def generate_training_data(date):
    log_files = glob.glob(os.path.join("./log/{}/{}/".format(pair, date),"*.csv"))
    if len(log_files) == 1:
        # return empty array when there is one log
        return np.array([]), np.array([])
    else:
        records_x = []
        for log in log_files:
            records_x.append(pd.read_csv(log)[price_related].values)
        # generate labels for training labels in [-1, 0, 1]
        labels = []
        x_matrix = []
        for idx in range(1, len(records_x), 1):
            dif = records_x[idx] - records_x[idx-1]
            mean_yen = np.mean(np.sum(dif,axis = 0)/dif.shape[0])
            x_substract = records_x[idx-1] - records_x[idx-1][0, :]
            x_matrix.append(x_substract.reshape(1, -1)[:, 5:])
            if mean_yen < -2:
                labels.append(-1)
            elif -2 < mean_yen < 2:
                labels.append(0)
            else:
                labels.append(1)
        x_matrix = np.concatenate(x_matrix)
    return x_matrix, labels


if __name__ == "__main__":
    # access args
    parser = argparse.ArgumentParser(description='Argument for loging')
    parser.add_argument('-platform', type=str, default="bitbank", dest="platform",
                        help='trading platform bitbank or zaif')
    parser.add_argument('-coin', type=str, default="mona", dest="coin",
                        help='coin in [xrp, mona, bcc, btc] for loging ')
    parser.add_argument('-interval', type=int, default=15, dest="interval",
                        help='time interval(/sec) for extracting information')
    args = parser.parse_args()

    # load initial setting and set global parameters
    config = pd.read_json('config.json')
    avalible_pairs = config[args.platform]['trade_pairs']
    pair = '{}_jpy'.format(args.coin)
    if pair not in avalible_pairs:
        raise ValueError("Coin of {} to jpy is not avaliable in {}".format(args.coin, args.platform))
    print("Autotrading of {} in {}...".format(pair,  args.platform))
    price_related = ['buy','sell','last',
                     'sells_min_ref_val', 'buys_max_ref_val']
    order_related = ['asks_min_ref_val', 'bids_max_ref_val',
                     'asks_min_ref_vol', 'bids_max_ref_vol']
    trends = ["desc", "cons", "asce"]
    refApi = bitbankcc.private(config['{}-ref'.format(args.platform)]['key'],
                               config['{}-ref'.format(args.platform)]['secret'])
    tradeApi = bitbankcc.private(config['{}-trade'.format(args.platform)]['key'],
                                 config['{}-trade'.format(args.platform)]['secret'])
    assets = refApi.get_asset()["assets"]
    assets = pd.DataFrame(assets)
    assets[['free_amount', 'locked_amount','onhand_amount']] = assets[['free_amount', 'locked_amount','onhand_amount']].astype(np.float32)

    active_orders = refApi.get_active_orders(pair)['orders']
    if active_orders:
        # cancel active orders
        active_orders = pd.DataFrame(active_orders)
        for idx, active_order in active_orders.iterrows():
            tradeApi.cancel_order(pair, str(active_order["order_id"]))

    schedule.every().day.at("02:15").do(train_model)
    while True:
        schedule.run_pending()
        try:
            infos = access_info(pair, 10)
            orders = make_decison(infos)
            if orders:
                send_orders(orders)
        except:
            print("ERROR IN API, Wait 10 min")
            time.sleep(60*5)
        time.sleep(1)
