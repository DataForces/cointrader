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
import sys
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
    coin_name = pair.split("_")[0]
    cash = assets[assets['asset']=="jpy"]['free_amount'].values[0]
    coin_asset = {}
    coin_record = []
    total = cash
    query_time = time.strftime("%Y%m%d-%H-%M")
    for _coin in tradable_coins:
        nb_coin = assets[assets['asset']==_coin]['free_amount'].values[0]
        coin_jpy = pubapi.get_ticker('{}_jpy'.format(_coin))['last']
        coin_asset[_coin] = [nb_coin, float(coin_jpy)]
        coin_record += coin_asset[_coin]
        total += (nb_coin * float(coin_jpy))
    print("[asset] Cash(JPY):{:.2f}; MONA-Coins:{:.2f}; XRP-Coins:{:.2f}; BTC-Coins:{:.2f}; BCC-Coins:{:.2f}.".
          format(cash, coin_asset['mona'][0], coin_asset['xrp'][0], coin_asset['btc'][0], coin_asset['bcc'][0]))
    sys.stdout.flush()
    cur_asset = [query_time, cash] + coin_record + [total]
    cur_asset = pd.DataFrame([cur_asset],
                             columns=['time', 'jpy', 'mona', 'mona_jpy',
                                      'xrp', 'xrp_jpy', 'btc', 'btc_jpy',
                                      'bcc', 'bcc_jpy', 'estimation'])

    assets_log = pd.read_csv("./assets/assets-log.csv")
    assets_log = assets_log.append(cur_asset, ignore_index=True)
    assets_log.to_csv("./assets/assets-log.csv", index=False)
    print("[info] Loging asset")
    sys.stdout.flush()
    return cash, coin_asset[coin_name]


def make_decison(infos, tmp_pred):
    # make decison on 30 min trading records
    # decison ->[trade_or_not, pair, volumn]
    orders = []
    if os.path.exists("./models/trend_{}_clf.pkl".format(pair)):
        clf = joblib.load("./models/trend_{}_clf.pkl".format(pair))
        with open('./models/{}_quality.json'.format(pair), 'r') as fp:
            quality = json.load(fp)
    else:
        clf, quality = train_model()

    if float(quality['training_acc']) < 0.8:
        print("[prediction] Skip making decison due to poor model accuracy.")
        sys.stdout.flush()
        train_model()
    else:
        record = infos[price_related].values
        record = record - record[0, :]
        record = record.reshape(1, -1)[:, 5:]
        pred_proba = clf.predict_proba(record)
        trend = trends[np.argmax(pred_proba)]
        confidence = np.max(pred_proba)
        prev_trend = tmp_pred["trend"]
        prev_confidence = tmp_pred["confidence"]
        print("[prediction] Previous trend : {}, confidence : {:.02f}".
              format(prev_trend, prev_confidence))
        print("[prediction] Current trend : {}, confidence : {:.02f}".
              format(trend, confidence))
        sys.stdout.flush()
        # when decison is asce, buy some coins
        if trend == "asce" and prev_trend == "asce":
            if confidence > 0.5 and prev_confidence > 0.5:
                # generate order
                recommend_price = max(infos['bids_max_ref_val'].values)+1
                recommend_amount = max(infos['asks_min_ref_vol'].values)/5
                cash_amount, nb_coins = checkout()
                buy_capacity = cash_amount / recommend_price
                order_amount = min(buy_capacity/5, recommend_amount)
                if nb_coins > args.max or order_amount < args.min:
                    orders = []
                else:
                    order = {'pair': pair,
                             'price': recommend_price,
                             'amount': order_amount,
                             'side': 'buy',
                             'type': 'limit'}
                    orders.append(order)
            else:
                orders = []
        # when decison is desc, sell some coins
        elif trend == "desc" and prev_trend == "desc":
            if confidence > 0.4 and prev_confidence > 0.4:
                # sell 1/3 when very confidence for descreasing
                recommend_price = min(infos['asks_min_ref_val'].values)-1
                recommend_amount = min(infos['bids_max_ref_vol'].values)/5
                cash_amount, nb_coins = checkout()
                order_amount = min(nb_coins/3, recommend_amount)
                if nb_coins > 3*args.min:
                    order_1 = {'pair': pair,
                               'price': recommend_price,
                               'amount': nb_coins-order_amount,
                               'side': 'sell',
                               'type': 'limit'}
                    order_2 = {'pair': pair,
                               'price': recommend_price,
                               'amount': order_amount,
                               'side': 'sell',
                               'type': 'market'}
                    orders.append(order_1)
                    orders.append(order_2)
                else:
                    orders = []
            else:
                # sell part of coins when not so confidence
                recommend_price = min(infos['asks_min_ref_val'].values)-1
                recommend_amount = min(infos['bids_max_ref_vol'].values)/5
                cash_amount, nb_coins = checkout()
                order_amount = min(nb_coins/4, recommend_amount)
                if order_amount < 0.01:
                    orders = []
                    return orders
                else:
                    order = {'pair': pair,
                             'price': recommend_price,
                             'amount': order_amount,
                             'side': 'sell',
                             'type': 'limit'}
                    orders.append(order)
        else:
            orders = []
        tmp_pred = {"trend":trend, "confidence":confidence}
        return orders, tmp_pred


def send_orders(orders):
    # make orders base on the decison
    for order in orders:
        order_time = time.strftime("%Y%m%d-%H-%M")
        print("[order] Sending order at {}.".format(order_time))
        print("[order] Pair:{} ; Price:{} ; Amount:{} ; Side:{} ; Type:{}.".
              format(order['pair'],
                     '{:.2f}'.format(order['price']),
                     '{:.2f}'.format(order['amount']),
                     order['side'],
                     order['type']))
        sys.stdout.flush()
        tradeApi.order(order['pair'],
                       '{:.2f}'.format(order['price']),
                       '{:.2f}'.format(order['amount']),
                       order['side'],
                       order['type'])
        time.sleep(10)
    return 0


def train_model():
    # train models for making prediction
    cur_date, cur_time = time.strftime("%Y%m%d,%H-%M").split(',')
    prev_date = str(int(cur_date)-1)
    X_train_cur, y_train_cur = np.array([]), []
    X_train_prev, y_train_prev = np.array([]), []
    if os.path.exists("./log/{}/{}/".format(pair, cur_date)):
        X_train_cur, y_train_cur = generate_training_data(cur_date)
    if os.path.exists("./log/{}/{}/".format(pair, prev_date)):
        X_train_prev, y_train_prev = generate_training_data(prev_date)

    if len(y_train_cur) == 0:
        X_train, y_train = X_train_prev, y_train_prev
        dataset = prev_date
    elif len(y_train_prev) == 0:
        X_train, y_train = X_train_cur, y_train_cur
        dataset = cur_date
    else:
        X_train = np.concatenate((X_train_cur, X_train_prev), axis=0)
        y_train = y_train_cur+y_train_prev
        dataset = '{}-{}'.format(prev_date, cur_date)
    if os.path.exists("./models/trend_{}_clf.pkl".format(pair)):
        clf = joblib.load("./models/trend_{}_clf.pkl".format(pair))
    else:
        clf = SVC(probability=True)
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    print("[model] Training model at {}-{}.".format(cur_date, cur_time))
    print("[model] Training model using dataset of {} with {}-records".format(dataset, len(y_train)))
    print("[model] Training accuracy: %.03f" % score)
    sys.stdout.flush()
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
        return np.array([]), []
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
    parser.add_argument('-interval', type=int, default=10, dest="interval",
                        help='time interval(/sec) for extracting information')
    parser.add_argument('-max', type=int, default=50, dest="max",
                        help='max number of coins')
    parser.add_argument('-min', type=float, default=0.01, dest="min",
                        help='min number for trading')
    args = parser.parse_args()

    # load initial setting and set global parameters
    config = pd.read_json('config.json')
    tradable_coins = config[args.platform]['coins']
    if args.coin not in tradable_coins:
        raise ValueError("Coin of {} to jpy is not tradable in {}".format(args.coin, args.platform))
    pair = '{}_jpy'.format(args.coin)
    print("[start] Autotrading of {} in {}...".format(pair,  args.platform))
    sys.stdout.flush()
    price_related = ['buy','sell','last',
                     'sells_min_ref_val', 'buys_max_ref_val']
    order_related = ['asks_min_ref_val', 'bids_max_ref_val',
                     'asks_min_ref_vol', 'bids_max_ref_vol']
    trends = ["desc", "cons", "asce"]
    pubapi = bitbankcc.public()
    refApi = bitbankcc.private(config['{}-ref'.format(args.platform)]['key'],
                               config['{}-ref'.format(args.platform)]['secret'])
    tradeApi = bitbankcc.private(config['{}-trade'.format(args.platform)]['key'],
                                 config['{}-trade'.format(args.platform)]['secret'])
    # initial training of the model
    train_model()
    checkout()
    schedule.every(6).hour.do(train_model)
    schedule.every().day.at("01:30").do(train_model)
    tmp_pred = {"trend":"cons", "confidence":0}
    while True:
        schedule.run_pending()
        time.sleep(1)
        try:
            infos = access_info(pair, args.interval)
            try:
                orders , tmp_pred = make_decison(infos, tmp_pred)
                try:
                    if orders:
                        send_orders(orders)
                except:
                    print("[error] Errors in sending order")
                    sys.stdout.flush()
            except:
                print("[error] Error in making decison")
                sys.stdout.flush()
        except:
            print("[error] Error with API, Wait 1 min")
            sys.stdout.flush()
            time.sleep(60)
