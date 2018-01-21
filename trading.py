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
import python_bitbankcc
import threading
import argparse


trader_key = 'c406977f-6cc1-4751-a059-f79124328fb3'
trader_secret = '7cc757e9cf73fcee9e30ebd157816fc35b7a9ac14f10bdfd07bd426b0501b211'


def make_decison():
    # make decison on 30 min trading records
    # decison ->[trade_or_not, pair, volumn]
    decison
    return decison


def make_orders():
    # make orders base on the decison
    records = pd.read_csv()
    order_price = round(buys_max_ref_val*1.001, 2)
    order_amount = round((sells_min_ref_vol/2), 2)
    mean_gap_percent = (sells_min_ref_val - buys_max_ref_val)/buys_max_ref_val
    return 0


def train_model():
    # train models for making prediction
    return 0


if __name__ == "__main__":
    # posiblle pair for trading of bitbank.cc
    avalible_coins = ['xrp', 'mona', 'bcc', 'btc']
    parser = argparse.ArgumentParser(description='Argument for loging')
    parser.add_argument('-coin', type=str, default="xrp", dest="coin",
                        help='coin in [xrp, mona, bcc, btc] for loging ')
    parser.add_argument('-interval', type=int, default=15, dest="interval",
                        help='time interval for trading')
    args = parser.parse_args()
    if args.coin not in avalible_coins:
        raise ValueError ("coin should in {}".format(args.coin, ','.join(avalible_coins)))
    pair = '{}_jpy'.format(args.coin)
    print("Autotrading of {} ...".format(pair))

    schedule.every(args.interval).minutes.do(make_decison)
    schedule.every().day.at("0:00").do(train_model)
    while True:
        schedule.run_pending()
        time.sleep(60)
