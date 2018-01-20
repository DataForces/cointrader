#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:39:15 2018

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



def access_info(pair):
    # public API class
    count = 0
    records = []
    pubapi = python_bitbankcc.public()
    ref_num = 5
    while count < 30:
        count += 1
        time.sleep(30)
        pubapi = python_bitbankcc.public()
        ref_num = 5

        # extract ticker info
        ticker_res = pubapi.get_ticker(pair)
        ticker_res = [float(ticker_res['buy']),
                      float(ticker_res['sell']),
                      float(ticker_res['high']),
                      float(ticker_res['low']),
                      float(ticker_res['last'])]

        #extract transaction info
        trans_res = pubapi.get_transactions(pair)
        trans_res = pd.DataFrame(trans_res['transactions'])
        trans_res[['amount', 'price']] = trans_res[['amount', 'price']].astype(np.float32)
        # extract and sort selling and buying
        sells = trans_res[trans_res['side']=='sell']
        sells = sells.sort_values(by=['price'])
        buys = trans_res[trans_res['side']=='buy']
        buys = buys.sort_values(by=['price'], ascending=False)
        # get info according to ref number of update records
        sells_min_ref_vol = np.sum(sells.iloc[:ref_num, 0])
        sells_min_ref_val = np.sum(sells.iloc[:ref_num, 2]*sells.iloc[:ref_num, 0])/sells_min_ref_vol
        buys_max_ref_vol = np.sum(buys.iloc[:ref_num, 0])
        buys_max_ref_val = np.sum(buys.iloc[:ref_num, 2]*buys.iloc[:ref_num, 0])/buys_max_ref_vol

        # record -> [buy, sell, high, low, last, sells_min_ref_vol, sells_min_ref_val, buys_max_ref_vol, buys_max_ref_val]
        record = ticker_res+[sells_min_ref_vol, sells_min_ref_val, buys_max_ref_vol, buys_max_ref_val]
        records.append(record)
    cur_date, cur_time = time.strftime("%Y%m%d,%H:%M").split(',')
    log_dir = './{}/log/{}/'.format(pair, cur_date)
    log_file = log_dir + '_{}.csv'.format(cur_time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    records = pd.DataFrame(records,
                           columns=['buy', 'sell', 'high', 'low', 'last',
                                    'sells_min_ref_vol', 'sells_min_ref_val',
                                    'buys_max_ref_vol', 'buys_max_ref_val'])
    records.to_csv(log_file, index=False)
    print("Saving log at {}".format(cur_time))
    return 0


def multi_access():
    t1 = threading.Thread(target=access_info(trade_pairs[0]))
    t2 = threading.Thread(target=access_info(trade_pairs[1]))
    t1.start()
    t2.start()

    t1.join()
    t2.join()

    return 0



if __name__ == "__main__":
    trade_pairs = ['xrp_jpy', 'mona_jpy']
    while True:
        multi_access()
