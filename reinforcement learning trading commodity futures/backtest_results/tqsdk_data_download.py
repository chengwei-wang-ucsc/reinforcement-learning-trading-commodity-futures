from datetime import datetime
from contextlib import closing
from tqsdk import TqApi, TqAuth
from tqsdk.tools import DataDownloader

api = TqApi(auth=TqAuth("username", "password"))
kd = DataDownloader(api, symbol_list="KQ.m@SHFE.rb", dur_sec=900,
                    start_dt=datetime(2021, 4, 15, 0, 0 ,0), end_dt=datetime(2022, 10, 15, 23, 59, 59), 
                    csv_file_name="C:\\Users\\budin\\Desktop\\github\\reinforcement learning trading commodity futures\\RB_15min_train.csv")
# 使用with closing机制确保下载完成后释放对应的资源
with closing(api):
    while not kd.is_finished():
        api.wait_update()
        print("progress: kline: %.2f%%" % (kd.get_progress()))
