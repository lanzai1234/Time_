import os
import time
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from influxdb import InfluxDBClient
from datetime import datetime, timedelta
from .error import DataloaderError, _code2mess

SIGNAL_RESAMPLE = lambda df, origin_sr, target_sr: df.apply(lambda x: librosa.resample(np.asfortranarray(x.values), orig_sr=origin_sr, target_sr=target_sr, fix=True))

def fetch_data(start_time, end_time, robot_id, sr, time_window, columns, pace=1):
    host = '10.1.1.46'
    port = 8086
    user = 'admin'
    pwd = 'bhyjy2019influx'
    dbname = 'robots'
    db_conn = InfluxDBClient(host=host, port=port, username=user, password=pwd, database=dbname)

    query = 'SELECT {} FROM "axle_dynamic_info" WHERE time >= {}s AND time <= {}s AND "robot_id" = \'{}\''.format(str(columns).replace('\'', '\"')[1:-1], start_time, end_time, robot_id)
    df_all = pd.DataFrame(db_conn.query(query, epoch='ms').get_points())

    if not len(df_all) > 0:
        raise DataloaderError(sample_len=np.nan, message=_code2mess[1])
    if not df_all.iloc[0, 0] - start_time * 1000 < 3000:
        raise DataloaderError(sample_len=len(df_all) // 6, message=_code2mess[2])
    if not end_time * 1000 - df_all.iloc[-1, 0] < 3000:
        raise DataloaderError(sample_len=len(df_all) // 6, message=_code2mess[2])
    if not len(df_all) // 6 > sr * time_window * 60 * 0.9:
        raise DataloaderError(sample_len=len(df_all) // 6, message=_code2mess[3])

    df_all = df_all.pivot(index='time', columns='axle_number')
    df_all.columns = [name[0] + '_' + name[1] for name in df_all.columns]
    df_all.dropna(axis=0, inplace=True)

    time_window = pace * time_window * 60
    origin_sr = len(df_all) / time_window
    target_sr = sr

    # 使用librosa.resample对信号进行重采样，使得信号频率为100Hz
    df_all = SIGNAL_RESAMPLE(df_all, origin_sr, target_sr)
    # 将信号按照pace进行采样
    df_all = SIGNAL_RESAMPLE(df_all, pace*target_sr, target_sr)
    # print(len(df_all))

    return df_all

def collect_data(years, months, days, hours, minutes, time_span, **kwargs):
    delta_time = timedelta(seconds=time_span * 60 - 1)
    time_iter = [datetime(year=t[0], month=t[1], day=t[2], hour=t[3], minute=t[4], second=0) for t in product(years, months, days, hours, minutes)]
    for t in tqdm(time_iter):
        s_time = int(time.mktime(t.timetuple()))
        e_time = int(time.mktime((t + delta_time).timetuple()))
        try:
            data = fetch_data(s_time, e_time, **kwargs)
        except Exception:
            continue

        dirs = os.path.join('.\\data\\IndustrialRobot', kwargs['robot_id'], t.strftime('%Y-%m-%d'))
        filename = '{}_{}Hz.csv'.format(s_time, kwargs['sr'])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        data.to_csv(os.path.join(dirs, filename))

if __name__ == '__main__':
    import pandas as pd
    from dataloader import collect_data
    from influxdb import InfluxDBClient

    host = '10.1.1.46'
    port = 8086
    user = 'admin'
    pwd = 'bhyjy2019influx'
    dbname = 'robots'

    robot_id = '0294065d-7742-4aa0-9a36-f22a40ac69af'
    time_span = 60
    period = 19150
    sample_rate = 100
    client = InfluxDBClient(host=host, port=port, username=user, password=pwd, database=dbname)

    collect_data(
        years=[2020],
        months=[5, 6, 7],
        days=[_ for _ in range(1, 31)],
        hours=[_ for _ in range(24)],
        # minutes=[_ for _ in np.arange(0, 60, time_span, dtype=np.int)],
        minutes=[0],
        time_span=time_span,
        db_conn=client,
        robot_id=robot_id,
        sr=sample_rate
    )