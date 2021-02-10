import pandas as pd
import numpy as np
from tqdm import tqdm
from Robotdataloader import fetch_data, init_Range

robot_id = 'd8acf141-b953-4ebc-817c-d8dbe55c3ef1'
# robot_id = 'fb4dc699-573d-4687-820c-86c33d4bc7d5'
# time_stamp_range = init_Range("test_fault")
# robot_id = '0294065d-7742-4aa0-9a36-f22a40ac69af'
time_stamp_range = init_Range("test_fault")
All_data = []
for _time_stamp in tqdm(time_stamp_range):
    try:
        s_time = _time_stamp[0]
        e_time = _time_stamp[1]
        data = fetch_data(
            start_time=s_time,
            end_time=e_time,
            robot_id=robot_id,
            sr=100,
            time_window=20,
            columns=['axle_moment', 'axle_speed', 'axle_position', 'axle_number']
        )
    except Exception:
        continue
    All_data.append(data.iloc[0: 120000, :].values)
    if len(All_data) == 1:
        break

np.save("./RobotDataset/20mins_fault_data.npy", All_data[0])