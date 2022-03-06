import os
import sys
import numpy as np
import datetime

def read_file1(file_path):
    f = open(file_path, "r")
    timestamps = []
    while True:
        line = f.readline()
        if line == "":
            break
        line = line[:-1]
        items = line.split('\t')
        year = int(items[2])
        month = int(items[3])
        day = int(items[4])
        hour = int(items[5])
        minitute = int(items[6])
        second = int(items[7])
        mircosecond = int(items[8])

        utc_time = datetime.datetime(year, month, day, hour, minitute, second, mircosecond)
        tt = int(utc_time.timestamp() * 1e9)
        # print(tt)
        timestamps.append(tt)

    return timestamps


def save_timestamp(out_file_path, timestamps):
    f = open(out_file_path, "w")
    for t in timestamps:
        f.write(str(t))
        f.write("\n")

def save_timestamp2(out_file_path, time_init, delta_time, num):
    f = open(out_file_path, "w")
    for i in range(num):
        f.write(str((int)((time_init + i * delta_time) * 1e9)))
        f.write("\n")

def main():
    root_path = "/media/cjg/Elements2/pano20210730/2/VID_20210730_104312"
    # timestamps = read_file1(os.path.join(root_path, "ins.txt"))
    # save_timestamp(os.path.join(root_path, "times.txt"), timestamps)

    init_time = 0 * 1e-9
    save_timestamp2(os.path.join(root_path, "times.txt"), init_time, 0.0333667000333667, 20143)


if __name__ == "__main__":
    main()