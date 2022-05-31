import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pylab as plt


def main():
    accelerometerDataFile = "accelerometer_data_5min.csv"
    readAccelerometerData = pd.read_csv(accelerometerDataFile, sep=";")
    x = readAccelerometerData.values[:, 1][1:]
    y = readAccelerometerData.values[:, 2][1:]
    z = readAccelerometerData.values[:, 3][1:]
    time = readAccelerometerData.values[:, 0][1:]
    start = time[0]
    end = time[len(time) - 1]
    size = int((end-start)/1000) + 1
    step = size/len(x)
    seconds = [i for i in np.arange(1, size + 1, step)]

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

    ax0.plot(seconds, x, 'b', linewidth=0.5)
    ax0.legend("x")
    ax0.set_title("Показания акселерометра смартфона, \n лежащего на столе в течение пяти минут")

    ax1.plot(seconds, y, 'g', linewidth=0.5)
    ax1.legend("y")
    ax1.set_ylabel("Ускорение", fontsize=12)

    ax2.plot(seconds, z, 'r', linewidth=0.5)
    ax2.legend("z")

    plt.xlabel('Секунды', fontsize=12)
    plt.show()

    print(sum([np.sqrt(x[i]**2 + y[i]**2 + z[i]**2) for i in range(len(x))])/len(x))



main()