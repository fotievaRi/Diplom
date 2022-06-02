import numpy as np
import pandas as pd
import datetime
import pylab
import matplotlib.dates
from matplotlib import pylab


class AccelerometerMSE:
    def __init__(self, start, end, mse):
        self.start = start
        self.end = end
        self.mse = mse



def main():

    accelerometerDataFile = "monitoring_date_base-table_accelerometer_data.csv"
    userFeelDataFile = "monitoring_date_base-table_user_feel_data.csv"

    readAccelerometerData = pd.read_csv(accelerometerDataFile, sep=";")
    readUserFeelData = pd.read_csv(userFeelDataFile, sep=";")

    accelerometerData = readAccelerometerData.values[1:]
    userFeelData = readUserFeelData.values

    fiveMin = 300000
    halfFiveMin = fiveMin//2

    # Список mse для 5-минутных промежутков времени
    listMse = []

    # Для опроксимации графика mse
    approximateScheduleMse = []

    # Для опроксимации графика mse
    approximateScheduleTime = []

    i = 0
    sizeAccelerometerData = accelerometerData.shape[0] - 1
    maxMse = 0
    while i < sizeAccelerometerData:
        j = i
        N = 0
        sumN = 0
        start = accelerometerData[i][0]
        while accelerometerData[j][0] - start <= fiveMin:
            sqrt = np.sqrt(accelerometerData[j][1] ** 2 + accelerometerData[j][2] ** 2 + accelerometerData[j][3] ** 2)
            sumN += (sqrt - 9.81) ** 2
            j += 1
            N += 1
            if j > sizeAccelerometerData:
                break

        mse = np.sqrt(sumN / N)
        if maxMse < mse:
            maxMse = mse
        end = accelerometerData[j - 1][0]
        listMse.append(
            AccelerometerMSE(datetime.datetime.fromtimestamp(start / 1000), datetime.datetime.fromtimestamp(end / 1000),
                             mse))
        approximateScheduleMse.append(mse)
        approximateScheduleTime.append(datetime.datetime.fromtimestamp((start + halfFiveMin)/1000))
        i += N


    # Список промежутков времени, в которые пользователь считает себя активным (начало - конец)
    userFeelTimeActive = []
    # Список моментов времени, в которые пользователь переключает состояния
    userFeelTimeActiveSwitch = []

    for i in range(0, userFeelData.shape[0]-1, 2):
        if userFeelData[i][1] == 1:
            activeStart = datetime.datetime.fromtimestamp(userFeelData[i][0]/1000)
            activeEnd = datetime.datetime.fromtimestamp(userFeelData[i+1][0]/1000)
            userFeelTimeActive.append([activeStart, activeEnd])
            userFeelTimeActiveSwitch.append(activeStart)
            userFeelTimeActiveSwitch.append(activeEnd)

    '''
    labels = ["dws", "ups", "sit", "std", "wlk", "jog"]
    colors = ['green', 'yellow', 'blue', 'orange', 'magenta', 'brown']

    approximateScheduleMsePeriods = []
    approximateScheduleTimePeriods = []
    readAccelerometerData = pd.read_csv("ml/accelerometer_result_m2.csv")
    # readAccelerometerData = pd.read_csv("ml/accelerometer_result_m2.csv")
    typesActivity = readAccelerometerData.values[:, 0]
    startActivity = readAccelerometerData.values[:, 1]
    endActivity = readAccelerometerData.values[:, 2]
    j = 0
    for i in range(len(typesActivity)):
        approximateScheduleMsePeriod = []
        approximateScheduleTimePeriod = []
        end = datetime.datetime.fromtimestamp(endActivity[i]/1000)

        if j != 0:
            approximateScheduleMsePeriod.append(approximateScheduleMse[j-1])
            approximateScheduleTimePeriod.append(approximateScheduleTime[j-1])

        while approximateScheduleTime[j] <= end:
            approximateScheduleMsePeriod.append(approximateScheduleMse[j])
            approximateScheduleTimePeriod.append(approximateScheduleTime[j])
            j += 1
        approximateScheduleMsePeriods.append(approximateScheduleMsePeriod)
        approximateScheduleTimePeriods.append(approximateScheduleTimePeriod)
    '''



    fig, ax = pylab.plt.subplots()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
    fig.set_size_inches(40, 20)

    # Рисуем графики MSE
    labels = ["dws", "ups", "sit", "std", "wlk", "jog"]
    colors = ['green', 'yellow', 'blue', 'orange', 'magenta', 'brown']
    readModelResult = pd.read_csv("ml/accelerometer_m1.csv")
    activityTypes = readModelResult.values[:, 0]
    timeStart = readModelResult.values[:, 1]
    timeEnd = readModelResult.values[:, 2]
    count = 0
    for mse in listMse:
        pylab.hlines(mse.mse, mse.start, mse.end, colors[int(activityTypes[count])], linewidth=5)
        d = (mse.end - mse.start).seconds//30
        while count < len(activityTypes) and datetime.datetime.fromtimestamp(timeEnd[count]//1000) < mse.end:
            x = [datetime.datetime.fromtimestamp(timeStart[count]//1000),
                 datetime.datetime.fromtimestamp(timeEnd[count]//1000)]
            pylab.fill_between(x, 0, mse.mse, alpha=0.8, color=colors[int(activityTypes[count])])
            count += 1

    maxMse += 0.5
    # Рисуем график апроксимации MSE
    pylab.plot_date(matplotlib.dates.date2num(approximateScheduleTime), approximateScheduleMse, 'k--', lw=1)
    '''
    for count in range(len(typesActivity)):
        pylab.plot_date(matplotlib.dates.date2num(approximateScheduleTimePeriods[count]),
                        approximateScheduleMsePeriods[count], colors[int(typesActivity[count])], lw=5)
    '''


    '''
    for count in range(len(activePeriod)):
        if activePeriod[count]:
            color = 'magenta'
        else:
            color = 'green'
        pylab.plot_date(matplotlib.dates.date2num(approximateScheduleTimePeriods[count]),
                        approximateScheduleMsePeriods[count], color, lw=5)
    '''

    # Отмечаем на графике периоды времени, в которые пользователь считает себя активным
    for i in range(len(userFeelTimeActive)):
        pylab.hlines(maxMse, userFeelTimeActive[i][0], userFeelTimeActive[i][1], 'r', linewidth=3)
        pylab.annotate(userFeelTimeActive[i][0].strftime("%H:%M"), xy=(userFeelTimeActive[i][0], maxMse),
                       xytext=(userFeelTimeActive[i][0], maxMse + 0.2), fontsize=30)
        pylab.annotate(userFeelTimeActive[i][1].strftime("%H:%M"), xy=(userFeelTimeActive[i][1], maxMse),
                       xytext=(userFeelTimeActive[i][1], maxMse - 0.2), fontsize=30)

    # Опускам пермендикуляры пунктиром на ось времени периодов активности
    pylab.vlines(userFeelTimeActiveSwitch, 0, maxMse, color='red', linestyle='--',  alpha=0.7, linewidth=3)

    ax.set_title("Активность человека в течение суток", fontsize=30)
    ax.set_xlabel("Время суток", fontsize=30)
    ax.set_ylabel("MSE", fontsize=30)
    ax.grid(which='major', color='gray', linewidth=2)
    ax.minorticks_on()
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=2)
    pylab.gcf().autofmt_xdate()
    pylab.tick_params(axis='both', which='major', labelsize=30)


    pylab.annotate('поднятие по ступенькам', xy=(matplotlib.dates.date2num(approximateScheduleTime[0]), maxMse - 0.5),
                   color='green', fontsize=40)
    pylab.annotate('спускание по ступенькам', xy=(matplotlib.dates.date2num(approximateScheduleTime[0]), maxMse - 1.0),
                   color='yellow', fontsize=40)
    pylab.annotate('положение сидя', xy=(matplotlib.dates.date2num(approximateScheduleTime[0]), maxMse - 1.5),
                   color='blue', fontsize=40)
    pylab.annotate('положение стоя', xy=(matplotlib.dates.date2num(approximateScheduleTime[0]), maxMse - 2.0),
                   color='orange', fontsize=40)
    pylab.annotate('ходьба', xy=(matplotlib.dates.date2num(approximateScheduleTime[0]), maxMse - 2.5), color='magenta',
                   fontsize=40)
    pylab.annotate('бег трусцой', xy=(matplotlib.dates.date2num(approximateScheduleTime[0]), maxMse - 3.0),
                   color='brown', fontsize=40)


    pylab.show()


main()