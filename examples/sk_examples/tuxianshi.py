from examples.sk_examples import banpei
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DatetimeTickFormatter
from bokeh.plotting import figure
from datetime import datetime
from math import radians
from pytz import timezone

raw_data123 = pd.read_csv('xrb.csv')

# raw_data = pd.read_csv('./data/periodic_wave.csv')
raw_data = np.array(raw_data123['label'])
raw_data1 = np.array(raw_data123['label1'])
# raw_date = np.array(raw_data['timestamp'])
data = []
results = []
data1 = []
results1 = []


def get_new_data():
    global data
    global raw_data
    data.append(raw_data[0])
    raw_data = np.delete(raw_data, 0)

    global raw_data1
    data1.append(raw_data1[0])
    raw_data1= np.delete(raw_data1, 0)


def update_data():
    global results
    get_new_data()
    ret = model.stream_detect(data)
    results.append(ret)
    # now = [date[-1]](tz=timezone("Asia/Tokyo"))
    now = datetime.now(tz=timezone("Asia/Tokyo"))
    new_data = dict(x=[now], y=[data[-1]], ret=[results[-1]])
    source.stream(new_data, rollover=500)

# Create Data Source
source = ColumnDataSource(dict(x=[], y=[], ret=[]))


# Create Banpei instance
model = banpei.SST(w=30)

# Draw a graph
fig = figure(x_axis_type="datetime",
             x_axis_label="Datetime",
             plot_width=950,
             plot_height=650)
fig.title.text = "Realtime monitoring with Banpei"
fig.line(source=source, x='x', y='y', line_width=2, alpha=.85, color='blue', legend_label='observed data')
fig.line(source=source, x='x', y='ret', line_width=2, alpha=.85, color='red', legend_label='change-point score')
fig.circle(source=source, x='x', y='y', line_width=2, line_color='blue', color='blue')
fig.legend.location = "top_left"

# Configuration of the axis
format = "%Y-%m-%d-%H-%M-%S"
fig.xaxis.formatter = DatetimeTickFormatter(
    seconds=[format],
    minsec =[format],
    minutes=[format],
    hourmin=[format],
    hours  =[format],
    days   =[format],
    months =[format],
    years  =[format]
)
fig.xaxis.major_label_orientation=radians(90)

# Configuration of the callback
curdoc().add_root(fig)
curdoc().add_periodic_callback(update_data, 10) #ms