import pandas as pd
import plotnine as pn
import numpy as np
import datetime

from utils import bootstrap_lci, bootstrap_uci
from plots import plot_jitter_comparison, plot_distribution_comparison, plot_ecdf_comparison


if __name__ == "__main__":
    dt = pd.read_csv("data/air_monitor_2022_03_30.csv")

    rtd = pd.read_csv("data/resistance_training_dates.csv")
    rtd['datetime'] = pd.to_datetime(rtd.date, format='%d/%m/%Y')

    dt.columns = ['date_time', 'PM10', "PM2p5", "CO2", "Temperature", "Humidity"]

    dt['datetime'] = pd.to_datetime(dt.date_time, format='%d/%m/%Y %H:%M')

    dt['home_workout'] = dt.datetime.dt.date.isin(rtd.datetime.dt.date.values)

    startdate = datetime.datetime(2022, 1, 15)
    enddate = datetime.datetime(2022, 3, 25, 14, 30)

    dt['datetime_of_experiment'] = dt['datetime'] - datetime.datetime(2022, 1, 15)

    dt = dt[(dt.datetime > startdate)]
    dt = dt[(dt.datetime < enddate)]

    dt['purifier_on'] = dt.datetime_of_experiment.dt.days % 2 == 1

    dt['period'] = np.floor(dt.datetime_of_experiment.dt.days / 2)

    dt['constday'] = dt.datetime.apply(lambda dt: dt.replace(day=1))
    dt['constday'] = dt.constday.apply(lambda dt: dt.replace(month=1))
    dt['constday'] = dt.constday.apply(lambda dt: dt.replace(year=2022))

    # how many days did I run the experiment for?
    print(dt.datetime_of_experiment.dt.days.max())

    mean_day = dt.groupby(['purifier_on','constday']).agg(
        PM2p5 = ('PM2p5', np.mean),
        blci = ('PM2p5', bootstrap_lci),
        buci = ('PM2p5', bootstrap_uci)).reset_index()

    p = pn.ggplot(mean_day, pn.aes('constday','PM2p5', ymax='buci',ymin='blci', fill='purifier_on', colour='purifier_on')) +\
        pn.geom_vline(xintercept=datetime.datetime(2022,1,1,3,00), linetype='dashed') +\
        pn.geom_ribbon(alpha=0.33,colour=None) +\
        pn.geom_line() +\
        pn.scale_x_datetime(date_labels='%H') +\
        pn.labs(x="Hour of the day", y="PM2.5 (ug/m^3)", colour="Purifier on?", fill="Purifier on?") +\
        pn.coord_cartesian(ylim=(0,None)) +\
        pn.scale_colour_brewer(type="qual", palette = "Set1") +\
        pn.theme(legend_position=(0.35,0.75))
    p.save("results/pm2p5_day_mean.png", dpi=300)


    mean_day_training = dt.groupby(['purifier_on','constday','home_workout']).agg(
        PM2p5 = ('PM2p5', np.mean),
        blci = ('PM2p5', bootstrap_lci),
        buci = ('PM2p5', bootstrap_uci)).reset_index()

    p = pn.ggplot(mean_day_training, pn.aes('constday','PM2p5', ymax='buci',ymin='blci', fill='purifier_on', colour='purifier_on')) +\
        pn.geom_vline(xintercept=datetime.datetime(2022,1,1,3,00), linetype='dashed') +\
        pn.geom_ribbon(alpha=0.33,colour=None) +\
        pn.geom_line() +\
        pn.scale_x_datetime(date_labels='%H') +\
        pn.labs(x="Hour of the day", y="PM2.5 (ug/m^3)", colour="Purifier on?", fill="Purifier on?") +\
        pn.coord_cartesian(ylim=(0,None)) +\
        pn.facet_wrap('~home_workout', labeller = "label_both") +\
        pn.scale_colour_brewer(type="qual", palette = "Set1") +\
        pn.theme(legend_position=(0.35,0.75))
    p.save("results/pm2p5_day_mean_split_by_training.png", dpi=300)


    p = pn.ggplot(dt.query('home_workout==False'), pn.aes('constday','PM2p5', colour='purifier_on', group='datetime_of_experiment.dt.days')) +\
        pn.geom_line(alpha=0.2) +\
        pn.scale_x_datetime(date_labels='%H:%M') +\
        pn.labs(x="Time", y="PM2.5") +\
        pn.scale_colour_brewer(type="qual", palette = "Set1") + pn.facet_wrap("~purifier_on")
    p.save("results/pm2p5_day.png", dpi=200)

    dt['PM2p5_censored'] = dt.PM2p5
    dt.loc[dt.datetime.dt.time <= datetime.time(3,00), 'PM2p5_censored'] = np.nan

    mean_pm2p5 = dt.groupby('purifier_on').PM2p5_censored.describe().reset_index()
    mean_pm2p5 = mean_pm2p5.rename(columns={'mean':'PM2p5_censored'})
    mean_pm2p5['lci'] = bootstrap_lci(mean_pm2p5.query('purifier_on==True').PM2p5_censored)
    mean_pm2p5['uci'] = bootstrap_uci(mean_pm2p5.PM2p5_censored)
    print(mean_pm2p5)

    mean_pm2p5 = dt.dropna(subset=['PM2p5_censored']).groupby(['purifier_on']).agg(
        PM2p5_censored = ('PM2p5_censored', np.mean),
        lci = ('PM2p5_censored', bootstrap_lci),
        uci = ('PM2p5_censored', bootstrap_uci)).reset_index()
    print(mean_pm2p5)

    p = plot_jitter_comparison(dt, 'PM2p5_censored', 'purifier_on', stats=mean_pm2p5, limit=5, limit_label="WHO limit") +\
        pn.labs(x='Purifier on?', y='PM2.5 (ug/m^3)')
    p.save("results/pm2p5_jitter.png", width=7, height=5, units='in', dpi=300)

    # what fraction of measurements above the limit?
    print(dt.dropna(subset=['PM2p5_censored']).groupby(['purifier_on']).agg(
        above_limit = ('PM2p5_censored', lambda x: (x > 5).mean())
    ))

    p = plot_distribution_comparison(dt, 'PM2p5_censored', 'purifier_on', x_lim=(0, 20)) +\
        pn.labs(y="Proportion of measurements", colour='Purifier on?') +\
        pn.theme(legend_position=(0.65,0.65))
    p.save("results/pm2p5_dist.png", width=7, height=5, units='in', dpi=300)

    p = plot_ecdf_comparison(dt, 'PM2p5_censored', 'purifier_on', limit=5, x_lim=(0, 20)) +\
        pn.annotate("text", x=5.5, y=0.2, label="WHO limit (5 ug/m^3)", color="blue", ha="left") +\
        pn.annotate("text", x=15, y=0.77, label="75% of 'Off' data is below here", size=8, alpha=0.6) +\
        pn.geom_hline(yintercept=0.75, linetype='dotted', alpha=0.5) +\
        pn.theme(legend_position=(0.65, 0.25))
    p.save("results/pm2p5_ecdf.png", width=7, height=5, units='in', dpi=300)

    # ecdf but only Feburary
    dt_feb = dt[dt.datetime.dt.month == 2]
    p = pn.ggplot(dt_feb, pn.aes('PM2p5_censored', colour='purifier_on')) +\
        pn.stat_ecdf() +\
        pn.geom_vline(xintercept=5, linetype='dashed', colour='blue') +\
        pn.geom_hline(yintercept=0.75, linetype='dotted', alpha=0.5) +\
        pn.scale_colour_brewer(type="qual", palette = "Set1") +\
        pn.labs(x="PM2.5 (ug/m^3)", y="Cumulative Proportion", colour='Purifier on?') +\
        pn.annotate("text", x=5.5, y=0.2, label="WHO limit (5 ug/m^3)", color="blue", ha="left") +\
        pn.annotate("text", x=15, y=0.77, label="75% of 'Off' data is below here", size=8, alpha=0.6) +\
        pn.theme(legend_position=(0.65,0.25)) +\
        pn.coord_cartesian(xlim=(0,20))
    p.save("results/pm2p5_ecdf_feb.png", width=7, height=5, units='in', dpi=300)



    p = pn.ggplot(dt, pn.aes('datetime','PM2p5', colour='purifier_on', group=1)) +\
        pn.geom_line() +\
        pn.scale_x_datetime(date_labels="%b %d") +\
        pn.labs(x="Datetime", y="PM2.5 (ug/m^3)") +\
        pn.scale_colour_brewer(type="qual", palette = "Set1")
    p.save("results/pm2p5_timeline_line.png", dpi=300, width=10, height=5, limitsize=False)


    from plots import plot_timeline_area
    p = plot_timeline_area(dt, 'datetime', 'PM2p5_censored', 'purifier_on', phase_col='period') +\
        pn.scale_x_datetime(date_labels="%b %d", minor_breaks='1 day', expand = (0.01,0)) +\
        pn.labs(x="", y="PM2.5 (ug/m^3)", fill='Purifier on?') +\
        pn.theme(legend_position=(0.2,0.75))
    p.save("results/pm2p5_timeline.png", dpi=300, width=10, height=5, limitsize=False)


    p = pn.ggplot(dt, pn.aes('datetime','CO2', group=1)) +\
        pn.geom_line() +\
        pn.scale_x_datetime(date_labels="%b %d") +\
        pn.geom_smooth(method='lowess', span=0.05, colour='red') +\
        pn.labs(x="Datetime", y="CO2 (ppm)")
    p.save("results/CO2_timeline.png", dpi=200)

    mean_day_co2 = dt.groupby(['constday']).CO2.mean().reset_index()

    p = pn.ggplot(mean_day_co2, pn.aes('constday','CO2')) +\
        pn.geom_line() +\
        pn.scale_x_datetime(date_labels='%H:%M') +\
        pn.labs(x="Time", y="CO2") +\
        pn.scale_colour_brewer(type="qual", palette = "Set1")
    p.save("results/CO2_day_mean.png", dpi=200)


