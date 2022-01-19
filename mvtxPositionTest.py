'''
Fit model to data and investigate the results

Abdennacer Hamdi, Ho San Ko, Yuan Mei
'''

import os, sys
import re
import argparse
from io import StringIO
from re import search
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import sqrt, pi
import circle_fit as cf
# sns.palplot(sns.hls_palette(8, l=.3, s=.8))
# sns.set_palette("husl")
sns.set_style("ticks",{'axes.grid' : True})
# sns.set(style = 'whitegrid')
sns.color_palette("deep")

parser = argparse.ArgumentParser(description='Fit model to data and investigate the results')
parser.add_argument('--input_model', type=str, help = "path to input model file, e.g: path/file.txt")
parser.add_argument('--input_cmm', type=str, help = "path to input CMM file, e.g: path/file.dat")
parser.add_argument('--r', default=1.0, type = float, nargs='?', help = 'fit parameter: radius, default is 1.0')
parser.add_argument('--a', default=1.0, type=float, nargs='?', help = 'fit parameter: a, default is 1.0')
parser.add_argument('--b', default=1.0, type=float, nargs='?', help = 'fit parameter: b, default is 1.0')
parser.add_argument('--g', default=1.0, type=float, nargs='?', help = 'fit parameter: g, default is 1.0')
parser.add_argument('--xt', default=1.0, type=float, nargs='?', help = 'fit parameter: xt, default is 1.0')
parser.add_argument('--yt', default=1.0, type=float, nargs='?', help = 'fit parameter: yt, default is 1.0')
parser.add_argument('--zt', default=1.0, type=float, nargs='?', help = 'fit parameter: zt, default is 1.0')
parser.add_argument('--cut', default=1000.0, type=float, nargs='?', help = 'remove outlyers with |residual| > cut, default = 1000')
args = parser.parse_args()
print(args)

# extract file and dir names from cmm data file
# cmm_file_name = os.path.basename(args.input_cmm)
cmm_file_name = os.path.splitext(os.path.basename(args.input_cmm))[0]
cmm_dir_name = os.path.dirname(args.input_cmm)
# print(cmm_file_name, cmm_dir_name)

# Fit the model to data and save fit results to output file
bashCmd = f'make fit && ./RotTransFit {args.input_model} {args.input_cmm} {args.r} {args.a} {args.b} {args.g} {args.xt} {args.yt} {args.zt} > fit_results.txt'
os.system(bashCmd)

# Read and clean data
with open('fit_results.txt','r', encoding='utf8') as f:
    fit_results = f.read()
# print(fit_results)

# separate the fit results into two table one for plane and second for cylinder
fit_results_plane_df = pd.read_csv(StringIO(fit_results.split("=")[0].strip()), sep=r'\s+', names=['fid', 'ftype', 'distance','x1','y1', 'z1'], skiprows=1)
fit_results_plane_df.insert(loc=0, column='point', value=fit_results_plane_df.index)

fit_results_cylinder_df = pd.read_csv(StringIO(fit_results.split("=")[-1].strip()), sep=r'\s+', names=['fid', 'ftype', 'distance','x1','y1', 'z1', 'd_nx', 'd_ny', 'd_nz', 'd'], skiprows=1)
fit_results_cylinder_df.insert(loc=0, column='point', value=fit_results_cylinder_df.index)

print(fit_results_plane_df.head(10))
print(fit_results_cylinder_df.head(10))


def plotResults(fit_results_plane_df, fit_results_cylinder_df, figName=""):

    # ------ draw residuals of CMM points
    fig_residuals_plane = sns.relplot(x='point', y='distance', hue='fid', style='ftype', palette='tab10', data=fit_results_plane_df)
    # Set x-axis and y-axis labels
    # fig_residuals_plane.set_axis_labels( "Point ID" , "Residuals (mm)") #$\\mu$m
    # set the range
    # fig_residuals_plane.set(ylim=(-60, 40))
    # plt.show()
    fig_residuals_plane.savefig(f'residuals_plane{figName}.pdf')
        
    # ------ draw stave model planes and compare to CMM points z-position
    
    # df_line_select = fit_results_plane_df[['y1']]
    # df_line = df_line_select.copy()
    
    # L0
    if re.search('L0', f'{args.input_model}'):
        print('------- contains : L0 -------')
        df_line_select = fit_results_plane_df.loc[(fit_results_plane_df['fid'] >= 3) & (fit_results_plane_df['fid'] <= 8)]
        df_line = df_line_select.copy()
        df_line['plane1'] = np.where((df_line['y1'] > 151.913) & (df_line['y1'] < 156.383), (-0.8910075863383323*(df_line['y1'] - 150.63020975589296))/(-0.45398841514683985) + 129.19987639033684, 0)
        df_line['plane2'] = np.where((df_line['y1'] > 143.814) & (df_line['y1'] < 152.87), (-(-0.544640997160222)*(df_line['y1'] - 141.4443001405247))/(0.8386692937101722) + 118.99784221138184, 0)
        df_line['plane3'] = np.where((df_line['y1'] > 131.21) & (df_line['y1'] < 141.993), (-(-0.05233829262814875)*(df_line['y1'] - 128.3880519672707))/(0.9986294123070731) + 114.75557624981391, 0)
        df_line['plane4'] = np.where((df_line['y1'] > 117.477) & (df_line['y1'] < 127.099), (-(0.4539884151468406)*(df_line['y1'] - 114.9598763903351))/(0.891007586338332) + 117.60979024411257, 0)
        df_line['plane5'] = np.where((df_line['y1'] > 106.297) & (df_line['y1'] < 112.178), (-(0.8386692937101663)*(df_line['y1'] - 104.75784221138012))/(0.5446409971602311) + 126.79569985948051, 0)
        df_line['plane6'] = np.where((df_line['y1'] > 100) & (df_line['y1'] < 102), (-(0.9986294123070726)*(df_line['y1'] - 100.51557624981201))/(0.052338292628161504) + 139.85194803273447, 0)
    else:
        # L2
        print('------- contains : L2 -------')
        df_line_select = fit_results_plane_df.loc[(fit_results_plane_df['fid'] >= 3) & (fit_results_plane_df['fid'] <= 12)]
        df_line = df_line_select.copy()
        df_line['plane1'] = np.where((df_line['y1'] > 168.515) & (df_line['y1'] < 172.122), (0.9025820648868889*(df_line['y1'] - 164.58025404698367))/(0.4305178464878314) + 123.31479333771121, 0)
        df_line['plane2'] = np.where((df_line['y1'] > 163.493) & (df_line['y1'] < 169.26), (0.7253692233553658*(df_line['y1'] - 157.20175200908088))/(0.6883600001516894) + 112.57917824444718, 0)
        df_line['plane3'] = np.where((df_line['y1'] > 154.899) & (df_line['y1'] < 151.812), (0.4771521882972612*(df_line['y1'] - 146.86689205654955))/(0.8788206809145624) + 104.64908407630396, 0)
        df_line['plane4'] = np.where((df_line['y1'] > 143.574) & (df_line['y1'] < 127.099), (0.1822281725338393*(df_line['y1'] - 134.58732228874692))/(0.9832562703257872) + 100.30076370267443, 0)
        df_line['plane5'] = np.where((df_line['y1'] > 130.626) & (df_line['y1'] < 138.933), (-0.13053360641553088*(df_line['y1'] - 121.56505255133887))/(0.9914438852482551) + 99.95986101825895, 0)
        df_line['plane6'] = np.where((df_line['y1'] > 117.324) & (df_line['y1'] < 124.886), (-0.43051784648783403*(df_line['y1'] - 109.07479333771137))/(0.9025820648868877) + 103.65974595301677, 0)
        df_line['plane7'] = np.where((df_line['y1'] > 104.969) & (df_line['y1'] < 111.046), (-0.688360000151692*(df_line['y1'] - 98.33917824444737))/(0.7253692233553634) + 111.03824799091961, 0)
        df_line['plane8'] = np.where((df_line['y1'] > 94.77) & (df_line['y1'] < 98.768), (-0.8788206809145638*(df_line['y1'] - 90.40908407630418))/(0.4771521882972583) + 121.37310794345096, 0)
        df_line['plane9'] = np.where((df_line['y1'] > 85.726) & (df_line['y1'] < 89.253), (-0.9832562703257878*(df_line['y1'] - 86.0607637026747))/(0.18222817253383589) + 133.65267771125363, 0)
        df_line['plane10'] = np.where((df_line['y1'] > 83.433) & (df_line['y1'] < 86.622), -(-0.9914438852482547*(df_line['y1'] - 85.71986101825925))/(0.1305336064155345) + 146.67494744866164, 0)
    
        df_line7 = df_line[df_line['plane7'] != 0.0]
        df_line8 = df_line[df_line['plane8'] != 0.0]
        df_line9 = df_line[df_line['plane9'] != 0.0]
        df_line10 = df_line[df_line['plane10'] != 0.0]
    
    df_line1 = df_line[df_line['plane1'] != 0.0]
    df_line2 = df_line[df_line['plane2'] != 0.0]
    df_line3 = df_line[df_line['plane3'] != 0.0]
    df_line4 = df_line[df_line['plane4'] != 0.0]
    df_line5 = df_line[df_line['plane5'] != 0.0]
    df_line6 = df_line[df_line['plane6'] != 0.0]
    
    # print(df_line.head(50))
    
    # fig_yz.map_dataframe(sns.lineplot, x="x1", y="line1", color="g")
    fig_yz, ax_yz = plt.subplots(figsize=(6, 4))
    fig_yz_cmm = sns.scatterplot(data=df_line, x='y1', y='z1', hue='fid', palette='tab10', ax=ax_yz)
    fig_yz_1 = sns.lineplot(data=df_line1, x='y1', y='plane1', color='b', ax=ax_yz)
    fig_yz_2 = sns.lineplot(data=df_line2, x='y1', y='plane2', color='b', ax=ax_yz)
    fig_yz_3 = sns.lineplot(data=df_line3, x='y1', y='plane3', color='b', ax=ax_yz)
    fig_yz_4 = sns.lineplot(data=df_line4, x='y1', y='plane4', color='b', ax=ax_yz)
    fig_yz_5 = sns.lineplot(data=df_line5, x='y1', y='plane5', color='b', ax=ax_yz)
    fig_yz_6 = sns.lineplot(data=df_line6, x='y1', y='plane6', color='b', ax=ax_yz)
    # L2
    if re.search('L2', f'{args.input_model}'):
        fig_yz_7 = sns.lineplot(data=df_line7, x='y1', y='plane7', color='b', ax=ax_yz)
        fig_yz_8 = sns.lineplot(data=df_line8, x='y1', y='plane8', color='b', ax=ax_yz)
        fig_yz_9 = sns.lineplot(data=df_line9, x='y1', y='plane9', color='b', ax=ax_yz)
        fig_yz_10 = sns.lineplot(data=df_line10, x='y1', y='plane10', color='b', ax=ax_yz)
    
    fig_yz.savefig(f'yz{figName}.pdf')

    # ------ draw CMM translated-and-rotated dx vs. dy with circle fit for each pin
    dfs = [x for _, x in fit_results_cylinder_df.groupby('fid')]
    # print(fit_results_cylinder_df.head(40))

    x_center = np.empty(len(dfs), dtype=float) 
    y_center = np.empty(len(dfs), dtype=float) 
    center = np.empty(len(dfs), dtype=float)

    for i in range(0, len(dfs)):
        # print("i = ", i)
        # print(dfs[i].head(20))

        coords = dfs[i][['d_nx', 'd_ny']].to_numpy()
        xc, yc, r, s = cf.least_squares_circle(coords)
        x_center[i] = xc
        y_center[i] = yc
        center[i] = sqrt(xc*xc + yc*yc)
        # print('The coordinates of the circle : ', xc, yc, r, s)
        # cf.plot_data_circle(dfs[i]['d_nx'],dfs[i]['d_ny'],xc,yc,r)

        plt.figure(facecolor='white')
        plt.axis('equal')

        theta_fit = np.linspace(-pi, pi, 180)
        x_fit = xc + r*np.cos(theta_fit)
        y_fit = yc + r*np.sin(theta_fit)

        # plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
        plt.plot(x_fit, y_fit, 'r-' , label="fitted circle", lw=2)
        plt.plot([xc], [yc], 'bD', mec='y', mew=1)
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        # plot data
        # plt.scatter(dfs[i].d_nx, dfs[i].d_ny, c=dfs[i].fid, label=f'pin {i}')
        sns.scatterplot(data=dfs[i], x='d_nx', y='d_ny', hue='fid', palette='dark')

        plt.legend(loc='best',labelspacing=0.1)
        plt.grid()
        plt.title('Fit Circle')
        plt.savefig(f'dxdy_{i}{figName}.pdf')

    fig_center, axs = plt.subplots(1, 3, figsize=(16,4))
    # axs[0].errorbar(np.arange(0, len(dfs), 1), x_center, yerr=0, fmt='o')
    axs[0].plot(np.arange(0, len(dfs), 1), x_center, 's')
    axs[1].plot(np.arange(0, len(dfs), 1), y_center, 's')
    axs[2].plot(np.arange(0, len(dfs), 1), center, 's')
    axs[0].set_title("X position residual from Cylinder fit")
    axs[1].set_title("Y position residual from Cylinder fit")
    axs[2].set_title("Center position residual from Cylinder fit")
    axs[0].set(xlabel='Pin ID', ylabel='x-residual (mm)')
    axs[1].set(xlabel='Pin ID', ylabel='y-residual (mm)')
    axs[2].set(xlabel='Pin ID', ylabel='radius-residual (mm)')
    fig_center.savefig(f'center{figName}.pdf')

plotResults(fit_results_plane_df, fit_results_cylinder_df, "")

# ------------ remove outlier points from CMM data file if the residuals exceed xxx mm

# check if there are outlier, if so remove them and refit again to produce new plots
if (abs(fit_results_plane_df['distance']) > args.cut).any() :

    # grab the cmm points index that do not satisfy/satisfy the residuals condition 
    outliers = [i for i in fit_results_plane_df.index[abs(fit_results_plane_df['distance']) > args.cut]]
    passed_cmm = [i for i in fit_results_plane_df.index[abs(fit_results_plane_df['distance']) < args.cut]]
    
    # read the CMM data file as DataFrame
    with open(f'{args.input_cmm}','r', encoding='utf8') as f:
        cmm_data_original = f.read()

    df_cmm_data = pd.read_csv(StringIO(cmm_data_original.strip()), sep=';', header=None)
    # remove the empty column
    # df_cmm_data.dropna(how='all', axis=1, inplace=True)
    # add a header
    df_cmm_data.columns = ['cmm_point', 'fid', 'name','unknown', 'x','y', 'z','unknown', 'sigma']

    # remove the outliers rows
    # df_cmm_data.drop(outliers, inplace=True)

    # print(df_cmm_data.at[1, 'fid'])
    # print(df_cmm_data.iloc[:, 0])
    df_cmm_data_new = df_cmm_data.copy()
    df_cmm_data_new.loc[outliers, ['sigma']] = 10000.000
    df_cmm_data_new.loc[passed_cmm, ['sigma']] = 0.0010

    # print(df_cmm_data_new.head(20))

    cmm_file_new = f'{cmm_dir_name}/{cmm_file_name}_new.dat'
    df_cmm_data_new.to_csv(cmm_file_new, header=None, index=None, sep=';')

    # Fit the model to data and save fit results to output file

    bashCmd_new = f'make fit && ./RotTransFit {args.input_model} {cmm_file_new} {args.r} {args.a} {args.b} {args.g} {args.xt} {args.yt} {args.zt} > fit_results_new.txt'
    os.system(bashCmd_new)

    # Read and clean data
    with open('fit_results_new.txt','r', encoding='utf8') as f_new:
       fit_results_new = f_new.read()

    # separate the fit results into two table one for plane and second for cylinder
    fit_results_plane_df_new = pd.read_csv(StringIO(fit_results_new.split("=")[0].strip()), sep=r'\s+', names=['fid', 'ftype', 'distance','x1','y1', 'z1'], skiprows=1)
    fit_results_plane_df_new.insert(loc=0, column='point', value=fit_results_plane_df_new.index)

    fit_results_cylinder_df_new = pd.read_csv(StringIO(fit_results_new.split("=")[-1].strip()), sep=r'\s+', names=['fid', 'ftype', 'distance','x1','y1', 'z1', 'd_nx', 'd_ny', 'd_nz', 'd'], skiprows=1)
    fit_results_cylinder_df_new.insert(loc=0, column='point', value=fit_results_cylinder_df_new.index)

    print(fit_results_plane_df_new.head(10))
    print(fit_results_cylinder_df_new.head(10))

    plotResults(fit_results_plane_df_new, fit_results_cylinder_df_new, figName="_new")

    print(f' ------------- There are outliers: {outliers}---------------')
else :
    print(' ----------------------- No outliers ------------------------')