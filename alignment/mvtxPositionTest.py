'''
Fit CMM data measurements against detector model and investigate the results

detailed setup go to: https://github.com/nacerh/mvtx/blob/main/README.md

Abdennacer Hamdi, Ho San Ko, Yuan Mei
'''
import os, sys
import re
import argparse
from io import StringIO
from re import search
from turtle import color
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import cufflinks as cfl
import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px
from math import sqrt, pi
import circle_fit as cf
sns.set_style("ticks",{'axes.grid' : True})
sns.color_palette("deep")
import plotly.io as pio
pio.renderers.default = "vscode"

parser = argparse.ArgumentParser(description='Fit model to data and investigate the results')
parser.add_argument('--input_model', type=str, help = "path to input model file, e.g: path/file.txt")
parser.add_argument('--input_cmm', type=str, help = "path to input CMM file, e.g: path/file.dat")
parser.add_argument('--r', default=1.0, type = float, nargs='?', help = 'fit parameter: radius, default is 1.0')
parser.add_argument('--a', default=0.1, type=float, nargs='?', help = 'fit parameter: a, default is 0.1')
parser.add_argument('--b', default=0.1, type=float, nargs='?', help = 'fit parameter: b, default is 0.1')
parser.add_argument('--g', default=0.1, type=float, nargs='?', help = 'fit parameter: g, default is 0.1')
parser.add_argument('--xt', default=0.1, type=float, nargs='?', help = 'fit parameter: xt, default is 0.1')
parser.add_argument('--yt', default=0.1, type=float, nargs='?', help = 'fit parameter: yt, default is 0.1')
parser.add_argument('--zt', default=0.1, type=float, nargs='?', help = 'fit parameter: zt, default is 0.1')
parser.add_argument('--xt0', default=0.0, type=float, nargs='?', help = 'model translation parameter: xt0, default is 0.0')
parser.add_argument('--fidList', default=[], type=int, nargs='+', help = 'selected fid receive sigma = sigmaList')
parser.add_argument('--pointList', default=[], type=int, nargs='+', help = 'selected point receive sigma = sigmaList')
parser.add_argument('--sigmaList', default=[], type=float, nargs='+', help = 'enter sigma value corresponding to fid and point list selected in the following order')
args = parser.parse_args()
print(args)

def cmmCleanFit(argsfidList, argspointList, argssigmaList):
    # extract file and dir names from cmm data file
    cmm_file_name = os.path.splitext(os.path.basename(args.input_cmm))[0]
    cmm_dir_name = os.path.dirname(args.input_cmm)

    # read the CMM data file as DataFrame
    with open(f'{args.input_cmm}','r', encoding='utf8') as f:
        cmm_data_raw = f.read()

    df_cmm_data = pd.read_csv(StringIO(cmm_data_raw.strip()), sep=';', header=None)

    # add a header
    df_cmm_data.columns = ['cmm_point', 'name', 'fid', 'unknown1', 'x','y', 'z','point', 'sigma']
    df_cmm_data['fid'] = df_cmm_data['fid'].str.replace(r'\D', '', regex=True)

    #now swap order betwee 'fid' and 'name'
    df_cmm_data = df_cmm_data[['cmm_point', 'fid', 'name', 'unknown1', 'x','y', 'z','point', 'sigma']]

    ### Set pins with fid = [13 - 22] to sigma = 10000
    df_cmm_data['fid'] = df_cmm_data['fid'].astype(int)

    ### initialze all pins with sigma = 0.005
    df_cmm_data.loc[df_cmm_data['fid'] > -1, ['sigma']] = 0.005

    ### Set pins with fid = [13 - 22] to sigma = 10000
    df_cmm_data.loc[(df_cmm_data['fid'] > 12) & (df_cmm_data['fid'] < 23), ['sigma']] = 10000

    ### replace the point column original values (NaN) with dataframe index list to use it in the selection
    df_cmm_data['point'] = df_cmm_data.index

    if (len(argsfidList) > 0 or len(argspointList) > 0 or len(argssigmaList) > 0) :
        print(f' ---------- fid = {argsfidList} | point =  {argspointList} | sigma = {argssigmaList}-----------')
        ### convert from str to int
        fidList = [i for i in argsfidList]
        pointList = [i for i in argspointList]
        sigmaList = [i for i in argssigmaList]

        ### modify sigma with choice of 'fid' or 'point' in the arguments

        for i in range(len(fidList)):
            df_cmm_data.loc[df_cmm_data['fid'] == fidList[i], ['sigma']] = sigmaList[i]
        for i in range(len(pointList)):
            df_cmm_data.loc[df_cmm_data['point'] == pointList[i], ['sigma']] = sigmaList[i+len(fidList)]
    else:
        print(" ---------- use all cmm fid and points with raw sigma -----------")

    # rewrite the cmm file
    cmm_file_corrected = f'{cmm_dir_name}/{cmm_file_name}_corrected.dat'
    df_cmm_data.to_csv(cmm_file_corrected, header=None, index=None, sep=';')

    # columnsTitles=["fid","name"]
    # df_cmm_data=df_cmm_data.reindex(columns=columnsTitles)
    # pd.set_option('display.max_rows', None)
    # print(df_cmm_data)

    # Fit the model to data and save fit results to output file
    bashCmd = f'make fit && ./RotTransFit {args.input_model} {cmm_file_corrected} {args.r} {args.a} {args.b} {args.g} {args.xt} {args.yt} {args.zt} {args.xt0} > fit_results.txt'
    os.system(bashCmd)

    # Read and clean data
    with open('fit_results.txt','r', encoding='utf8') as f:
        fit_results = f.read()
    # print(fit_results)

    # put fit results into two tables

    ## Plane
    fit_results_plane_df = pd.read_csv(StringIO(fit_results.split("=")[0].strip()), sep=r'\s+', names=['fid', 'ftype', 'distance','x1','y1', 'z1'], skiprows=1)
    fit_results_plane_df.insert(loc=0, column='point', value=fit_results_plane_df.index)
    # Cylinder
    fit_results_cylinder_df = pd.read_csv(StringIO(fit_results.split("=")[-1].strip()), sep=r'\s+', names=['fid', 'ftype', 'distance','x1','y1', 'z1', 'd_nx',  'd_ny', 'd_nz', 'd'], skiprows=1)
    fit_results_cylinder_df.insert(loc=0, column='point', value=fit_results_cylinder_df.index)

    return fit_results_plane_df, fit_results_cylinder_df
    
def plotResults(fit_results_plane_df, fit_results_cylinder_df, figName=""):

    # ------ draw residuals of CMM points
    fig_residuals_plane = sns.relplot(x='point', y='distance', hue='fid', style='ftype', palette='tab10', data=fit_results_plane_df)
    # fig_residuals_plane.set_titles("")
    fig_residuals_plane.set_ylabels("residuals (mm)", clear_inner=False)
    fig_residuals_plane.set_xlabels("Point ID", clear_inner=False)
    fig_residuals_plane.savefig(f'residuals_plane{figName}.pdf')
    fit_results_plane_df['fid'] = fit_results_plane_df['fid'].astype(str)
    fig_residuals_plane2 = px.scatter(fit_results_plane_df, x="point", y="distance", color="fid", facet_col="ftype", hover_data=['distance'],
    labels={"point": "Point ID", "distance": "residuals (mm)", "fid": "face ID", "ftype": "face type"},
    title="")
    # , symbol="ftype"
    # fig_residuals_plane2.update_xaxes(rangeslider_visible=True)
    fig_residuals_plane2.update_yaxes(automargin=True)
    if (1 in fit_results_plane_df['ftype']): fig_residuals_plane2.layout.annotations[0]['text'] = 'Staves'
    if (2 in fit_results_plane_df['ftype']): fig_residuals_plane2.layout.annotations[1]['text'] = 'Pins'
    fig_residuals_plane2.update_layout(autosize=False, width=1050, height=700)
    # fig_residuals_plane2.show()
    fig_residuals_plane2.write_html(f'residuals_plane{figName}.html')
    fit_results_plane_df['fid'] = fit_results_plane_df['fid'].astype(int)
        
    # ------ draw stave model planes and compare to CMM points z-position
    
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
    ax_yz.set_xlabel('Y (mm)')
    ax_yz.set_ylabel('Z (mm)')
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
        coords = dfs[i][['d_nx', 'd_ny']].to_numpy()
        xc, yc, r, s = cf.least_squares_circle(coords)
        x_center[i] = xc
        y_center[i] = yc
        center[i] = sqrt(xc*xc + yc*yc)
        cmmOffCircle = np.empty(len(dfs[i]), dtype=float)

        for j in range(0, len(dfs[i])):
            dnx = dfs[i].iloc[j]['d_nx']
            dny = dfs[i].iloc[j]['d_ny']
            cmmOffCircle[j] = 1000 * (r - sqrt((xc-dnx)**2 + (yc-dny)**2))

        theta_fit = np.linspace(-pi, pi, 180)
        x_fit = xc + r*np.cos(theta_fit)
        y_fit = yc + r*np.sin(theta_fit)

        fig_dxdy, axs = plt.subplots(1, 2, figsize=(16,7))
        sns.scatterplot(data=dfs[i], x='d_nx', y='d_ny', hue='fid', palette='dark', ax=axs[0])
        axs[0].plot(x_fit, y_fit, 'r-' , label="fitted circle", lw=2)
        axs[0].plot([xc], [yc], 'bD', mec='y', mew=1)
        axs[0].set_title('Circle Fit')
        axs[0].set(xlabel='X (mm)', ylabel='Y (mm)')
        axs[0].legend(loc='best',labelspacing=0.1)
        axs[1].plot(np.arange(0, len(dfs[i]), 1), cmmOffCircle, 's')
        axs[1].set_title('CMM deviation from Circle')
        axs[1].set(xlabel='Points', ylabel=r'Deviation ($\mu$m)')
        fig_dxdy.savefig(f'fig_dxdy_{i}{figName}.pdf')

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
    fig_center.savefig(f'fig_center{figName}.pdf')

    dy_gap = np.array([abs(y_center[i] - y_center[i+1])*1000 for i in range(len(dfs)-1)])
    fig_dy_gap = plt.figure()
    plt.xlabel("pair Points ID") 
    plt.ylabel(r'Y position gap ($\mu$m)')
    plt.plot(np.arange(0, len(dfs)-1, 1), dy_gap, "ob")
    plt.axhline(y=40, color='r', linestyle='-')
    fig_dy_gap.savefig(f'fig_dy_gap{figName}.pdf')

    return fig_residuals_plane, fig_yz, fig_center

fit_results_plane_df, fit_results_cylinder_df = cmmCleanFit(args.fidList, args.pointList, args.sigmaList)
plotResults(fit_results_plane_df, fit_results_cylinder_df, "")
