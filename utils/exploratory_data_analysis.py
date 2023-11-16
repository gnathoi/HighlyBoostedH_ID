import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import mplhep as hep
import matplotlib
matplotlib.use('Agg')  # must be called before importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

#data_path = './training_dataset.pkl' # directory where data is stored
#output = './outputs/'

# Change the overall theme, context and palette
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.6)
sns.set_palette("pastel")

class EDA():
    def __init__(self,data_path,output):
        self.df = pd.read_pickle(data_path)
        self.output = output
        # we define which featires are categoric and which are continuous
        self.categ_var = ['Eta_Bin','pT_Bin','FatElectron_nConstituents','FatElectron_ntrks']
        self.cont_var = [col for col in self.df.columns if col not in self.categ_var and col != 'type' and col != 'weight_ey_2d']
        self.figsize = (10,15)
        self.labels = [r'$\rm H \rightarrow \gamma\gamma $',r'$\rm Z \rightarrow e^+e^- $',r'$\rm q/g$', r'$\rm e/\gamma $', r'$\rm (\tau)\tau$']
        self.colors = ['#4462a5', '#689aff', 'tab:orange', '#c50807', '#68cd67']

    def bar(self):
        type_counts = self.df['type'].value_counts()
        print("Number of instances for each type:")
        print(type_counts)

        # bar plot for categorical features
        plt.style.use([hep.style.ATLAS])
        fig = plt.figure(figsize=(20,16))
        ax1 = plt.subplot2grid((3, 2), (0, 0))
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        axes = [ax1, ax2, ax3, ax4]

        color_dict = dict(zip(range(len(self.labels)), self.colors))
        xlabels = [r'$\rm |\eta|$ Bin',r'$\rm p_T$ Bin',r'$\rm N^{Const.}$',r'$\rm N^{Trks}$']

        for i, col in tqdm(enumerate(self.categ_var), total=len(self.categ_var), desc='Generating bar plots'):
            sns.countplot(data=self.df, x=col, hue='type', ax=axes[i], palette=color_dict)
            #cleaned_label = col.replace('FatElectron_', '')
            axes[i].set_xlabel(xlabels[i], fontsize=30)
            axes[i].set_ylabel('Count', fontsize=30)
            axes[i].tick_params(axis='x', labelsize=24)
            axes[i].tick_params(axis='y', labelsize=24)
            axes[i].legend_.remove()
            axes[i].legend(labels=self.labels, fontsize=20)
            axes[i].set_yscale('log')

        axes[3].tick_params(axis='x', labelsize=12.5)
        axes[0].set_ylim(0.1, 1.2e12)
        axes[1].set_ylim(0.1, 1.2e12)
        axes[1].legend(labels=self.labels, fontsize=20, loc='upper left')

        plt.tight_layout()
        output_path = os.path.join(self.output, 'barplot.pdf')
        plt.savefig(output_path,bbox_inches='tight')

        #print('Bar plots generated and saved!')

    def enlarged_bar_plot(self):
        plt.style.use([hep.style.ATLAS])
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(21, 30))  # One row, two columns, very tall

        color_dict = dict(zip(range(len(self.labels)), self.colors))
        xlabels = [r'$\rm N^{Const.}$',r'$\rm N^{Trks}$']

        for i, col in tqdm(enumerate(self.categ_var[2:]), total=2, desc='Generating ax3 and ax4 plots'):  # Looping only over the columns for ax3 and ax4
            sns.countplot(data=self.df, y=col, hue='type', ax=[ax3, ax4][i], palette=color_dict, orient='h')  # Horizontal bars

            [ax3, ax4][i].set_ylabel(xlabels[i], fontsize=32)
            [ax3, ax4][i].set_xlabel('Count', fontsize=32)
            [ax3, ax4][i].tick_params(axis='x', labelsize=24)
            [ax3, ax4][i].tick_params(axis='y', labelsize=24)
            [ax3, ax4][i].legend_.remove()
            [ax3, ax4][i].legend(labels=self.labels, fontsize=24)
            [ax3, ax4][i].set_xscale('log')

        plt.tight_layout()
        output_path = os.path.join(self.output, 'enlarged_bar_plot.pdf')
        plt.savefig(output_path, bbox_inches='tight')

    def violin(self):
        # violin plots for continuous data
        ylabels = [r'$\rm \Delta R(t_1,j)$',r'$\rm \Delta R(t_2,j)$',r'$\rm r^{(\beta=1)}_{N=1}$',r'$\rm E/p$',r'$\rm f_{EM}$',
                    r'$\rm \max{(E_{Layer}/E_{Jet})}$','Planar Flow', 'Width', 'Balance', r'$\rm m$ [GeV]', r'$\rm \Delta R(t_1,t_2)$']
        plt.style.use([hep.style.ATLAS])
        fig, axs = plt.subplots(4, 3, figsize=self.figsize)
        axs = axs.flatten()
        color_dict = dict(zip(range(len(self.labels)), self.colors))
        for i, col in tqdm(enumerate(self.cont_var), total=len(self.cont_var), desc='Generating violin plots'):
            ax = axs[i]
            sns.violinplot(data=self.df, x='type', y=col, ax=ax, palette=color_dict)
            cleaned_label = col.replace('FatElectron_', '')
            ax.set_xlabel('Jet', fontsize=16)
            ax.xaxis.set_label_coords(1, -0.2)

            ax.set_ylabel(ylabels[i], fontsize=16)

            # Increase the fontsize for tick labels
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_xticklabels(self.labels,fontsize=14, rotation=90)

        # Remove any unused subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        output_path = os.path.join(self.output, 'violinplot.pdf')
        plt.savefig(output_path,bbox_inches='tight')#, pad_inches=0)
        #print('violin plots generated and saved!')

    def box(self):
        # box plots for continuous data
        fig, axs = plt.subplots(2, 6, figsize=self.figsize)
        axs = axs.flatten()

        for i, col in tqdm(enumerate(self.cont_var), total=len(self.cont_var), desc='Generating box plots'):
            sns.boxplot(data=self.df, x='type', y=col, ax=axs[i])

        # Remove any unused subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        output_path = os.path.join(self.output, 'boxplot.png')
        plt.savefig(output_path)

        #print('Box plots generated and saved!')

    def hist(self):
        # histograms for continuous data
        fig, axs = plt.subplots(2, 6, figsize=self.figsize)
        axs = axs.flatten()

        for i, col in tqdm(enumerate(self.cont_var), total=len(self.cont_var), desc='Generating histograms'):
            sns.histplot(data=self.df, x=col, hue='type', element='step', ax=axs[i], kde=True)

        # Remove any unused subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        output_path = os.path.join(self.output, 'histograms.png')
        plt.savefig(output_path)

        #print('Histograms generated and saved!')


    def comptomass(self):
        fig, axs = plt.subplots(2, 6, figsize=self.figsize)
        axs = axs.flatten()
        vars = self.cont_var
        vars.remove('FatElectron_cM')
        for i, col in tqdm(enumerate(vars), total=len(self.cont_var), desc='col vs mass scatterplots'):
            sns.scatterplot(data=self.df, x='FatElectron_cM', y=col, ax=axs[i],marker='x')

        # Remove any unused subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        output_path = os.path.join(self.output, 'comparison_Mass.png')
        plt.savefig(output_path)

        fig, axs = plt.subplots(2, 2, figsize=(20,16))
        axs = axs.flatten()
        for i, col in tqdm(enumerate(self.categ_var), total=len(self.categ_var), desc='col vs mass boxplots'):
            sns.boxplot(x=col, y='FatElectron_cM', data=self.df,ax=axs[i])

        # Remove any unused subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        output_path = os.path.join(self.output, 'comparison_Mass_box.png')
        plt.savefig(output_path)


    def corrplot(self):
        print('Calculating Spearman rank correlation coefficient...')
        df = self.df.drop(['type','weight_ey_2d'],axis=1)
        df.columns = [col.replace('FatElectron_', '') for col in df.columns]

        labels = [r'$\rm \Delta R(t_1,j)$',r'$\rm \Delta R(t_2,j)$',r'$\rm r^{(\beta=1)}_{N=1}$',r'$\rm E/p$',r'$\rm f_{EM}$',
                    r'$\rm \max{(E_{Layer}/E_{Jet})}$','Planar Flow', 'Width', 'Balance', r'$\rm m$', r'$\rm \Delta R(t_1,t_2)$',
                    r'$\rm |\eta|$ Bin',r'$\rm p_T$ Bin',r'$\rm N^{Const.}$',r'$\rm N^{Trks}$']

        spearman_corr = df.corr(method='spearman').round(2)
        plt.figure(figsize=(15, 12))
        #mask = np.tril(np.ones_like(spearman_corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        cmap.set_bad(color='white')
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool),k=1)
        sns.heatmap(spearman_corr, annot=True, vmin=-1, vmax=1, cmap=cmap,mask=mask,
                    annot_kws={'size':14},square=True,xticklabels=labels,yticklabels=labels,cbar_kws={'label':'Spearman rank correlation coefficient'})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        #plt.title("Spearman Rank Correlation Coefficient")
        output_path = os.path.join(self.output, 'corrplot.pdf')
        plt.savefig(output_path)
