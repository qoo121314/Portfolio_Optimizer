import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import itertools



sns.set_style('darkgrid')

class Portfolio():
    @staticmethod
    def get_price_table(code_list, start='2015-01-01', end='2019-12-31'):
        '''put list of stock code here
        '''
        temp=[]
        adj_close_data=[]

        for i in code_list:
            temp.append(web.DataReader(name='{}'.format(i), data_source='yahoo', start=start, end=end))
            print('Finding  {}'.format(i))

        for j in temp:
            adj_close_data.append( j ['Adj Close'])
        close_df = pd.concat(adj_close_data, axis=1, keys=code_list, join='inner')

        return close_df
    
    def __init__(self, data):
        self.price_data = data
        self.start_day = data.index[0]
        self.end_day = data.index[-1]
        self.daily_return = np.log (data /data.shift(1))
        self.weights = np.ones(len(data.columns)) / len(data.columns)
        self.portfolio_return = (self.daily_return * self.weights).sum(axis=1)
        self.portfolio_cumulative_return  = np.exp((self.daily_return * self.weights).sum(axis=1).cumsum())
    
    def __repr__(self):
        return str (self.price_data.columns)
    
    def price_corr_map(self):
        sns.heatmap(self.price_data.corr(), cmap='coolwarm', annot=True)
        
    def return_corr_map(self):
        sns.heatmap(self.daily_return.corr(), cmap='coolwarm', annot=True)
    
    def Return_Plot(self):
        
        Portfolio_Return =(self.daily_return * self.weights).sum(axis=1).cumsum()

        Cum_Trade_Percent_Return=np.exp(Portfolio_Return)
        MDD_Series = Cum_Trade_Percent_Return.cummax()-Cum_Trade_Percent_Return
        High_index = Cum_Trade_Percent_Return[Cum_Trade_Percent_Return.cummax()==Cum_Trade_Percent_Return].index

        fig, ax = plt.subplots(2, 1, figsize=[15,6], gridspec_kw={'height_ratios': [3, 1]})
        sns.set_style('darkgrid')

        sns.lineplot(x=Cum_Trade_Percent_Return.index ,y=Cum_Trade_Percent_Return, ax=ax[0], color='0.38')
        sns.scatterplot(x=High_index, y=Cum_Trade_Percent_Return.loc[High_index], color='#02ff0f', ax=ax[0])
        sns.lineplot(x=MDD_Series.index ,y=-MDD_Series, ax=ax[1], color='r')
        ax[1].fill_between(x=MDD_Series.index ,y1=-MDD_Series, color='r')

        ax[0].set_xlabel('')
        ax[0].set_ylabel('Return%')
        ax[1].set_yticklabels(['{:,.1%}'.format(y) for y in ax[1].get_yticks()])

        plt.suptitle('Return & MDD',fontsize=16)
        plt.show()
        
    def Get_Average_Return(self, begin=None, end=None, weight=None):

        if begin==None:
            start_day=self.price_data.index[0]
        else:
            start_day = pd.Timestamp(begin)

        if end==None:
            end_day=self.price_data.index[-1]
        else:
            end_day = pd.Timestamp(end)

        Return = self.daily_return

        if weight == None:
            Average_Return = Return * self.weights
            Average_Return = Average_Return.sum(axis=1).iloc[1:]
        else:
            Average_Return = Return * weight
            Average_Return = Average_Return.sum(axis=1).iloc[1:]

        return Average_Return.loc[start_day : end_day].mean() * 252

    def Get_Average_SD(self, begin=None, end=None, weight=None):

        if begin==None:
            start_day=self.price_data.index[0]
        else:
            start_day = pd.Timestamp(begin)

        if end==None:
            end_day=self.price_data.index[-1]
        else:
            end_day = pd.Timestamp(end)

        Return = self.daily_return

        if weight == None:
            Average_Return = Return * self.weights
            Average_Return = Average_Return.sum(axis=1).iloc[1:]
        else:
            Average_Return = Return * weight
            Average_Return = Average_Return.sum(axis=1).iloc[1:]

        return Average_Return.loc[start_day : end_day].std() * np.sqrt(252)

    def Get_Sharpe_Ratio(self, begin=None, end=None, weight=None, rf=0.05):

        if begin==None:
            start_day=self.price_data.index[0]
        else:
            start_day = pd.Timestamp(begin)

        if end==None:
            end_day=self.price_data.index[-1]
        else:
            end_day = pd.Timestamp(end)

        Return = self.daily_return

        if weight == None:
            Average_Return = Return * self.weights
            Average_Return = Average_Return.sum(axis=1).iloc[1:]
        else:
            Average_Return = Return * weight
            Average_Return = Average_Return.sum(axis=1).iloc[1:]

        Sharpe_Ratio = (Average_Return.loc[start_day : end_day].mean() - (rf / 252) ) * 252 / (Average_Return.loc[start_day : end_day].std() * np.sqrt(252))

        return Sharpe_Ratio
    
    def Get_Sotino_Ratio(self, begin=None, end=None, weight=None, rf=0.05, threshold=None):

        if begin==None:
            start_day=self.price_data.index[0]
        else:
            start_day = pd.Timestamp(begin)

        if end==None:
            end_day=self.price_data.index[-1]
        else:
            end_day = pd.Timestamp(end)

        Return = self.daily_return

        if weight == None:
            Average_Return = Return * self.weights
            Average_Return = Average_Return.sum(axis=1).iloc[1:]
        else:
            Average_Return = Return * weight
            Average_Return = Average_Return.sum(axis=1).iloc[1:]
        
        Average_Return= Average_Return.loc[start_day : end_day]
        
        if threshold==None:
            threshold = Average_Return.mean()

        DownTable = Average_Return[Average_Return < threshold].apply(lambda x : (x - threshold) ** 2)
        DownRisk = np.sqrt(DownTable.sum() / Average_Return.count())
        Sotino_Ratio = (Average_Return.mean() - (rf / 252) ) * 252 / (DownRisk * np.sqrt(252))

        return Sotino_Ratio
    
    def Get_MDD(self):
        
        Portfolio_Return =(self.daily_return * self.weights).sum(axis=1).cumsum()

        Cum_Trade_Percent_Return=np.exp(Portfolio_Return)
        MDD_Series = Cum_Trade_Percent_Return.cummax()-Cum_Trade_Percent_Return
        
        return MDD_Series.max()
    
    def Summary(self):
        print('{:^50}'.format('Period'))
        print('From {}  to {}, {} days.'.format(str(self.start_day.date()), str(self.end_day.date()), (self.end_day-self.start_day).days  ))
        print('-'*60)
        print('{:^50}'.format('Weights of Portfolio:'))
        print('-'*60)
        for i, j in zip(self.price_data.columns, self.weights):
            j = round(j,5)
            print('{:<30s}    {:>12.2%}'.format(i , j) )
        print('-'*60)
        print('\n')
        print('{:^50}'.format('Technical Indicator:'))
        print('-'*60)
        print('Average Return : {:>55.3f}'.format(self.Get_Average_Return()))
        print('Average Standard Deviation : {:>31.3f}'.format(self.Get_Average_SD()))
        print('Sharpe Ratio : {:>60.3f}'.format(self.Get_Sharpe_Ratio()))
        print('Sotino Ratio : {:>61.3f}'.format(self.Get_Sotino_Ratio()))
        print('Maximum Drop Down : {:>42.3f}'.format(self.Get_MDD()))
        print('-'*60)

        
    def set_optimize(self, begin=None , end=None, rf=0.05):
        if begin==None:
            self.__opt_begin = self.start_day
        else:
            self.__opt_begin = pd.Timestamp(begin)
            
        if end==None:
            self.__opt_end = self.end_day
        else:
            self.__opt_end = pd.Timestamp(end)        

        self.__rf = rf

    
    def optimize_set(self):
        print('Begin :  {}'.format(self.__opt_begin))
        print('End :  {}'.format(self.__opt_end))
        print('Rf :  {}'.format(self.__rf))

    def __sharpe_target(self, weights):
        
        temp = self.daily_return * weights
        
        temp = temp.sum(axis=1)
        Average_Return = temp.iloc[1:]
        Sharpe_Ratio = (Average_Return.loc[self.__opt_begin : self.__opt_end].mean() - (self.__rf / 252) ) * 252 / (Average_Return.loc[self.__opt_begin : self.__opt_end].std() * np.sqrt(252))

        return  - Sharpe_Ratio
    
    def __mdd_target(self, weights):
        
        
        Portfolio_Return =(self.daily_return * weights).sum(axis=1).cumsum()

        Cum_Trade_Percent_Return=np.exp(Portfolio_Return)
        MDD_Series = Cum_Trade_Percent_Return.cummax()-Cum_Trade_Percent_Return
        
        return  MDD_Series.max()

    def __check_sum_positive(self, weights):
        #return 0 if sum of the weights is 1
        return np.sum(weights)-1

    def __check_sum_hedge(self, weights):
        #return 0 if sum of the weights is 1
        return np.sum(weights)

    def Get_Best_Portfolio(self, hedge=None, bothside=None, method='sharpe'):
        Return_Table = self.daily_return

        bounds=[]

        if hedge == True:
            for i in itertools.repeat( (-1,1) ,len(self.daily_return.columns)):
                bounds.append(i)
            bounds=tuple(bounds)
            cons=({'type':'eq', 'fun': self.__check_sum_hedge})

        else:
            for i in itertools.repeat( (0,1) ,len(self.daily_return.columns)):
                bounds.append(i)
            bounds=tuple(bounds)
            cons=({'type':'eq', 'fun':self.__check_sum_positive})

        init_guess= []
        for i in itertools.repeat( (1/len(Return_Table.columns)) ,len(Return_Table.columns)):
            init_guess.append(i)
            
        if method == 'sharpe':
            opt_results= minimize(self.__sharpe_target, init_guess, method='SLSQP', bounds= bounds, constraints=cons)
            return -opt_results['fun'], list(opt_results['x'])
        
        elif method=='mdd':
            opt_results= minimize(self.__mdd_target, init_guess, method='SLSQP', bounds= bounds, constraints=cons)
            return opt_results['fun'], list(opt_results['x'])

    
    def set_weights(self, weights=None):
        if weights == None:
            self.weights = np.ones(len(self.price_data.columns)) / len(self.price_data.columns)
        else:
            self.weights = weights
        
    def Get_Monte_Carlo_Forecast(self, capital=1 , path=1000, period=3, yearly=None):
        
        if yearly==True:
            s=np.repeat(capital, path)
            ret = self.Get_Average_Return()
            vol = self.Get_Average_SD()
            
            for i in range(period):
                b = np.random.normal(0, 1, path)
                s = s + s * (ret + b * vol)
        
        else:
            s=np.repeat(capital, path)
            ret = self.Get_Average_Return()
            vol = self.Get_Average_SD()
            
            for i in range(period*252):
                b=np.random.normal(0, 1, path)
                s = s + s * (ret / 252 + b * vol / np.sqrt(252))          
        
        
        res = s
        res.sort()
        sns.set_style('darkgrid')
        
        if capital==1:
            print('Minimum : {0:21.3%} '.format( np.percentile(res,0)  )   )
            print('5   Percentile : {0:15.3%} '.format( np.percentile(res,5)  )   )
            print('25 Percentile : {0:15.3%} '.format( np.percentile(res,25)  )   )
            print('Median : {0:25.3%} '.format( np.percentile(res,50)  )   )
            print('Mean : {0:28.3%} '.format( res.mean()  )   )
            print('75 Percentile : {0:15.3%} '.format( np.percentile(res,75)  )   )
            print('Maximum : {0:20.3%} '.format( np.percentile(res,100)  )   )
            sns.distplot(res,axlabel ='Expected Earnings (%)',label ='ya')
                    
        else:
            print('Minimum : {0:21.3f} '.format( np.percentile(res,0)  )   )
            print('5   Percentile : {0:15.3f} '.format( np.percentile(res,5)  )   )
            print('25 Percentile : {0:15.3f} '.format( np.percentile(res,25)  )   )
            print('Median : {0:25.3f} '.format( np.percentile(res,50)  )   )
            print('Mean : {0:28.3f} '.format( res.mean()  )   )
            print('75 Percentile : {0:15.3f} '.format( np.percentile(res,75)  )   )
            print('Maximum : {0:20.3f} '.format( np.percentile(res,100)  )   )
            sns.distplot(res,axlabel ='Expected Earnings (USD)',label ='ya')
        
        
        plt.yticks(visible=False)
        plt.figure(figsize=(16,8))
        plt.show()
    
