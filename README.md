# Portfolio_Optimizer
By Markowitz Model


### Step1. Define the stock code list for downloading stock prices from Yahoo Finance.

　　Portfolio.get_price_table(code_list, start='2015-01-01', end='2019-12-31')

### Step2. Set the parameters for optimizing.

　　Portfolio.set_optimize(self, begin=None , end=None, rf=0.05)

### Step3. Get the optimized weights.

　　Portfolio.Get_Best_Portfolio(self, hedge=None, bothside=None)

　　Portfolio.return (Sharpe Ratio, Weights)

### Step4. Reset the weights by optimizing result.

　　Portfolio.set_weights(self, weights=Weights)

### Step5. Plot and summarize the optimal portfolio.
  
　　Portfolio.Return_Plot(self)
  
　　Portfolio.Summary(self)
