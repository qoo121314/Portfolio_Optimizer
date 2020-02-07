# Portfolio_Optimizer
By Markowitz Model


### Step1. Define the stock code list for downloading stock prices from Yahoo Finance.

  Portfolio.get_price_table(code_list, start='2015-01-01', end='2019-12-31')

### Step2. Set the parameters for optimizing.

  Portfolio.set_optimize(self, begin=None , end=None, rf=0.05)

### Step3. Get the optimized weights.

  Get_Best_Portfolio(self, hedge=None, bothside=None)
  '''
  '''
  return (Sharpe Ratio, Weights)

### Step4. Reset the weights by optimizing result.

  set_weights(self, weights=Weights)

### Step5. Plot and summarize the optimal portfolio.
  
  Return_Plot(self)
  
  Summary(self)
