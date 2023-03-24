# reinforcement-learning-trading-commodity-futures
In this implementation, I created a streamline data analysis, data prepare, reinforcement learning agent training and testing example. The commodity future contract used is Rebar from Shanghai Futures Exchange(https://www.shfe.com.cn/en/products/SteelRebar/). 


Firstly, download training data from tqsdk(a market data provider, https://www.shinnytech.com/tianqin/).

Secondly, use the "Financial Indicator Analysis.ipynb" in the "Data Analysis" folder to find out the most useful indicators offered by Ta-lib.

Thirdly, since futures contracts are contract-based, this means that one only like to trade the most active contract for the given trade day. After the Ta-lib indicators are determined, I used another quantitative trading softwarwe provider "goldminer" (ttps://www.myquant.cn/) to pre-process the training and testing data into individual gym observation windows from each trade day's most active futures contract(the most active futures contract tend to change every 2.5-3 month, so the consistency is not a huge problem). Then, the data is save as *.npy files in "Processed Data" folder.

Finally, use the "Model Training(A2C).ipynb" to train a reinforcement learning agent and test it using the "Model Testing.ipynb" to test the saved model file in "best_models" folder.
