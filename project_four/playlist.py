import numpy as np
from collections import defaultdict
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

class Auction:
        def __init__(self, trialCount, num_intervals, num_rounds, num_players, price_pay):
                self.k = num_intervals
                self.n = num_rounds
                self.p = num_players
                self.trials = trialCount
                self.regrets = defaultdict(float)
                self.payoffs = defaultdict(float)
                self.ind = price_pay
                # self.lr = np.sqrt(np.log(self.k)/self.n)
                self.lr=1
                self.price_pay=price_pay
                print("this is lr",self.lr)
                self.average_rp = 0
                self.average_rev = 0
                self.alg_rev = 0
                self.max_value = 10
        
        def init_reserves(self):
                self.reserves = np.linspace(0,self.max_value,self.k)
                #print(self.reserves)
        def reset_payoffs(self):
                self.rev = 0
                for i in self.reserves:
                        self.payoffs[i]=0
        def init_players(self):
                self.values = np.zeros(self.p)
                for i in range(self.p):
                        # x = np.random.uniform(0,1000)
                        # self.values[i]=np.cbrt(x)
                        self.values[i] = np.random.uniform(0,self.max_value)
                # print("THIS IS VALUES",self.values)
        def pick_reserve_price(self):
                prob_list = np.zeros(self.k)
                total_prob = 0
                
                for i, payoff in enumerate(self.payoffs):
                        action_prob = np.power((1 + self.lr), payoff / self.max_value)
                        prob_list[i] = action_prob
                        total_prob += action_prob

                prob_list = prob_list / total_prob

                self.chosen_action = np.random.choice(self.k, 1, p=prob_list)[0]
                return self.reserves[self.chosen_action]

        def do_auction(self):
                self.init_reserves()
                self.reset_payoffs()
                for round in range(self.n):
                        self.init_players()
                        bids = []
                        for i in range(self.p):
                                bids.append(self.values[i])
                        #if bids are all lower than reserve price, then 
                        bids.sort(reverse=True)
                        actual_reserve_price = self.pick_reserve_price()
                        for i in range(self.k):
                                
                                reserve_price  = self.reserves[i]
                                cur_rev = 0
                                
                                if bids[self.price_pay-2] < reserve_price:
                                        continue
                                else:
                                        if bids[self.price_pay-1] < reserve_price:
                                                self.payoffs[reserve_price] += reserve_price
                                                cur_rev = reserve_price
                                        else:
                                                self.payoffs[reserve_price] += bids[1]
                                                cur_rev = bids[1]

                                if actual_reserve_price == reserve_price:
                                        self.rev += cur_rev
                arg=0
                val=0
                for i,v in self.payoffs.items():
                        if val<v:
                                arg=i
                                val=v
                # print("reserve price: ",arg/self.n)
                # print("revenue: ",val/self.n)
                self.average_rp += arg / self.trials
                self.average_rev += val / self.trials / self.n
                self.alg_rev += self.rev / self.trials / self.n
                
                # for i,p in enumerate:
                #         self.regrets[actual_reserve_price] += p.get_regret() / self.n / self.trials
                        

        def run_experiments(self):
                for i in range(self.trials):
                        self.do_auction()
                print('regret:',self.average_rev - self.alg_rev)
                print((self.average_rev - self.alg_rev)/self.average_rev * 100)
                        
                # print("REGRET DICT:", self.regrets)
                # print("MAX DICT", self.max_dict)
                #print("BEST BID", self.best_bid)
        def please_plot(self):
                regrets=self.payoffs
                x = np.array([i for i in regrets if regrets[i]!=regrets.default_factory()])
                y = np.array([regrets[i]/self.n for i in regrets if regrets[i]!=regrets.default_factory()])
                # myLocator = mticker.MultipleLocator(4)
                # plt.xaxis.set_major_locator(myLocator)
                plt.plot(x,y)
                plt.xlabel("Reserve Prices")
                plt.ylabel("Average Revenue")
                plt.title("10 bidders and 5th price auction")
                
                # plt.show()
                plt.xticks(rotation = 45)
                plt.xticks(fontsize= 5)

                plt.savefig("10 bidders and 5th price auction")
                plt.close()


spotify = Auction(100, 50, 1000, 10, 4)

spotify.run_experiments()

spotify.please_plot()

print('reserve price:',spotify.average_rp)
print(spotify.average_rev)
print(spotify.alg_rev)
