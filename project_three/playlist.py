import numpy as np
from collections import defaultdict
from artist import Artist
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

class Playlist:
    def __init__(self, trialCount, num_intervals, num_rounds, num_players):
        self.k = num_intervals
        self.n = num_rounds
        self.p = num_players
        self.trials = trialCount
        self.init_lr()
        self.init_clickp()
        self.init_players()
        self.max_dict = defaultdict(float)
        self.avg_dict = defaultdict(float)
        self.best_bid = defaultdict(float)
        self.regrets = defaultdict(float)
        self.bid_val_ratio = defaultdict(float)

    def init_clickp(self):
        self.w = np.zeros(self.p)
        sum_w = 0
        for i in range(self.p):
                self.w[i] = np.random.uniform(0,1)
                sum_w += self.w[i]
        self.w /= sum_w
        self.w.sort()
        # self.w = [.195, .196, .197, .198, .199, .20, .201, .202, .203, .204, .205]
        self.w = self.w[::-1]

	
    def init_lr(self):
        self.lr = []
        # learning_rates.append(np.sqrt(np.log(self.k) / self.n))
        learning_rates_range = np.linspace(0,2,self.p)
        # learning_rates_range = [0.2 for i in range(self.p)]
        # learning_rates_range[-1] = 5
        for i in learning_rates_range:
            self.lr.append(i)

    def init_players(self):
        self.values = np.random.randint(50, size=self.p)
        # self.values.sort()
        self.players = []
        for i in range(self.p):
            new_player = Artist(i, self.lr[i], self.values[i]+1, self.k)
            self.players.append(new_player)

    def do_auction(self):
        for round in range(self.n):
            bids = []
            for i, player in enumerate(self.players):
                bids.append((player.make_decision(), i))
            bids.sort(reverse=True, key=lambda x: x[0])

            # append 0 for last bidder
            bids.append((0,-1))
            for i, (bid, key) in enumerate(bids[:-1]):
                payoff = (self.players[key].val - bids[i+1][0]) * self.w[i]
                arr= []
                for pair in bids:
                    if pair != (bid,key):
                        arr.append(pair)
                self.players[key].update_payoff(payoff,arr, self.w)

        for i,p in enumerate(self.players):
            # print("Player ",i)
            # print("Learning rate: ", p.lr)
            # print("Value: ", p.val)
            # print("Best bid: ", p.get_best_bid())
            # print("Max Payoff: ", max(p.payoffs))
            # maxPayoff = max(p.payoffs)
            # self.max_dict[p.lr] += maxPayoff / self.trials
            # self.avg_dict[p.lr] += (sum(p.payoffs)/self.k) / self.trials 
            # self.best_bid[p.lr] = p.get_best_bid()

            self.regrets[p.lr] += p.get_regret() / self.n / self.trials
            # self.regrets[p.id] += p.get_regret() / self.n / self.trials
            
            self.bid_val_ratio[p.lr] += p.get_best_bid_to_value_ratio() / self.trials

    def run_experiments(self):
        for i in range(self.trials):
            self.init_players()
            self.do_auction()
                
        # print("REGRET DICT:", self.regrets)
        # print("MAX DICT", self.max_dict)
        #print("BEST BID", self.best_bid)

    def please_plot(self):
        regrets=self.regrets
        x = np.array([i for i in regrets if regrets[i]!=regrets.default_factory()])
        y = np.array([regrets[i] for i in regrets if regrets[i]!=regrets.default_factory()])
        # myLocator = mticker.MultipleLocator(4)
        # plt.xaxis.set_major_locator(myLocator)
        plt.plot(x,y)
        plt.xlabel("Epsilon Values (Learning Rates)")
        # plt.xlabel("Player ID")
        plt.ylabel("Average Regrets")
        
        plt.xticks(rotation = 45)
        plt.xticks(fontsize= 5)
        # plt.show()
        plt.savefig("LR vs Regret")
        plt.close()

        regrets=self.bid_val_ratio
        x = np.array([i for i in regrets if regrets[i]!=regrets.default_factory()])
        y = np.array([regrets[i] for i in regrets if regrets[i]!=regrets.default_factory()])
        # myLocator = mticker.MultipleLocator(4)
        # plt.xaxis.set_major_locator(myLocator)
        plt.plot(x,y)
        plt.xlabel("Epsilon Values (Learning Rates)")
        # plt.xlabel("Player ID")
        plt.ylabel("Bid to Val ratio")
        
        plt.xticks(rotation = 45)
        plt.xticks(fontsize= 5)
        # plt.show()
        plt.savefig("Bid to val vs Regret")
        plt.close()
		

spotify = Playlist(100, 40, 100, 20)

spotify.run_experiments()

spotify.please_plot()