import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class EW:
    def __init__(self, num_actions, num_rounds, strategy):
        self.k = num_actions
        self.n = num_rounds
        self.strategy = strategy
        self.learning_rates = self.initialize_learning_rates()
        self.cur_payoff = np.zeros(self.k)
        self.payoffs = np.zeros(self.k)
        self.stock_payoffs = None
        self.bpl = np.zeros(self.k)
        self.alg = 0
        self.h = 1
        self.regrets = defaultdict(float)

    def initialize_learning_rates(self):
        # Initialize learning rates with three learning rates
        learning_rates = []
        learning_rates.append(np.sqrt(np.log(self.k) / self.n))
        # learning_rates_range = np.linspace(0,1,50)
        # for i in learning_rates_range:
        #     learning_rates.append(i)
        for i in [0,1]:
            learning_rates.append(i)
        learning_rates.sort()
        return learning_rates

    def setup_stock(self):
        my_data = np.genfromtxt('mastersheet.csv', delimiter=',')
        payoffs=[]
        for i in range(100):
            payoffs.append([])
        for i in range(100):
            for j in range(4):
                payoffs[i].append(my_data[i][2*j]/ my_data[i][2*j+1])
        self.stock_payoffs = payoffs

    def run_simulation(self):
        if self.strategy == "Bernoulli":
            self.bern_set_prob()
        if self.strategy == "Stock":
            self.setup_stock()
        for lr in self.learning_rates:
            # print("Learning Rate ", lr)
            self.alg = 0
            self.reset_payoffs() 
            for round in range(self.n):                
                # Choosing action based on EW algorithm
                prob_list = np.zeros(self.k)
                total_prob = 0
                print("round ",round)
                for i, payoff in enumerate(self.payoffs):
                    action_prob = np.power((1 + lr), payoff / self.h)
                    prob_list[i] = action_prob
                    total_prob += action_prob

                new_prob_list = prob_list / total_prob
                print(new_prob_list)

                chosen_action_idx = np.random.choice(self.k, 1, p=new_prob_list)

                # update payoff based on stategy
                if self.strategy == "Adversarial":
                    self.afp()
                if self.strategy == "Bernoulli":
                    self.bp()
                if self.strategy == "Stock":
                    self.stock(round)
                # print("current payoff ", self.cur_payoff)
                # print("payoffs", self.payoffs)

                # Calculate algorithmic payoff
                self.alg += self.cur_payoff[chosen_action_idx]

            # Calculate optimal payoff
            opt = max(self.payoffs)
            # print("Payoffs at the end: ", self.payoffs)
            print("OPT: ", opt)
            print("Alg payoff", self.alg)

            # Calculate Regret
            regret = (opt - self.alg) / self.n
            self.regrets[str(lr)] = regret

    def afp(self):
        # Assign payoff from a uniform distribution U[0, 1]
        # to the action with the smallest payoff
        payoff = np.random.uniform()
        lowest_payoff_index = np.argmin(self.payoffs)
        self.cur_payoff = np.zeros(self.k)
        self.cur_payoff[lowest_payoff_index] = payoff
        self.payoffs[lowest_payoff_index] += payoff

    def bern_set_prob(self):
        # Initialize probability for each action before running simulation
        # Payoff 0 with probability 1-p_i and 1 with probability p_i
        self.bpl = np.random.uniform(0, 1/2, self.k)
        # print("Probility list for Bernoulli", self.bpl)
        
    def bp(self):
        for i in range(self.k):
            p_i = self.bpl[i]
            self.cur_payoff[i] = np.random.choice(2, p = [1-p_i, p_i])
            self.payoffs[i] += self.cur_payoff[i]
    
    def stock(self, round):
        self.cur_payoff = self.stock_payoffs[round]
        for i in range(self.k):
            self.payoffs[i] += self.cur_payoff[i]

    def reset_payoffs(self):
        self.cur_payoff = np.zeros(self.k)
        self.payoffs = np.zeros(self.k)
        
            
    
# Monte carlo trials

N = 1
regrets_ad = defaultdict(float)
regrets_bn = defaultdict(float)
regrets_stocks = defaultdict(float)

def please_plot(regrets,strategy):
    # print(regrets)
    x = np.array([i for i in regrets if regrets[i]!=regrets.default_factory()])
    y = np.array([regrets[i] for i in regrets if regrets[i]!=regrets.default_factory()])
    
    plt.plot(x,y)
    plt.xlabel("Epsilon Values (Learning Rates)")
    plt.ylabel("Average Regrets")
    plt.savefig(str(strategy))
    plt.show()
    plt.close()

for i in range(N):

    # print('adversarial')
    ew = EW(5, 100, "Adversarial")
    ew.run_simulation()
    print(ew.regrets.items())
    for k,v in ew.regrets.items():
        regrets_ad[k] += v/N    
    

    # print('bernoulli')
    # ew = EW(5, 100, "Bernoulli")
    # ew.run_simulation()
    # for k,v in ew.regrets.items():
    #     regrets_bn[k] += v/N
    
    # ew = EW(4, 100, "Stock")
    # ew.run_simulation()
    # for k,v in ew.regrets.items():
    #     regrets_stocks[k] += v/N

# print(regrets_ad)
# print(regrets_stocks)
# print(regrets_bn)



please_plot(regrets_ad, "Adversarial")
# please_plot(regrets_bn,"Bernoulli")
# please_plot(regrets_stocks,"Stocks")