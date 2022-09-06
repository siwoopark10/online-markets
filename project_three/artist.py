import numpy as np

class Artist:
    def __init__(self, id, learn_rate, value, intervals):
        self.id = id
        self.lr = learn_rate
        self.val = value
        self.k = intervals + 1
        self.payoffs = np.zeros(intervals + 1)
        self.chosen_action = None
        self.alg=0
    
	#Returns the bid using EW        
    def make_decision(self):
        # if self.id == 0:
        #     return 0
        prob_list = np.zeros(self.k)
        total_prob = 0
        for i, payoff in enumerate(self.payoffs):
            action_prob = np.power((1 + self.lr), payoff / self.val)
            if not action_prob:
                print(payoff," ", self.val)
                print("something is wrong")
            prob_list[i] = action_prob
            total_prob += action_prob

        prob_list = prob_list / total_prob
        for i in prob_list:
            if not i:
                print(prob_list)

        self.chosen_action = np.random.choice(self.k, 1, p=prob_list)[0]

        # we add 1 to ensure that 0 is never chosen
        return (self.chosen_action + 1) * self.val/(self.k)

    def update_payoff(self, payoff, bids, weights):
        #for each bid we need to calculate the hypothetical payoff
        for i in range(self.k):
            mybid= (i + 1) * self.val/(self.k)
            #insert my bid after before the first one im bigger than since it's reverse order
            for j in range(len(bids)):
                if bids[j][0] < mybid:
                    self.payoffs[i] += (self.val - bids[j][0]) * weights[j]
                    break
        self.alg += payoff

    def get_best_bid_to_value_ratio(self):
        return (np.argmax(self.payoffs) + 1) * self.val/(self.k) / self.val

    def get_opt(self):
        return max(self.payoffs)
    
    def get_alg(self):
        return self.alg
    
    def get_regret(self):
        return self.get_opt()-self.get_alg()

