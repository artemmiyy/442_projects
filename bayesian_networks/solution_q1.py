# Question 1: Implementing Variable Elimination for Inference in Bayes Nets

# provided probabilies

P_of_B = {True: 0.001, False: 0.999}
P_of_E = {True: 0.002, False: 0.998}

P_of_A_given_BE = {
    (True, True): 0.95, 
    (True, False): 0.94,
    (False, True): 0.29,
    (False, False): 0.001
}

P_of_J_given_A = {True: 0.9, False: 0.05}
