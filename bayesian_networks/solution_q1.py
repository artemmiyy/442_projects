# Define probabilities
burglary_probs = {True: 0.001, False: 0.999}
earthquake_probs = {True: 0.002, False: 0.998}

alarm_cond = {
    (True, True): 0.95,
    (True, False): 0.94,
    (False, True): 0.29,
    (False, False): 0.001
}

john_cond = {True: 0.9, False: 0.05}
mary_cond = {True: 0.7, False: 0.01}

# need P(B | J) = P(B, J) / P(J)

if __name__ == "__main__":
    print("Q1, Author: Artemii Polshcha")
    print("----------------------------")
    # find the joint probability for B, E, A
    def joint_probability(b, e, a):
        return (burglary_probs[b] *
                earthquake_probs[e] *
                alarm_cond[(b, e)] *
                john_cond[a])

    # find P(J = +j)
    p_j_true = sum(
        joint_probability(b, e, a)
        for b in [True, False]
        for e in [True, False]
        for a in [True, False]
    )

    p_b_true_j_true = sum(
        joint_probability(True, e, a)
        for e in [True, False]
        for a in [True, False]
    )

    p_b_given_j = p_b_true_j_true / p_j_true

    # test
    print("P(B | J = +j) =", p_b_given_j)
    # print out the network here
    