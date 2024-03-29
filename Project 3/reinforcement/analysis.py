# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():

    # the optimum policy is hindered by the noise but if we make the 
    # noise 0, then the optimum policy will be to cross the bridge. 
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():

    # here if we make the discount low, then the over all reward will
    # be high for a short path as compare to large path. 
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3b():

    # here if we make the discount low, then the over all reward will
    # be high for a short path as compare to large path. To avoid the
    # Cliffs we have to introduce a low noise. 
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3c():

    # here if we make the discount higher, then the over all reward will
    # be low for a short path as compare to large path. 
    answerDiscount = 0.8
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3d():

    # here if we make the discount higher, then the over all reward will
    # be low for a short path as compare to large path. To avoid the
    # Cliffs we have to introduce a low noise. 
    answerDiscount = 0.8
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3e():

    # if we make the discount parameter 0, and make the living Reward 1,
    # so that the agent will avoid both exits and the cliff. 
    answerDiscount = 0.0
    answerNoise = 0.0
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    answerEpsilon = None
    answerLearningRate = None
    # return answerEpsilon, answerLearningRate
    # it is not possible to select an epsilon and a learning rate for which
    # it is highly likely (greater than 99%) that the optimal policy will be
    # learned after 50 iteration
    return "NOT POSSIBLE"
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
