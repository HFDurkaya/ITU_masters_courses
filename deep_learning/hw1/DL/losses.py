import numpy as np

def loss(probs, y):
    '''
        Calculate the softmax loss
        --------------------------
        :param probs: softmax probabilities
        :param y: correct labels
        :return: loss
    '''
    loss = None

    #### Your implementation starts ######

    # compute the loss
    N = probs.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y]))

    loss = loss / N

    ##### End of your implementation #####
    return loss
