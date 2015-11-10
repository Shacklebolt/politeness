from Model import Model
from request import Request
import numpy as np
import pickle

# Note: Total sentences are 2050, not 2113 in number.

if __name__ == '__main__':

    # don't change this
    np.random.seed(43)
    #####################################
    # hyper parameters to be tweaked here
    load_rnn_from_pickle = False
<<<<<<< Updated upstream
    training_size = 2109  # maximum of 2113, 2050 requests when Ali ran this (September 23, 2015)
    l_rate = 0.001
    mini_batch_size = 20
    reg_cost = 0.001
    epochs = 100
    activation_func = "tanh"
=======
    training_size = 2050  # maximum of 2050 requests when Ali ran this (September 23, 2015)
    l_rate = 0.01
    mini_batch_size = 20
    reg_cost = 0.001
    epochs = 200
>>>>>>> Stashed changes
    dim = 100
    activ_func = "sig"
    ######################################

    # perform 60-15-25 percent split into train-val-test sets
    train = np.ceil(0.6 * training_size)
    val = np.ceil(0.15 * training_size)
    test = training_size - train - val

    # load parsed trees from file in PTB format
    if load_rnn_from_pickle is False:
        with open('WikiTreebankQuartilesRefined.txt', 'rb') as fh:
            RNN = Model(dim=dim, l_rate=l_rate, mini_batch=mini_batch_size, reg_cost=reg_cost, epochs=epochs,
                        activation_func=activ_func)
            all_lines = fh.readlines()

            # NOTE: There should be NO sentence with < or > 2 sentences
            i = 0
            while i < len(all_lines) - 1:
                req = Request()

                # Sentence number 1
                line = all_lines[i]
                p = line.find(' ')
                ptb_string = line[p + 1:]
                rid = line[:p]
                req.id = rid
                req.add_tree(ptb_string, rid)

                # Sentence number 2
                line = all_lines[i + 1]
                p = line.find(' ')
                ptb_string = line[p + 1:]
                rid = line[:p]
                req.add_tree(ptb_string, rid)

                RNN.add_request(req)
                i += 2
    else:
        with open('rnn.pickle', 'rb') as pickle_file:
            RNN = pickle.load(pickle_file)

    # Just debugging
    '''
    RNN = Model(dim=dim, l_rate=l_rate, mini_batch=1, reg_cost=reg_cost, epochs=epochs)
    req = Request()
    req.id = '244336'
    req.add_tree("(ROOT (@NP (@NP (NP thanks) (PP (ADVP (RB very) (RB much)) (@PP (IN for) (NP (PRP$ your) (NN edit))))) (PP (TO to) (NP (DT the) (@NP (JJ <url>) (NN article))))) (. .))", req.id)
    req.add_tree("(ROOT (@SQ (@SQ (MD would) (NP you)) (VP (VB be) (ADJP (JJ interested) (PP (IN in) (S (VBG tackling) (NP (NP (DT the) (NN <url>)) (PP (IN of) (NP <url>)))))))) (. ?))", req.id)
    RNN.add_request(req)
    '''

    indices = np.arange(0, training_size)
    # create separate indices for the 3 data sets
    np.random.shuffle(RNN.requests)
    np.random.shuffle(indices)
    RNN.request_train = indices[:train]
    RNN.request_val = indices[train:train + val]
    RNN.request_test = indices[train + val:]
    # print RNN.cross_validate()
    RNN.train(True)
<<<<<<< Updated upstream

    # Save the trained model
    if load_rnn_from_pickle is False:
        with open('rnn.pickle', 'wb') as pickle_file:
            pickle.dump(RNN, pickle_file, pickle.HIGHEST_PROTOCOL)

    #RNN.check_model_veracity()
=======
    # RNN.check_model_veracity()
>>>>>>> Stashed changes
    print "Test Cost Function, Accuracy, Incorrectly classified sentence Ids"
    print RNN.test()

    hyper_params = "training_size={0}\nl_rate={1}\nmini_batch_size={2}\nreg_cost={3}\nepochs={4}\ndim={5}".format(
        training_size, l_rate, mini_batch_size, reg_cost, epochs, dim)
    print hyper_params
