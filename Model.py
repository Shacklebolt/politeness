from Node import Node
from Tree import Tree
from utilities import *
import pickle


class Model:
    word_to_vec = None
    targets = None
    sent_per_req = 2

    def __init__(self, dim=50, reg_cost=0.001, l_rate=0.05, mini_batch=20,
                 epochs=100, activation_func="sig"):
        # list of requests in training set
        self.requests = []

        # training set
        self.request_test = None

        # validation set
        self.request_val = None

        # test set
        self.request_test = None

        # number of classes
        self.classes = 2

        # weight matrix for internal nodes
        self.w = init_random_w((dim, (2*dim+1)))

        # weight matrix for softmax prediction
        self.ws = init_random_ws((self.classes, dim+1))

        # size of total parameters
        self.param_size = dim * (2*dim+1) + (self.classes * (dim+1))

        # weight decay
        self.reg_cost = reg_cost

        # learning rate
        self.l_rate = l_rate

        # number of epochs to run
        self.epochs = epochs

        # mini batch size
        self.mini_batch = mini_batch

        # word vector dimension
        self.dim = dim

        # type of activation function
        self.activation_func = activation_func

        # word-embdeddings dictionary
        file_name = 'treebank_vectors_' + str(self.dim) + 'd_new.pickle'
        with open(file_name, 'rb') as pickle_file:
            Model.word_to_vec = pickle.load(pickle_file)

        # target value for each tree
        with open('treebank_scores.pickle', 'rb') as pickle_file:
            Model.targets = pickle.load(pickle_file)

    def reset_weights(self):
        """
        Assigns new values to weights of the network
        """
        # weight matrix for internal nodes
        self.w = init_random_w((self.dim, 2*self.dim + 1))

        # weight matrix for softmax prediction
        self.ws = init_random_ws((self.classes, self.dim + 1))

    def cross_validate(self, num_folds=5):
        """
        Performs K-Fold Cross Validation on the model.
        Returns the list of accuracies and their mean.
        """
        size = len(self.requests)
        folds = size // num_folds * np.ones(num_folds, dtype=np.int)
        folds[:size % num_folds] += 1
        indices = np.arange(0, size)
        np.random.shuffle(indices)
        np.random.shuffle(self.requests)

        current = 0
        accuracies = np.zeros(num_folds)
        for i, fold in enumerate(folds):
            # Assign training and test sets
            start, stop = current, current + fold
            self.request_test = indices[start:stop]
            self.request_train = np.concatenate((indices[:start], indices[stop:]), axis=0)
            current = stop

            # perform training
            self.train()
            _, accuracies[i], _ = self.test()
            self.reset_weights()

        return np.mean(accuracies), accuracies

    def add_request(self, request):
        """
        Makes a root node, fills up its structure from PTB format and adds it to the list of trees
        """
        self.requests.append(request)

    def add_tree(self, penn_tree, _id):
        """
        Makes a root node, fills up its structure from PTB format and adds it to the list of trees
        """
        tree = Tree(_id=_id)
        _class = Model.targets[_id]
        tree.set_target(_class)
        tree.root = Node()
        tree.root.read(penn_tree, 0, True)
        # Chaipi for life
        tree.root = tree.root.children[0]
        self.trees.append(tree)

    def forward(self, node):
        """
        Runs a forward pass on the whole tree and calculates vectors at each node
        """
        if node.num_child == 0:
            return Model.get_vec(node.word)

        elif node.num_child == 1:
            return np.tanh(Model.get_vec(node.children[0].word))

        elif node.num_child == 2:
            left = self.forward(node.children[0])
            right = self.forward(node.children[1])
            children = concat_with_bias(left, right)

            node.vec = np.tanh(np.dot(self.w, children))

            return node.vec

    def calc_outputs(self, request):
        output_vec = self.forward(request.trees[0].root)
        request.trees[0].predictions = softmax(np.dot(self.ws, concat_with_bias(output_vec)))

        output_vec = self.forward(request.trees[1].root)
        request.trees[1].predictions = softmax(np.dot(self.ws, concat_with_bias(output_vec)))

        # Combine predictions
        request.combine_scores()
        return request.request_prediction

    def back_prop(self, node, delta_com, delta_w, delta_ws):
        """
        Back propagates errors from the root node to all the nodes.
        Computes derivatives for all the model parameters
        """
        if node.num_child == 0:
            # TODO: take word vector derivatives
            return
        elif node.num_child == 1:
            return
        elif node.num_child == 2:
            left_vector = node.children[0].vec
            right_vector = node.children[1].vec
            # [x3, p1]
            # concatenate with bias here
            children = concat_with_bias(left_vector, right_vector)

            # delta_w = delta_com * [x3, p1]
            delta_w += np.dot(delta_com, children.T)

            # W.T * delta_com (*) f'([x3, p1])
            delta_down = np.multiply(np.dot(self.w.T, delta_com), tanh_derivative(children))

            left_delta_down = delta_down[:self.dim]
            right_delta_down = delta_down[self.dim: 2 * self.dim]

            self.back_prop(node.children[0], left_delta_down, delta_w, delta_ws)
            self.back_prop(node.children[1], right_delta_down, delta_w, delta_ws)

    def calc_errors(self, request, delta_w, delta_ws):
        """
        Calls back prop and computes prediction error from the root node.
        For a request, you would want to do this for every sentence
        """
        # For first sentence
        tree = request.trees[0]
        # y - t
        diff_class = tree.predictions - tree.target
        # delta_ws = (y - t) * p2
        delta_ws += np.dot(diff_class, concat_with_bias(tree.root.vec).T)

        # Ws.T * (y-t)
        delta = np.dot(self.ws.T, diff_class)
        # Ws.T * (y-t) * f'(p2)
        delta_node = np.multiply(delta[:-1], tanh_derivative(tree.root.vec))

        tree.error = self.get_cost(tree)
        self.back_prop(tree.root, delta_node, delta_w, delta_ws)

        # For second sentence
        tree = request.trees[1]
        # y - t
        diff_class = tree.predictions - tree.target
        # delta_ws = (y - t) * p2
        delta_ws += np.dot(diff_class, concat_with_bias(tree.root.vec).T)

        # Ws.T * (y-t)
        delta = np.dot(self.ws.T, diff_class)
        # Ws.T * (y-t) * f'(p2)
        delta_node = np.multiply(delta[:-1], tanh_derivative(tree.root.vec))

        tree.error = self.get_cost(tree)
        self.back_prop(tree.root, delta_node, delta_w, delta_ws)

    def update(self, sumGrads, grads):
        """
        Updates model parameters from the computed derivatives
        """
        eps = 1e-3
        params = self.get_params()
        sumGrads += (grads * grads)
        params = params - (self.l_rate * grads / (np.sqrt(sumGrads) + eps))
        # params = params - (self.l_rate * grads)
        self.set_params(params)

    def sgd(self, training_batch):
        """
        Runs Stochastic Gradient Descent on the training batch given
        """
        delta_w = np.zeros(self.w.shape)
        delta_ws = np.zeros(self.ws.shape)
        for r in training_batch:
            request = self.requests[r]
            # perform calculations
            self.calc_outputs(request)
            self.calc_errors(request, delta_w, delta_ws)

        # scale and regularize the parameters
        scale = 1. / (len(training_batch) * 2)
        self.scale_regularize(delta_w, delta_ws, scale)

        return self.get_gradients(delta_w, delta_ws)

    def train(self, is_val=False):
        """
        Runs forward and backward passes on the training set.
        Computes errors and errors derivatives and regularizes.
        Updates model parameters.
        Runs till a stopping criterion is not met.
        """
        # early stopping parameters
        min_cost = np.inf
        max_count = 40
        count_down = max_count
        error_factor = 0.001
        train_size = len(self.request_train)
        val_costs = []

        # best set of parameters
        w_best = None
        ws_best = None

        # AdaGrad parameters
        sumGrads = np.zeros(shape=(self.param_size, 1))

        for epoch in xrange(self.epochs):
            # Shuffle training set and create mini batches
            np.random.shuffle(self.request_train)
            mini_batches = [self.request_train[i:min(i + self.mini_batch, train_size)]
                            for i in xrange(0, train_size, self.mini_batch)]
            # run SGD for each mini batch
            for mini_batch in mini_batches:
                # error derivatives with respect to parameters
                grads = self.sgd(mini_batch)
                self.update(sumGrads, grads)

            sumGrads.fill(0.)
            if is_val:
                # check performance on validation set for early stopping
                pred_cost = self.validate()
                val_costs.append(pred_cost)
                if pred_cost < (1 - error_factor) * min_cost:
                    min_cost = pred_cost
                    count_down = max_count
                    w_best = self.w.copy()
                    ws_best = self.ws.copy()
                else:
                    count_down -= 1

                # performance on validation set has not decreased significantly in the past
                if count_down == 0:
                    self.w = w_best
                    self.ws = ws_best
                    print "last training epoch", epoch
                    break

        print "val_costs:"
        print val_costs

    def validate(self):
        """
        Computes and returns prediction accuracy on the validation set
        """
        val_cost = 0
        for r in self.request_val:
            request = self.requests[r]
            self.calc_outputs(request)
            val_cost += self.get_cost(request.trees[0])
            val_cost += self.get_cost(request.trees[1])

        return val_cost

    def test(self):
        """
        Computes and returns predictions on the hand-held test set.
        Also Returns the number of correct predictions made and the
        ids of incorrectly predicted trees
        """
        test_cost = 0
        correct = 0
        incorrect = []
        for r in self.request_test:
            request = self.requests[r]
            self.calc_outputs(request)
            test_cost += self.get_cost(request.trees[0]) + self.get_cost(request.trees[1])
            request.pred_label = np.argmax(request.request_prediction)
            true_label = np.where(request.target == 1)[0]
            if true_label == request.pred_label:
                correct += 1
            else:
                incorrect.append(request.id)

        return np.around(test_cost, 3), 1.*correct/len(self.request_test), incorrect

    def check_model_veracity(self):
        """
        Checks whether the model is correct by performing numerical gradient check.
        """
        grad = self.sgd(self.request_train)
        epsilon = 1e-5
        initial_params = self.get_params()
        num_grad = np.zeros(self.param_size)
        vector = np.zeros(initial_params.shape)
        scale = 1. / (len(self.request_train) * 2)
        for i in range(self.param_size):
            vector[i] = epsilon

            self.set_params(initial_params + vector)
            self.sgd(self.request_train)
            c_plus = 0
            for t in self.request_train:
                c_plus += self.requests[t].trees[0].error + self.requests[t].trees[1].error
            c_plus *= scale

            self.set_params(initial_params - vector)
            self.sgd(self.request_train)
            c_minus = 0
            for t in self.request_train:
                c_minus += self.requests[t].trees[0].error + self.requests[t].trees[1].error
            c_minus *= scale
            num_grad[i] = (c_plus - c_minus) / (2 * epsilon)

            vector[i] = 0

        print np.around(np.sum(np.abs(grad - num_grad) / np.abs(grad + num_grad)), 10)
        self.set_params(initial_params)

    def scale_regularize(self, delta_w, delta_ws, scale):
        """
        Performs regularization of the cost function.
        L2 regularization with weight decay
        """
        delta_w *= scale
        delta_w += (self.reg_cost * self.w)
        delta_ws *= scale
        delta_ws += (self.reg_cost * self.ws)

    def get_cost(self, tree):
        """
        Computes the Cross Entropy cost function with regularization.
        Uses computed predictions from the tree.
        """
        # For first sentence
        # Summation {t * log(y)}
        _log = np.log(tree.predictions)
        cost = -(np.sum(np.multiply(tree.target, _log)))
        # L2 weight decay
        cost += self.reg_cost / 2 * (np.sum(self.w * self.w))
        cost += self.reg_cost / 2 * (np.sum(self.ws * self.ws))

        return cost

    def get_params(self):
        """
        Concatenates all model parameters into one-dimensional vector and returns.
        """
        w_ = np.reshape(np.ravel(self.w), (-1, 1))
        ws_ = np.reshape(np.ravel(self.ws), (-1, 1))
        params = np.vstack((w_, ws_))

        return params

    def set_params(self, new_params):
        """
        Sets all the model parameters in a one-dimensional vector
        """
        self.w = np.reshape(new_params[:self.dim * (2*self.dim+1)], self.w.shape)
        self.ws = np.reshape(new_params[self.dim * (2*self.dim+1):], self.ws.shape)

    def numerical_gradient(self, tree):
        """
        Performs numerical gradient checking by taking derivative by definition.
        See Stanford UFLDL for theoretical details.
        """
        epsilon = 1e-5
        initial_params = self.get_params()
        vector = np.zeros(initial_params.shape)
        exp_grad = np.zeros(initial_params.shape)

        for i in range(self.param_size):
            vector[i] = epsilon

            self.set_params(initial_params + vector)
            self.calc_outputs(tree)
            c_plus = self.get_cost(tree)

            self.set_params(initial_params - vector)
            self.calc_outputs(tree)
            c_minus = self.get_cost(tree)

            exp_grad[i] = (c_plus - c_minus) / (2 * epsilon)

            vector[i] = 0

        self.set_params(initial_params)

        return exp_grad

    def activation(self, _input):
        """
        Computes and returns the activation function
        """
        if self.activation_func == "tanh":
            return np.tanh(_input)
        elif self.activation_func == "sig":
            return 1. / (1 + np.exp(-_input))

    def activation_derivative(self, _input):
        """
        Computes and returns derivative of the activation function
        """
        if self.activation_func == "tanh":
            return 1. - np.square(_input)
        elif self.activation_func == "sig":
            return _input * (1. - _input)

    @staticmethod
    def get_gradients(delta_w, delta_ws):
        """
        Concatenates the derivatives of all model parameters and returns.
        """
        deltaw_ = np.reshape(np.ravel(delta_w), (-1, 1))
        deltaws_ = np.reshape(np.ravel(delta_ws), (-1, 1))

        return np.vstack((deltaw_, deltaws_))

    @staticmethod
    def get_vec(word):
        """
        Maps word to its vector from the embedding matrix
        """
        if word in Model.word_to_vec:
            return Model.word_to_vec[word]
        else:
            return Model.word_to_vec['unknown']
