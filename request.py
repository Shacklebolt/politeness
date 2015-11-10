import numpy as np
from Tree import Tree
from Model import Model
from Node import Node


class Request:
    def __init__(self, _id=None):
        self.id = _id
        self.trees = []
        self.target = np.zeros(shape=(2, 1))
        self.pred_label = None
        self.error = 0
        self.request_prediction = None

    def set_target(self, _class):
        self.target[_class] = 1

    def add_tree(self, penn_tree, _id):
        """
        Makes a root node, fills up its structure from PTB format and adds it to the list of trees
        """
        tree = Tree(_id)
        _class = Model.targets[_id]
        self.set_target(_class)
        tree.set_target(_class)
        tree.root = Node()
        tree.root.read(penn_tree, 0, True)
        # Chaipi for life
        tree.root = tree.root.children[0]
        self.trees.append(tree)

    def combine_scores(self):
        """
        Change score combining logic here if you have to
        :return: Combined scores of all the trees that the request has
        Currently, we're averaging the softmaxed probabilities of the trees
        """
        self.request_prediction = (self.trees[0].predictions + self.trees[1].predictions)/2.0
