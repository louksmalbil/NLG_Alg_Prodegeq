# coding: utf-8


"""
This decision tree algorithm was developed by Louk Smalbil and Silvia Pagliaro for their research internship at the 
Utrecht University, department of Information and Computing Science.
The algorithm can determine which generalised quantifier belongs to which configuration.
This algorithm can be used in the future to improve object detection and reasoning procedures involving quantities.  
"""

# For Python 2 / 3 compatability
from __future__ import print_function
import numpy as np
import random

# Toy dataset.
# Format: each row is an example.
# The last column is the label.
# The first two columns are features.
# Feel free to play with it by adding more features & examples.
# Interesting note: I've written this so the 2nd and 5th examples
# have the same features, but different labels - so we can see how the
# tree handles this case.
training_data = [
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [-0.29,"Few"],
    [0.00,"All"],
    [0.29,"Most"],
    [0.29,"Lots of"],
    [-0.29,"Few"],
    [0.00,"All"],
    [0.29,"Most"],
    [-0.29,"Some"],
    [0.00,"All"],
    [0.29,"Most"],
    [1.00,"As many as"],
    [0.29,"Majority"],
    [0.00,"All"],
    [1.00,"Half"],
    [0.29,"Almost all"],
    [0.00,"All"],
    [1.00,"As many as"],
    [-0.29,"Some"],
    [1.00,"Evenly divided"],
    [0.80,"Half"],
    [0.50,"Almost equally mixed"],
    [0.50,"Predominantly"],
    [-0.50,"Few"],
    [0.50,"Mainly"],
    [0.50,"Most"],
    [1.00,"As many as"],
    [0.50,"Most"],
    [-0.50,"Some"],
    [0.00,"Only"],
    [0.00,"All"],
    [0.00,"All"],
    [-0.29,"a pair of"],
    [0.29,"A lot of"],
    [-0.29,"Some"],
    [0.00,"All"],
    [0.29,"Mostly"],
    [0.00,"All"],
    [0.29,"Most"],
    [-0.29,"Some"],
    [0.29,"Most"],
    [-0.29,"Some"],
    [0.00,"Only"],
    [0.50,"Mostly"],
    [0.00,"All"],
    [0.50,"Most"],
    [0.50,"Lots of"],
    [-0.50,"Few"],
    [0.00,"All"],
    [0.50,"Most"],
    [-0.50,"Some"],
    [-0.29,"a pair of"],
    [0.00,"Only"],
    [0.00,"Only"],
    [0.29,"Lots of"],
    [-0.29,"A couple of"],
    [0.00,"All"],
    [0.00,"All"],
    [-0.29,"Some"],
    [-0.50,"Some"],
    [0.50,"A lot of"],
    [0.29,"Lots of"],
    [0.17,"Mostly"],
    [0.00,"All"],
    [0.17,"Vast majority"],
    [-0.13,"A"],
    [-0.29,"Some"],
    [0.13,"All except"],
    [-0.13,"A"],
    [0.13,"A lot of"],
    [0.13,"Nearly all"],
    [0.00,"All"],
    [-0.13,"A"],
    [0.13,"A lot of"],
    [-0.13,"A"],
    [0.17,"Mostly"],
    [-0.29,"Aside from"],
    [0.29,"A lot of"],
    [0.17,"Most"],
    [1.00,"Equally divided"],
    [0.29,"Overwhelmingly"],
    [0.17,"Mostly"],
    [0.29,"Most"],
    [0.17,"Most"],
    [-0.29,"Few"],
    [1.00,"As many as"],
    [0.50,"Mostly"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"Every"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"Several"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.00,"All"],
    [0.29,"Most"],
    [-0.29,"A couple of"],
    [0.29,"All except"],
    [-0.29,"A couple of"],
    [0.00,"All"],
    [0.00,"All"],
    [0.29,"Most"],
    [-0.29,"Few"],
    [0.00,"All"],
    [0.00,"All"],
    [0.29,"Many more"],
    [0.00,"Only"],
    [0.00,"All"],
    [0.29,"A bit more"],
    [0.00,"All"],
    [0.29,"Most"],
    [0.00,"All"],
    [0.29,"Majority"],
    [-0.29,"A couple of"],
    [0.29,"Some"],
    [0.29,"Almost three quarters of"],
    [0.00,"All"],
    [0.00,"All"],
    [0.29,"Majority"],
    [0.00,"All"],
    [0.29,"Two thirds"],
    [0.00,"All"],
    [0.00,"All"],
    [0.29,"Mostly"],
    [0.00,"Only"],
    [0.00,"All"],
    [-0.29,"Only a couple of"],
    [0.00,"All"],
    [0.29,"Mixed"],
    [-0.29,"Mixed"],
    [0.00,"Only"],
    [0.29,"Majority"],
    [-0.29,"Still a number of"],
    [0.00,"All"],
    [0.00,"All"],
    [0.29,"Almost all"],
    [-0.29,"A couple of"],
    [0.29,"Many"],
    [0.00,"Every"],
    [0.00,"All"],
    [0.29,"Most except"],
    [-0.29,"Some"],
    [0.00,"All"],
    [0.29,"Most"],
    [-0.29,"a pair of"],
    [0.29,"Majority"],
    [0.80,"Mostly"],
    [-0.40,"Some"],
    [0.80,"Most"],
    [-0.29,"A couple of"],
    [-0.29,"A couple of"],
    [0.80,"Almost half"],
    [-0.29,"A couple of"],
    [-0.29,"A couple of"],
    [0.80,"Most"],
    [-0.29,"Few"],
    [-0.29,"Few"],
    [1.00,"An equal amount of"],
    [0.29,"More"],
    [-0.29,"Minority"],
    [0.29,"Mostly"],
    [-0.29,"A couple of"],
    [-0.29,"A couple of"],
    [0.29,"Mostly except"],
    [-0.29,"A couple of"],
    [-0.29,"A couple of"],
    [1.00,"An equal amount of"],
    [0.80,"Almost half"],
    [0.80,"Just over half"],
    [0.80,"Just over half"],
    [0.29,"Most"],
    [1.00,"Half"],
    [-0.29,"A couple of"],
    [0.80,"Some"],
    [0.80,"Lots of"],
    [-0.29,"A couple of"],
    [-0.29,"A couple of"],
    [0.80,"Most"],
    [0.80,"More"],
    [1.00,"Some"],
    [0.29,"Most except"],
    [0.29,"All except"],
    [0.29,"Several"],
    [0.40,"Mix"],
    [-0.40,"Mix"],
    [0.80,"Most"],
    [-0.29,"Some"],
    [-0.29,"Some"],
    [0.29,"Most"],
    [0.80,"Most"],
    [0.80,"Most except"],
    [-0.29,"a pair of"],
    [0.80,"More"],
    [0.50,"Mostly"],
    [-0.50,"Some"],
    [-0.50,"Some"],
    [-0.50,"Some"],
    [-0.29,"Few"],
    [0.80,"Mixture"],
    [0.50,"Mixture"],
    [0.80,"Mixture"],
    [0.50,"Mixture"],
    [0.80,"Mix"],
    [0.50,"Mix"],
    [0.50,"More"],
    [1.00,"Equally divided"],
    [0.50,"A bit more"],
    [0.50,"Mix"],
    [0.50,"Predominantly"],
    [0.80,"Predominantly"],
    [-0.50,"One third of"],
    [1.00,"An equal amount of"],
    [0.80,"Almost the same amount of"],
    [-0.50,"One third of"],
    [0.50,"Two thirds"],
    [1.00,"Half"],
    [0.80,"Just over half"],
    [0.50,"Two thirds"],
    [0.80,"Slightly more"],
    [0.50,"Twice as many"],
    [0.67,"Mix"],
    [0.33,"Mix"],
    [0.80,"Some"],
    [-0.80,"Some"],
    [0.80,"Half"],
    [0.50,"Most except"],
    [-0.80,"Half"],
    [-0.50,"Some"],
    [-0.50,"Some"],
    [-0.29,"Small amount of"],
    [0.50,"Majority"],
    [0.80,"Majority"],
    [1.00,"Half"],
    [-0.33,"One third of"],
    [0.00,"All"],
    [0.00,"All"],
    [0.29,"Most"],
    [-0.29,"A couple of"],
    [0.00,"All"],
    [0.29,"Some"],
    [-0.29,"Some"],
    [0.29,"All except"],
    [0.00,"All"],
    [0.29,"Mostly"],
    [-0.29,"Only a few"],
    [0.00,"All"],
    [0.29,"Many more"],
    [0.00,"All"],
    [-0.29,"Less"],
    [-0.29,"Minority"],
    [0.00,"All"],
    [-0.29,"Some"],
    [0.29,"Mostly"],
    [-0.29,"Few"],
    [0.00,"All"],
    [0.29,"Majority"],
    [0.29,"Mostly"],
    [-0.29,"A couple of"],
    [0.00,"All"],
    [0.29,"Three quarters of"],
    [0.00,"All"],
    [0.29,"All except"],
    [-0.29,"Almost one third of"],
    [0.29,"More"],
    [0.00,"Only"],
    [0.00,"All"],
    [0.29,"Most"],
    [-0.29,"Only a couple of"],
    [0.00,"All"],
    [0.29,"Almost all"],
    [-0.29,"A couple of"],
    [0.00,"Exclusively"],
    [0.29,"Mostly"],
    [0.00,"All"],
    [0.29,"Majority"],
    [0.29,"Most"],
    [-0.29,"Some"],
    [0.00,"All"],
    [0.29,"Most"],
    [-0.29,"Some"],
    [0.00,"All"],
    [0.29,"Most"],
    [0.29,"Most"],
    [-0.29,"a pair of"],
    [0.00,"All"],
    [0.50,"Double"],
    [0.00,"All"],
    [0.50,"A combination of"],
    [0.50,"Most"],
    [0.00,"Only"],
    [0.00,"All"],
    [0.50,"More"],
    [0.00,"All"],
    [-0.50,"One third of"],
    [0.00,"All"],
    [0.50,"A bit more"],
    [0.00,"All"],
    [0.50,"Mix"],
    [0.00,"Every"],
    [0.00,"All"],
    [-0.50,"One third of"],
    [0.50,"Two thirds"],
    [0.00,"All"],
    [0.00,"Only"],
    [0.29,"Mostly"],
    [0.00,"Just"],
    [0.50,"More than half"],
    [0.00,"All"],
    [0.50,"Most"],
    [-0.50,"Few"],
    [0.00,"All"],
    [0.50,"More"],
    [-0.50,"Half"],
    [0.00,"All"],
    [0.50,"Most"],
    [-0.50,"Some"],
    [0.50,"Some"],
    [-0.50,"Some"],
    [0.50,"More"],
    [0.00,"Every"],
    [-0.50,"Some"],
    [0.00,"All"],
    [0.50,"Most"],
    [0.00,"All"],
    [0.50,"Most"],
    [0.00,"All"],
    [-0.50,"One third of"],
    [0.29,"Mostly"],
    [-0.29,"A couple of"],
    [0.29,"Most"],
    [-0.29,"A couple of"],
    [0.29,"Majority"],
    [-0.29,"Few"],
    [0.29,"Mostly"],
    [-0.29,"Some"],
    [0.29,"Most"],
    [-0.29,"Some"],
    [0.00,"All"],
    [0.29,"A lot of"],
    [-0.29,"Only a few"],
    [-0.29,"Minority"],
    [0.29,"Majority"],
    [0.29,"Mostly"],
    [-0.29,"Few"],
    [0.29,"Lots of"],
    [-0.29,"A couple of"],
    [0.29,"All except"],
    [0.29,"Most"],
    [-0.29,"Only a few"],
    [0.29,"Majority"],
    [0.29,"Almost all except"],
    [0.29,"Lots of"],
    [-0.29,"A couple of"],
    [0.29,"Most"],
    [0.29,"Mostly"],
    [0.29,"Almost all"],
    [0.29,"All except"],
    [0.29,"Many more"],
    [0.29,"Majority"],
    [-0.29,"a pair of"],
    [0.29,"Most"],
    [0.50,"Majority"],
    [0.50,"Many"],
    [-0.50,"Only a couple of"],
    [0.50,"More"],
    [0.00,"All"],
    [0.50,"Mixed"],
    [0.50,"Mostly"],
    [-0.29,"Some"],
    [0.29,"Mostly"],
    [-0.50,"Few"],
    [0.50,"Lots of"],
    [-0.50,"One third of"],
    [0.50,"Two thirds"],
    [0.50,"Two thirds"],
    [0.50,"Two thirds"],
    [0.50,"Majority"],
    [-0.50,"One third of"],
    [0.86,"All except"],
    [-0.29,"Some"],
    [0.50,"Twice as many"],
    [-0.29,"a pair of"],
    [0.50,"Slightly Predominantly"],
    [0.00,"Only"],
    [0.50,"Mostly"],
    [-0.50,"Some"],
    [-0.50,"Less frequent"],
    [0.50,"Many"],
    [0.00,"Every"],
    [0.50,"Most"],
    [-0.50,"Some"],
    [0.00,"All"],
    [0.50,"Two thirds"],
    [0.00,"All"],
    [0.13,"All but one"],
    [0.13,"Most"],
    [0.13,"Almost all"],
    [0.13,"All except"],
    [0.00,"All"],
    [0.13,"All but one"],
    [0.13,"All except"],
    [0.13,"All but one"],
    [0.13,"Lots of"],
    [0.13,"All but one"],
    [0.00,"All"],
    [0.13,"Almost all"],
    [0.13,"All except"],
    [0.13,"Almost all"],
    [-0.13,"Only exception"],
    [0.00,"Only"],
    [0.00,"All"],
    [0.13,"Majority"],
    [0.13,"Almost all except"],
    [0.00,"All"],
    [0.13,"All except"],
    [0.13,"Every except"],
    [0.13,"Almost all except"],
    [0.13,"All except"],
    [0.13,"All but one"],
    [0.50,"Majority"],
    [-0.29,"A couple of"],
    [0.50,"Most"],
    [-0.29,"Few"],
    [-0.13,"A"],
    [-0.13,"A"],
    [-0.13,"A"],
    [-0.13,"A"],
    [-0.13,"A"],
    [-0.13,"One"],
    [-0.13,"A"],
    [-0.13,"A"],
    [-0.13,"A"],
    [-0.13,"A"],
    [0.50,"Big majority"],
    [0.29,"Primary"],
    [-0.29,"Few"],
    [0.50,"Majority"],
    [-0.13,"Not many"],
    [-0.13,"Only one"],
    [-0.14,"Only one"],
    [0.29,"More"],
    [0.29,"Mostly"],
    [0.29,"Mostly"],
    [0.50,"Mostly"],
    [0.50,"Most"],
    [0.29,"Almost three quarters of"],
    [0.29,"Almost three quarters of"],
    [0.50,"Two thirds"],
    [0.50,"Two thirds"],
    [-0.50,"One third of"],
    [0.14,"All but one"],
    [0.50,"Some"],
    [0.29,"Majority"],
    [0.17,"Majority"],
    [0.29,"Most"],
    [-0.29,"A couple of"],
    [-0.13,"Only one"],
    [-0.13,"Only one"],
    [0.50,"Mostly"],
    [-0.14,"Only one"],
    [0.29,"Almost all"],
    [0.50,"Many"],
    [0.50,"Majority"],
    [1.00,"An equal number of"],
    [0.50,"A lot of"],
    [-0.50,"Much lower amount of"],
    [0.29,"More"],
    [0.50,"Most"],
    [0.50,"Two thirds"],
    [0,"Only"],
    #Louk
    [0,"Even"],
    [0,"Only"],
    [0,"All"],
    [0,"A lot of"],
    [0,"Only"],
    [0,"Even amount of"],
    [0,"All"],
    [0,"All"],
    [0,"All"],
    [0,"All"],
    [0,"Only"],
    [0,"Even"],
    [0.33,"Mostly"],
    [0.33,"Most"],
    [0,"Only"],
    [0.33,"More than"],
    [0.33,"Many more"],
    [0.33,"More than"],
    [0,"All"],
    [0.33,"About a third of"],
    [0,"Only"],
    [0.33,"More than"],
    [-0.25,"A small amount of"],
    [0.25,"Most"],
    [0,"Only"],
    [0.25,"Four times more"],
    [0,"All"],
    [0.25,"More than"],
    [0,"Only"],
    [0.25,"More than"],
    [0,"Only"],
    [0.25,"More than"],
    [0,"All"],
    [0.25,"More than"],
    [0.54,"Majority"],
    [1,"Almost the same amount of"],
    [0.54,"More than"],
    [0.43,"More than"],
    [0.67,"More than"],
    [0.86,"As many as"],
    [0.75,"More than"],
    [1,"The same number as"],
    [0.54,"More than"],
    [0.75,"A couple more "],
    [0.43,"Slightly more than"],
    [0.82,"Slightly more than"],
    [0.54,"Majority"],
    [0.54,"Most"],
    [1,"Same"],
    [0.86,"Greater than"],
    [0.75,"Same"],
    [0.86,"An equal amount of"],
    [0.75,"An equal amount of"],
    [0.54,"More than"],
    [0.54,"More than"],
    [0.82,"More than"],
    [0.82,"More than"],
    [0.54,"More than"],
    [0.82,"Slightly more than"],
    [0.54,"Slightly more than"],
    [-0.11,"Minority"],
    [1,"Even"],
    [0.18,"Most"],
    [-0.18,"Less"],
    [0.89,"More"],
    [0.18,"More than"],
    [0.5,"More than"],
    [0.13,"More than"],
    [0.25,"More than"],
    [0.82,"More than"],
    [0.82,"More than"],
    [0.82,"There is a mixture of"],
    [0.0,"All"],
    [0.5,"Some"],
    [0.67,"More"],
    [-0.67,"Less"],
    [0.5,"Most"],
    [0.75,"Most"],
    [1,"An equal amount of"],
    [0.67,"More"],
    [1,"As many as"],
    [0.33,"More than"],
    [0.67,"More than"],
    [0.67,"There is a mixture of"],
    [0.67,"There is a mixture of"],
    [0.67,"Mostly"],
    [0.82,"More"],
    [1,"An equal number of"],
    [1,"An equal number of"],
    [0.71,"More"],
    [1,"As many as"],
    [0.8,"More"],
    [0.67,"More than"],
    [0.67,"More than"],
    [0.82,"More than"],
    [1,"As many as"],
    [0.67,"There is a mixture of"],
    [0.67,"More"],
    [0.18,"Majority"],
    [-0.18,"Few"],
    [0.42,"Most"],
    [0,"All"],
    [0.6,"More than"],
    [0.67,"More than half"],
    [0.18,"More than"],
    [0.18,"Most"],
    [-0.67,"About a third of"],
    [0,"All"],
    [0.82,"Almost the same amount of"],
    [0.82,"A bit more"],
    [0.82,"Most"],
    [0.83,"More than"],
    [0.8,"More than"],
    [0.82,"More than"],
    [0.82,"Almost equally divided"]
]


# Column labels.
# These are used only to print the tree.
header = ["type", "label"]


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


#######
# Demo:
unique_vals(training_data, 0)
# unique_vals(training_data, 1)
#######

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


#######
# Demo:
class_counts(training_data)
#######


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity



def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


my_tree = build_tree(training_data)


print_tree(my_tree)


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):        
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

#######
# Demo:
# The tree predicts the 1st row of our
# training data is an apple with confidence 1.
classify(training_data[0], my_tree)
#######


def print_leaf(counts):
    probabsLoc = []
    arrayLoc = []
    arrayLoc = [k for k, v in counts.items()]
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for i in range(0,len(counts)):
        probabsLoc.append(float([x for x in counts.values()][i]) / total)
    probabs = probabsLoc
    array = arrayLoc
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    
    return probs, probabs, array


#######
# Demo:
# Printing that a bit nicer
print_leaf(classify(training_data[0], my_tree))
#######


#######
# Demo:
# On the second example, the confidence is lower
print_leaf(classify(training_data[1], my_tree))
#######


# Evaluate
testing_data = [
[0.8, ""],
]


# for row in testing_data:
#     print ("Actual: %s. Predicted: %s" %
#            (row[-1], print_leaf(classify(row, my_tree))))


# for row in testing_data:
#     lista, probabs, array = print_leaf(classify(row, my_tree))
#     print ("Type %s" % row[-2])

#     print("Chosen quantifier: %s" % np.random.choice(array, 1, p=probabs)[0])
#     print('--------------------------------------')
    

from itertools import combinations
import itertools

def compositions(num, width):
    m = num + width - 1
    last = (m,)
    first = (-1,)
    for t in combinations(range(m), width - 1):
        yield [v - u - 1 for u, v in zip(first + t, t + last)]

        
def genSamples(n, nrSamples):
    # test
    i = 0
    for t in compositions(n, 4):
        #print(t)
        i = i + 1
    #print('- ' * 20)
    #print(i)
    comp = list(compositions(n,4))
    comp = sorted(comp)
    rand_items = random.sample(comp, nrSamples)
    return rand_items
    
testData = genSamples(9,10)


def divideByZero(arg1, arg2):
    if(arg1 == 0 or arg2 == 0):
        return 0.00
    else:
        return round(arg1/arg2, 2)

# General case: 'There are' + Q + '...'
#
# Adverbs = ending in "ly" ==> templates: 
# 1) 'There are' + Q + '...' 
# 2) 'The objects are' + Q + '...'
#
# Special case: 'All, Many, Most, A lot of, Lots of, Few,' ==> template: Q + 'objects are' + '...'
#
# Majority, minority: 'The' + Q + 'of objects is' + '...'
#
# Comparatives

def templateGenerator(quantifier, value):
    quantifier = quantifier.lower()
    special = ['all', 'many', 'most', 'a lot of', 'lots of', 'few']
    majmin = ['majority', 'minority']

    if(quantifier in special):
        if('but' in quantifier):
            quantifier = quantifier - 'but'
            print('%s objects but n are %s' % (quantifier, value))
        elif('except' in quantifier): 
            quantifier = quantifier - 'except'
            print('%s objects except for n are %s' % (quantifier, value))
        else:
            print('%s objects are %s' % (quantifier, value))
    elif(quantifier == 'a'):
        print('There is %s %s' % (quantifier, value))
    elif(quantifier in majmin):
        print('The %s of objects is %s' % (quantifier, value))
    elif(quantifier.endswith('ly')):
        tempChosen = np.random.choice([1,2], 1, p=[0.5, 0.5])[0]
        if(tempChosen == 1):
            print('The objects are %s %s' % (quantifier, value))
        else:
            print('There are %s %s' % (quantifier, value))
    elif(quantifier.endswith('of')):
        print('%s the objects are %s' % (quantifier, value))
    elif('more' in quantifier or 'less' in quantifier):
        if(value == 'red'):
            value == 'red objects'
            opposite = 'blue objects'
        elif(value == 'blue'):
            value == 'blue objects'
            opposite = 'red objects'
        elif(value == 'square'):
            value = 'squares'
            opposite = 'circles'
        else:
            value = 'circles'
            opposite = 'squares'
        print('There are %s %s than %s' % (quantifier, value, opposite))
    else:
        print('There are %s %s' % (quantifier, value))
    
    
def colourQuant(ratio):
    lista, probabs, array = print_leaf(classify(ratio, my_tree))
    
    print ("Ratio: %s" % ratio[-2])
    if(ratio[-2] == 1.0):
        col = 'blue and red'
    elif(ratio[-2] == 0.00):
        if(max(R,B) == B):
            col = 'blue'
        else:
            col = 'red'
    elif(ratio[-2] > 0):
        if(B>R):
            col = 'blue'
        else:
            col = 'red'
    else:
        if(B>R):
            col = 'red'
        else:
            col = 'blue'
   
    #print("Chosen quantifier for colour %s: %s" % (col, np.random.choice(array, 1, p=probabs)[0]))
    templateGenerator(np.random.choice(array,1,p=probabs)[0], col)
    if(ratio[-2] == 1.0):
        return
    else:
        checkInnerShape(col)
    
def checkInnerShape(col):
    if(col == 'red'):
        innerRatio = [min(divideByZero(RS,RC), divideByZero(RC,RS)), ""]
        if(innerRatio[0] == divideByZero(RS,RC)):
            shape = 'circles'
        else:
            shape = 'squares'
    else:
        innerRatio = [min(divideByZero(BS,BC), divideByZero(BC,BS)), ""]
        if(innerRatio[0] == divideByZero(BS,BC)):
            shape = 'circles'
        else:
            shape = 'squares'
    second_list, probs, arr = print_leaf(classify(innerRatio, my_tree))
    print ("Ratio: %s" % innerRatio[0])
    print("Within the %s, %s are %s" % (col, np.random.choice(arr, 1, p=probs)[0], shape))
    print('--------------------------------------')
    
def shapeQuant(ratio):
    lista, probabs, array = print_leaf(classify(ratio, my_tree))
    print ("Ratio: %s" % ratio[-2])
    if(ratio[-2] == 1.0):
        shape = 'circles and squares'
    elif(ratio[-2] == 0.00):
        if(max(S,C) == S):
            shape = 'squares'
        else:
            shape = 'circles'
    elif(ratio[-2] > 0):
        if(C>S):
            shape = 'circles'
        else:
            shape = 'squares'
    else:
        if(C>S):
            shape = 'squares'
        else:
            shape = 'circles'
    #print("Chosen quantifier for shape %s: %s" % (shape, np.random.choice(array, 1, p=probabs)[0]))
    templateGenerator(np.random.choice(array,1,p=probabs)[0], shape)
    if(ratio[-2] == 1.0):
        return
    else:
        checkInnerColour(shape)
    
def checkInnerColour(shape):
    if(shape == 'squares'):
        innerRatio = [min(divideByZero(BS,RS), divideByZero(RS,BS)), ""]
        if(innerRatio[0] == divideByZero(BS,RS)):
            col = 'red'
        else:
            col = 'blue'
    else:
        innerRatio = [min(divideByZero(BC,RC), divideByZero(RC,BC)), ""]
        if(innerRatio[0] == divideByZero(BC,RC)):
            col = 'red'
        else:
            col = 'blue'
    print ("Ratio: %s" % innerRatio[0])
    second_list, probs, arr = print_leaf(classify(innerRatio, my_tree))
    print("Within the %s, %s are %s" % (shape, np.random.choice(arr, 1, p=probs)[0], col))
    print('--------------------------------------')
    
def combinationQuant(ratio):
    lista, probabs, array = print_leaf(classify(ratio, my_tree))
    print ("Ratio: %s" % ratio[-2])
    col = max(R,B)
    if(col == R):
        col = 'red'
    else:
        col = 'blue'
    
    shape = max(S,C)
    if(shape == S):
        shape = 'squares'
    else:
        shape = 'circles'

    #print("Chosen quantifier for %s %s: %s" % (col, shape, np.random.choice(array, 1, p=probabs)[0]))
    comb = col + ' ' + shape 
    templateGenerator(np.random.choice(array,1,p=probabs)[0], comb)
    print('--------------------------------------')
    return col, shape


# input = [RS, BS, BC, RC]

n = 16

for scenario in testData:
    print('[RS, BS, BC, RC]')
    print(scenario)
    input = scenario
    
    RS = input[0]
    BS = input[1]
    BC = input[2]
    RC = input[3]

    R = RS + RC
    B = BS + BC
    S = RS + BS
    C = RC + BC

    # Colour ratio: RED VS BLUE
    colourRatio = [R,B]
    # print(colourRatio)

    # Shape ratio: SQUARES VS CIRCLES
    shapeRatio = [S,C]
    # print(shapeRatio)

    # =============================================================================================
    
    # Compute ratios for colour
    ratioC1 = round(divideByZero(R,B), 2)
    ratioC2 = round(divideByZero(B,R), 2)
    finalRatioColour = min(ratioC1, ratioC2)

    # Randomise the choice of adding the minus before the final ratio IF ratio is different from 0 or 1
    if(finalRatioColour == 1.0 or finalRatioColour == 0.0):
        finalRatioColour = [finalRatioColour, ""]
    else:
        finalRatioColour = [np.random.choice([-finalRatioColour,finalRatioColour], 1, p=[0.5, 0.5])[0],""]

    # =============================================================================================
    
    # Compute ratios for shape
    ratioS1 = round(divideByZero(S,C), 2)
    ratioS2 = round(divideByZero(C,S), 2)
    finalRatioShape = min(ratioS1, ratioS2)

    # Randomise the choice of adding the minus before the final ratio IF ratio is different from 0 or 1
    if(finalRatioShape == 1.0 or finalRatioShape == 0.0):
        finalRatioShape = [finalRatioShape, ""]
    else:
        finalRatioShape = [np.random.choice([-finalRatioShape,finalRatioShape], 1, p=[0.5, 0.5])[0],""]
    
    # =============================================================================================
    
    doneShape = ''
    doneColour = ''
        
    for elem in input:
        if(elem > n/2):
            doneColour, doneShape = combinationQuant([round((n-elem)/elem, 2), ""])
            # TODO: call functions to output quantifiers
            if(doneColour == 'red'):
                if(R>B):
                    if(finalRatioColour[0] >= 0):
                        finalRatioColour[0] = -finalRatioColour[0]
                    colourQuant(finalRatioColour)
            else:
                if(B>R):
                    if(finalRatioColour[0] >= 0):
                        finalRatioColour[0] = -finalRatioColour[0]
                    colourQuant(finalRatioColour)
            if(doneShape == 'squares'):
                if(S>C):
                    if(finalRatioShape[0] >= 0):
                        finalRatioShape[0] = -finalRatioShape[0]
                    shapeQuant(finalRatioShape)
            else:
                if(C>S):
                    if(finalRatioShape[0] >= 0):
                        finalRatioShape[0] = -finalRatioShape[0]
                    shapeQuant(finalRatioShape)
                    
    if(not doneColour or not doneShape):
        if(finalRatioShape == 0.0):
            shapeQuant(finalRatioShape)
        elif(finalRatioColour == 0.0):
            colourQuant(finalRatioColour)
        else:
            shapeQuant(finalRatioShape)
            colourQuant(finalRatioColour)
            
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------\n\n')

