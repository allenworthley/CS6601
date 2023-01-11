# CS 6601: Artificial Intelligence - Assignment 2 - Search

## Setup

Clone this repository:

`git clone https://github.gatech.edu/omscs6601/assignment_1.git`

The submission scripts depend on the presence of 3 python packages - `requests`, `future`, and `nelson`. Install them using the command below:

`pip install -r requirements.txt`

Python 2.7 is recommended and has been tested.

Read [setup.md](./setup.md) for more information on how to effectively manage your git repository and troubleshooting information.

## Overview

Search is an integral part of AI. It helps in problem solving across a wide variety of domains where a solution isn’t immediately clear.  You will implement several graph search algorithms with the goal of solving bi-directional search.

### Due Date

This assignment is due on Bonnie and T-Square on June 11th by 11:59PM UTC-12 (Anywhere on Earth). The deliverables for the assignment are:
• All functions completed in `search_submission.py`

### The Files

While you'll only have to edit and submit **_search_submission.py_**, there are a number of notable files:

1. **_search_submission.py_**: Where you will implement your _PriorityQueue_, _Breadth First Search_, _Uniform Cost Search_, _A* Search_, _Bi-directional Search_
2. **_search_submission_tests.py_**: Sample tests to validate your searches locally.
3. **_romania_graph.pickle_**: Serialized graph files for Romania.
4. **_atlanta_osm.pickle_**: Serialized graph files for Atlanta (optional for robust testing for Race!).
4. **_submit.py_**: A script to submit your work.
5. **_explorable_graph.py_**: A subclass of `networkx.graph` that includes tracking of explored and neighbor nodes.
6. **_visualize_graph.py_**: Module to visualize search results.
6. **_osm2networkx.py_**: Module used by visualize graph to read OSM networks.

## The Assignment

Your task is to implement several informed search algorithms that will calculate a driving route between two points in Romania with a minimal time and space cost.
There is a search_submission_tests file to help you along the way. Your searches should be executed with minimal runtime and memory overhead.

We will be using an undirected network.  The graph consists of pages as nodes, and the edges are the links interconnecting them.

### Warmups
We'll start by implementing some simpler optimization and search algorithms before the real exercises.

#### Warmup 1: Priority queue

_[10 points]_

In all searches that involve calculating path cost or heuristic (e.g. uniform-cost), we have to order our search frontier. It turns out the way that we do this can impact our overall search runtime.

To show this, you'll implement a priority queue and demonstrate its performance benefits. For large graphs, sorting all input to a priority queue is impractical. As such, the data structure you implement should have an amortized O(1) insertion and O(lg n) removal time. It should do better than the naive implementation in our tests (InsertionSortQueue), which sorts the entire list after every insertion.

> Hint:
> The heapq module has been imported for you.
> Each edge has an associated weight.

#### Warmup 2: BFS

_[10 pts]_

To get you started with handling graphs, implement and test breadth-first search over the test network.

You'll do complete this by writing the "breadth_first_search" method. This returns a path of nodes from a given start node to a given end node, as a list.

For this part, it is optional to use the PriorityQueue as your frontier. You will require it from the next question onwards. You can use it here too if you want to be consistent.

> **Notes**:
> 1. You need to include start and goal in the path.
> 2. If your start and goal are the same then just return [].
> 3. Both of the above are just to keep your results consistent with our test cases.
> 4. You can access all the neighbors of a given node by calling `graph[node]`, or `graph.neighbors(node)` ONLY. To measure your search performance, the explorablegraph.py provided keeps track of which nodes you have accessed in this way (this is referred to as the set of 'Explored' nodes). To retrieve the set of nodes you've explored in this way, call `graph.explored_nodes`. If you wish to perform multiple searches on the same graph instance, call `graph.reset_search()` to clear out the current set of 'Explored' nodes. Note however, that you will not have access to the explored set on the test server. Also, there is no need to reset the graph while submitting to the test server.

#### Warmup 3: Uniform-cost search

_[15 points]_

Implement uniform-cost search, using PriorityQueue as your frontier. From now on, PriorityQueue should be your default frontier.

`uniform_cost_search()` should return the same arguments as breadth-first search: the path to the goal node (as a list of nodes).

> **Notes**:
> 1. You do need to include start and goal in the path.
> 2. If your start and goal are the same then just return []
> 3. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.
> 4. The above are just to keep your results consistent with our test cases.

#### Warmup 4: A* search

_[15 points]_

Implement A* search using Euclidean distance as your heuristic. You'll need to implement heuristic_euclid() then pass that function to a_star() as the heuristic parameter. We provide null_heuristic() as a baseline heuristic to test against when calling a_star tests.

> **Hint**:
> You can find a node's position by calling the following to check if the key is available.
> * Romania Map - `graph.node[n]['pos']`
> * Atlanta Map - `graph.node[n]['position']`

> **Notes**:
> 1. You do need to include start and goal in the path.
> 2. If your start and goal are the same then just return []
> 3. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.
> 4. The above are just to keep your results consistent with our test cases.

---
### Exercises
The following exercises will require you to implement several kinds of bidirectional searches. The benefits of these algorithms over uninformed or unidirectional search are more clearly seen on larger graphs. As such, during grading, we will evaluate your performance on the map of Atlanta [OpenStreetMap](http://wiki.openstreetmap.org) included in this assignment.

For these exercises, we recommend you take a look at the following resources.

1. [A Star meets Graph Theory](https://github.gatech.edu/omscs6601/assignment_2/raw/master/resources/A%20Star%20meets%20Graph%20Theory.pdf)
2. [Applications of Search](https://github.gatech.edu/omscs6601/assignment_2/raw/master/resources/Applications%20of%20Search.pdf)
3. [Bi Directional A Star - Slides](https://github.gatech.edu/omscs6601/assignment_2/raw/master/resources/Bi%20Directional%20A%20Star%20-%20Slides.pdf)
4. [Bi Directional A Star with Additive Approx Bounds](https://github.gatech.edu/omscs6601/assignment_2/raw/master/resources/Bi%20Directional%20A%20Star%20with%20Additive%20Approx%20Bounds.pdf)
5. [Bi Directional A Star](https://github.gatech.edu/omscs6601/assignment_2/raw/master/resources/Bi%20Directional%20A%20Star.pdf)
6. [Search Algorithms Slide Deck](https://github.gatech.edu/omscs6601/assignment_2/raw/master/resources/Search%20Algorithms%20Slide%20Deck.pdf)

#### Exercise 1: Bidirectional uniform-cost search

_[20 points]_

Implement bidirectional uniform-cost search. Remember that this requires starting your search at both the start and end states.

`bidirectional_ucs()` should return the path from the start node to the goal node (as a list of nodes).

> **Notes**:
> 1. You do need to include start and goal in the path.
> 2. If your start and goal are the same then just return []
> 3. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.

> The notes above are to keep your results consistent with our test cases.

#### Exercise 2: Bidirectional A* search

_[30 points]_

Implement bidirectional A* search. Remember that you need to calculate a heuristic for both the start-to-goal search and the goal-to-start search.

To test this function, as well as using the provided tests, you can compare the path computed by bidirectional A star to bidirectional ucs search above.
bidirectional_a_star should return the path from the start node to the goal node, as a list of nodes.

> **Notes**:
> 1. You do need to include start and goal in the path.
> 2. If your start and goal are the same then just return []
> 3. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.

### The Race!

Here's your chance to show us your best stuff. This part is mandatory if you want to compete in the race for extra credit. Implement `custom_search()` using whatever strategy you like.

Race will be based on Atlanta Pickle data.

## References

Here are some notes you might find useful.
1. [Bonnie: Error Messages](https://docs.google.com/document/d/1hykYneVoV_JbwBjVz9ayFTA6Yr3pgw6JBvzrCgM0vyY/pub)
2. [Bi-directional Search](https://docs.google.com/document/d/14Wr2SeRKDXFGdD-qNrBpXjW8INCGIfiAoJ0UkZaLWto/pub)
3. [Using Landmarks](https://docs.google.com/document/d/1YEptGbSYUtu180MfvmrmA4B6X9ImdI4oOmLaaMRHiCA/pub)

## Frequent Issues and Solutions
1. Make sure you clean up any changes/modifications/additions you make to the networkx graph structure before you exit the search function. Depending on your changes, the auto grader might face difficulties while testing. The best alternative is to create your own data structure(s).
2. If you're having problems (exploring too many nodes) with your Breadth first search implementation, one thing many students have found useful is to re-watch the Udacity videos for an optimization trick mentioned.
3. While submitting to Bonnie, many times the submission goes through even if you get an error on the terminal. You should check the web interface to make sure it’s not gone through before re-submitting. On the other hand, make sure your final submission goes through with Bonnie.
4. Most 'NoneType object ...' errors are because the path you return is not completely connected (a pair of successive nodes in the path are not connected). Or because the path variable itself is empty.
5. Adding unit tests to your code may cause your Bonnie submission to fail. It is best to comment them out when you submit to Bonnie.
6. Make sure you're returning [] for when the source and destination points are the same.
