                                                                      |
                                                                      |
                                                                      |
                                                                      |
                1                                         23          |
                                                                      |
                                                                      |
                                                                      |
                                                                      |
0                                                                     |
                                                                      |
                                                                      |
        4                                                  5          |
                                                                      |
                                                                      |
                                                                      |
                                                                      |
                                                                      |
                6                                    7                |
                                                   8                  |
                                                                      |
                                                     9                |
                                                                      |
                                                                      |
        0                                                             |
                                                                      |
                                                                      |
                                                                     1|
                                                                      |
                                                                      |
                              2                                       |
        3                                          4                  |
                                                                  5   |

POINTS
------
00: 1,10
01: 17,5
02: 59,5
03: 60,5
04: 9,13
05: 60,13
06: 17,19
07: 54,19
08: 52,20
09: 54,22
10: 9,25
11: 70,28
12: 31,31
13: 9,32
14: 52,32
15: 67,33


ARRAY INDEX
-----------
0
1   2
3   4   5   6

DATA
----
dimension=5
00: (1,2,3,3,5)
01: (0,0,0,4,4)
02: (5,0,0,0,0)
03: (0,0,2,0,0)
04: (0,0,0,8,0)
05: (0,6,0,0,0)
06: (0,0,3,0,0)
07: (0,0,1,0,0)
08: (1,1,1,0,0)
09: (0,0,9,1,0)
09: (0,0,0,1,0)
10: (0,0,5,1,0)
11: (0,2,9,1,0)
12: (0,0,6,1,0)
13: (0,0,9,2,0)
14: (0,0,0,1,9)
15: (0,0,0,7,8)

HEAP
----
8
7   1
2   0   15  13


Brute force
0: ->4->1
1: ->4->0
2: ->5->3
3: ->5->2
4: ->1->6
5: ->2->3
6: ->1->4
7: ->9->8
8: ->9->7
9: ->7->8
                                                             |
          *                                                  |
                                                             |
                                                             |
                                                             |
                 *                                         **|
                                                             |
                                                             |
                                                             |
                                                             |
                                                             |
                                                             |
                                                             |
         *                                                  *|
                                                             |
                                                             |
                                                             |
                                                             |
                                                             |
                 *                                    *      |
                                                    *        |
                                                             |
                                                      *      |

0: ->10->4->1
1: ->0->6->4
2: ->7->3->5
3: ->7->2->5
4: ->1->0->6
5: ->7->2->3
6: ->1->10->4
7: ->5->9->8
8: ->5->9->7
9: ->14->7->8
10: ->0->6->4
11: ->5->9->15
12: ->10->13->6
13: ->12->6->4
14: ->11->8->9
15: ->7->9->11

VALGRIND
--------
2023-05-02 ==7250== I   refs:      1,122,463,133

FMNIST_TRAIN
------------

function          | pynndescent USB | nndescent USB | pynndescent H | nndescent H |
------------------|-----------------|---------------|---------------|-------------|
total             | 13000           | 17086         | 14282         | 17009       |
make_forest       | 4400            | 4124          | 3654          | 3449        |
update by rp tree | 1700            | 2711          | 1992          | 2840        |
1/16              | 2500            | 5000          | 2990          | 4369        |
2/16              | 2500            | 5000          | 4317          | 4234        |
3/16              | 500             | 100           | 997           | 367         |
4/16              | 500             | 0             | 332           | 0           |



(nnd_env) dr_b@requiem:~/Dropbox/WorkHome/programming/nnd$ make run
./nnd
Reading data/data16x2.csv
Time passed: 0 ms (Reading csv)

NNDescent parameters
********************
Data dimension: 16x2
metric=euclidean
n_neighbors=4
n_trees=0
leaf_size=10
pruning_degree_multiplier=1.5
diversify_prob=1
n_search_trees=1
tree_init=0
seed=1234
low_memory=1
max_candidates=4
n_iters=5
delta=0.001
n_jobs=-1
compressed=0
parallel_batch_queries=0
verbose=1

Time passed: 0 ms (Constructor)
curent graph 0=HeapList(n_heaps=16, n_nodes=4, KeyType=f,
    0 [size=16]
    └──(idx=-1 key=3.40282e+38 flag=0)
        ├──(idx=14 key=3085 flag=0)
        │   └──(idx=0 key=0 flag=0)
        └──(idx=3 key=3506 flag=0)
    1 [size=16]
    └──(idx=14 key=1954 flag=0)
        ├──(idx=13 key=793 flag=0)
        │   └──(idx=6 key=196 flag=0)
        └──(idx=2 key=1764 flag=0)
    2 [size=16]
    └──(idx=-1 key=3.40282e+38 flag=0)
        ├──(idx=1 key=1764 flag=0)
        │   └──(idx=5 key=65 flag=0)
        └──(idx=-1 key=3.40282e+38 flag=0)
    3 [size=16]
    └──(idx=-1 key=3.40282e+38 flag=0)
        ├──(idx=13 key=3330 flag=0)
        │   └──(idx=15 key=833 flag=0)
        └──(idx=8 key=289 flag=0)
    4 [size=16]
    └──(idx=11 key=3946 flag=0)
        ├──(idx=15 key=3764 flag=0)
        │   └──(idx=2 key=2564 flag=0)
        └──(idx=5 key=2601 flag=0)
    5 [size=16]
    └──(idx=15 key=449 flag=0)
        ├──(idx=11 key=325 flag=0)
        │   └──(idx=3 key=64 flag=0)
        └──(idx=5 key=0 flag=0)
    6 [size=16]
    └──(idx=-1 key=3.40282e+38 flag=0)
        ├──(idx=15 key=2696 flag=0)
        │   └──(idx=6 key=0 flag=0)
        └──(idx=10 key=100 flag=0)
    7 [size=16]
    └──(idx=4 key=2061 flag=0)
        ├──(idx=15 key=365 flag=0)
        │   └──(idx=7 key=0 flag=0)
        └──(idx=12 key=673 flag=0)
    8 [size=16]
    └──(idx=1 key=1450 flag=0)
        ├──(idx=2 key=274 flag=0)
        │   └──(idx=7 key=5 flag=0)
        └──(idx=6 key=1226 flag=0)
    9 [size=16]
    └──(idx=4 key=2106 flag=0)
        ├──(idx=3 key=325 flag=0)
        │   └──(idx=8 key=8 flag=0)
        └──(idx=15 key=290 flag=0)
    10 [size=16]
    └──(idx=2 key=2900 flag=0)
        ├──(idx=9 key=2034 flag=0)
        │   └──(idx=10 key=0 flag=0)
        └──(idx=14 key=1898 flag=0)
    11 [size=16]
    └──(idx=-1 key=3.40282e+38 flag=0)
        ├──(idx=4 key=3946 flag=0)
        │   └──(idx=5 key=325 flag=0)
        └──(idx=1 key=3338 flag=0)
    12 [size=16]
    └──(idx=1 key=872 flag=0)
        ├──(idx=8 key=562 flag=0)
        │   └──(idx=6 key=340 flag=0)
        └──(idx=4 key=808 flag=0)
    13 [size=16]
    └──(idx=11 key=3737 flag=0)
        ├──(idx=0 key=548 flag=0)
        │   └──(idx=10 key=49 flag=0)
        └──(idx=14 key=1849 flag=0)
    14 [size=16]
    └──(idx=-1 key=3.40282e+38 flag=0)
        ├──(idx=3 key=793 flag=0)
        │   └──(idx=5 key=425 flag=0)
        └──(idx=8 key=144 flag=0)
    15 [size=16]
    └──(idx=-1 key=3.40282e+38 flag=0)
        ├──(idx=0 key=4885 flag=0)
        │   └──(idx=7 key=365 flag=0)
        └──(idx=10 key=3428 flag=0)
)
Time passed: 1 ms (random init neighbours)
2023-05-01 13:23:30 NN descent for 5 iterations
Time passed: 0 ms (nn descent: init)
2023-05-01 13:23:30     1  /  5
Time passed: 0 ms (sample_candidates: start^)
new_candidates=
HeapList(n_heaps=16, n_nodes=4, KeyType=i,
    0 [size=16]
    └──(idx=15 key=1547875112 flag=x)
        ├──(idx=13 key=1040236340 flag=x)
        │   └──(idx=14 key=966261150 flag=x)
        └──(idx=0 key=542711556 flag=x)
    1 [size=16]
    └──(idx=11 key=231975662 flag=x)
        ├──(idx=14 key=-429293563 flag=x)
        │   └──(idx=8 key=-719133295 flag=x)
        └──(idx=6 key=-593603424 flag=x)
    2 [size=16]
    └──(idx=4 key=718244547 flag=x)
        ├──(idx=8 key=49683271 flag=x)
        │   └──(idx=5 key=-994723389 flag=x)
        └──(idx=10 key=-421551478 flag=x)
    3 [size=16]
    └──(idx=9 key=1183701000 flag=x)
        ├──(idx=13 key=-667912938 flag=x)
        │   └──(idx=14 key=-1500467959 flag=x)
        └──(idx=5 key=622063368 flag=x)
    4 [size=16]
    └──(idx=2 key=718244547 flag=x)
        ├──(idx=11 key=-666973880 flag=x)
        │   └──(idx=12 key=-984165353 flag=x)
        └──(idx=5 key=-1444807508 flag=x)
    5 [size=16]
    └──(idx=14 key=-585396664 flag=x)
        ├──(idx=2 key=-994723389 flag=x)
        │   └──(idx=4 key=-1444807508 flag=x)
        └──(idx=15 key=-967809775 flag=x)
    6 [size=16]
    └──(idx=8 key=534835064 flag=x)
        ├──(idx=12 key=419647108 flag=x)
        │   └──(idx=1 key=-593603424 flag=x)
        └──(idx=10 key=-1362711421 flag=x)
    7 [size=16]
    └──(idx=15 key=1658900655 flag=x)
        ├──(idx=4 key=1089439369 flag=x)
        │   └──(idx=8 key=359833048 flag=x)
        └──(idx=7 key=686612079 flag=x)
    8 [size=16]
    └──(idx=2 key=49683271 flag=x)
        ├──(idx=1 key=-719133295 flag=x)
        │   └──(idx=9 key=-1856445903 flag=x)
        └──(idx=14 key=-885119487 flag=x)
    9 [size=16]
    └──(idx=4 key=1888355621 flag=x)
        ├──(idx=3 key=1183701000 flag=x)
        │   └──(idx=8 key=-1856445903 flag=x)
        └──(idx=15 key=-2020513563 flag=x)
    10 [size=16]
    └──(idx=2 key=-421551478 flag=x)
        ├──(idx=6 key=-1362711421 flag=x)
        │   └──(idx=15 key=-1923441561 flag=x)
        └──(idx=13 key=-878391184 flag=x)
    11 [size=16]
    └──(idx=4 key=2028759508 flag=x)
        ├──(idx=13 key=1717363315 flag=x)
        │   └──(idx=5 key=193710454 flag=x)
        └──(idx=1 key=231975662 flag=x)
    12 [size=16]
    └──(idx=1 key=1717817808 flag=x)
        ├──(idx=6 key=419647108 flag=x)
        │   └──(idx=4 key=-984165353 flag=x)
        └──(idx=8 key=444727173 flag=x)
    13 [size=16]
    └──(idx=1 key=931590867 flag=x)
        ├──(idx=3 key=-667912938 flag=x)
        │   └──(idx=10 key=-878391184 flag=x)
        └──(idx=14 key=-327824936 flag=x)
    14 [size=16]
    └──(idx=1 key=-429293563 flag=x)
        ├──(idx=5 key=-585396664 flag=x)
        │   └──(idx=3 key=-1500467959 flag=x)
        └──(idx=8 key=-885119487 flag=x)
    15 [size=16]
    └──(idx=5 key=-967809775 flag=x)
        ├──(idx=10 key=-1923441561 flag=x)
        │   └──(idx=9 key=-2020513563 flag=x)
        └──(idx=7 key=-1582941412 flag=x)
)
 old_candidates=
HeapList(n_heaps=16, n_nodes=4, KeyType=i,
    0 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    1 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    2 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    3 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    4 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    5 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    6 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    7 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    8 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    9 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    10 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    11 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    12 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    13 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    14 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
    15 [size=16]
    └──(idx=-1 key=2147483647 flag=x)
        ├──(idx=-1 key=2147483647 flag=x)
        │   └──(idx=-1 key=2147483647 flag=x)
        └──(idx=-1 key=2147483647 flag=x)
)
Time passed: 4 ms (sample_candidates: stop^)
Time passed: 0 ms (sample_candidates)
2023-05-01 13:23:30             Generate updates 1/16
2023-05-01 13:23:30             Generate updates 5/16
2023-05-01 13:23:30             Generate updates 9/16
2023-05-01 13:23:30             Generate updates 13/16
updates=[]
Time passed: 0 ms (generate graph updates)
2023-05-01 13:23:30             0 updates applied
Time passed: 0 ms (apply graph updates)
2023-05-01 13:23:30 Stopping threshold met -- exiting after 0 iterations
2023-05-01 13:23:30 NN descent done.

NNDescent parameters
********************
Data dimension: 16x2
metric=euclidean
n_neighbors=4
n_trees=0
leaf_size=10
pruning_degree_multiplier=1.5
diversify_prob=1
n_search_trees=1
tree_init=0
seed=1234
low_memory=1
max_candidates=4
n_iters=5
delta=0.001
n_jobs=-1
compressed=0
parallel_batch_queries=0
verbose=1

Time passed: 8 ms (nnd)
Time passed: 0 ms (brute force)
Recall accuracy: 0.234375 (15/64)
(nnd_env) dr_b@requiem:~/Dropbox/WorkHome/programming/nnd$
