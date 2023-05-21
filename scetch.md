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

FMNIST_TRAIN
------------

function          | pynndescent USB | nndescent USB | pynndescent H | nndescent H |
------------------|-----------------|---------------|---------------|-------------|
total             | 21801           | 15519         | 14282         | 12402       |
make_forest       | 7339            | 6294          | 3654          | 3500        |
update by rp tree | 2784            | 2961          | 1992          | 2972        |
1/16              | 4072            | 2555          | 2990          | 2614        |
2/16              | 5353            | 2696          | 4317          | 2595        |
3/16              | 1594            | 394           | 997           | 336         |
4/16              | 526             | 0             | 332           | 0           |


BENCHMARK TEST
--------------

Thu May 18 11:42:42 2023 Building RP forest with 11 trees
Thu May 18 11:42:42 2023 NN descent for 10 iterations
         1  /  10
         2  /  10
        Stopping threshold met -- exiting after 2 iterations
Time passed: 28497.493267059326 ms (pynndescent dim=(1440, 1025))
Thu May 18 11:43:11 2023 Building RP forest with 11 trees
Thu May 18 11:43:11 2023 NN descent for 10 iterations
         1  /  10
         2  /  10
         3  /  10
        Stopping threshold met -- exiting after 3 iterations
Time passed: 396.700382232666 ms (pynndescent dim=(1440, 1025))
NNDescent(
        data=Matrix<float>(n_rows=1440, n_cols=1025),
        metric=euclidean,
        n_neighbors=30,
        n_trees=11,
        leaf_size=30,
        pruning_degree_multiplier=1.5,
        diversify_prob=1,
        tree_init=1,
        seed=-1,
        low_memory=1,
        max_candidates=30,
        n_iters=10,
        delta=0.001,
        n_threads=4,
        compressed=0,
        parallel_batch_queries=0,
        verbose=1,
        algorithm=nnd,

        angular_trees=0,
)
Time passed: 0 ms (Constructor)
2023-05-18 11:43:11 Building RP forest with 11 trees
Time passed: 47 ms (make forest)
2023-05-18 11:43:11 Update Graph by  RP forest
Time passed: 0 ms (make leaf array)
Time passed: 52 ms (update by leaf array)
Time passed: 0 ms (update graph by rp-tree forest)
Time passed: 0 ms (random init neighbours)
2023-05-18 11:43:11 NN descent for 10 iterations
Time passed: 0 ms (nn descent: init)
2023-05-18 11:43:11     1  /  10
Time passed: 14 ms (sample_candidates)
Time passed: 96 ms (generate_graph_updates)
2023-05-18 11:43:11             2116 updates applied
Time passed: 34 ms (apply apply_graph_updates updates)
2023-05-18 11:43:11     2  /  10
Time passed: 6 ms (sample_candidates)
Time passed: 76 ms (generate_graph_updates)
2023-05-18 11:43:11             1015 updates applied
Time passed: 23 ms (apply apply_graph_updates updates)
2023-05-18 11:43:11     3  /  10
Time passed: 15 ms (sample_candidates)
Time passed: 8 ms (generate_graph_updates)
2023-05-18 11:43:11             0 updates applied
Time passed: 1 ms (apply apply_graph_updates updates)
2023-05-18 11:43:11 Stopping threshold met -- exiting after 2 iterations
2023-05-18 11:43:11 NN descent done.
Time passed: 0 ms (nn_descent)
Time passed: 4 ms (heapsort)
Time passed: 385.94746589660645 ms (nndescent dim=(1440, 1025))
NNDescent(
        data=Matrix<float>(n_rows=1440, n_cols=1025),
        metric=euclidean,
        n_neighbors=30,
        n_trees=11,
        leaf_size=30,
        pruning_degree_multiplier=1.5,
        diversify_prob=1,
        tree_init=1,
        seed=-1,
        low_memory=1,
        max_candidates=30,
        n_iters=10,
        delta=0.001,
        n_threads=4,
        compressed=0,
        parallel_batch_queries=0,
        verbose=1,
        algorithm=bf,

        angular_trees=0,
)
[==================================================] 100 %
Time passed: 473.3738899230957 ms (nndescent brute force dim=(1440, 1025))

coil20: Accuracy pynndescent vs bf
Average accuracy of 0.9999074074074075


coil20: Accuracy nndescent vs bf
Average accuracy of 0.9999074074074075

Time passed: 2821.687698364258 ms (KDTree dim=(1440, 1025))

coil20: Accuracy bf vs kdtree
Average accuracy of 1.0


coil20: Accuracy nndescent vs pynndescent
Average accuracy of 0.9998148148148148

Benchmarking glove25 ...
***********************

Thu May 18 11:43:19 2023 Building RP forest with 32 trees
Thu May 18 11:43:44 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
         7  /  20
         8  /  20
         9  /  20
         10  /  20
         11  /  20
        Stopping threshold met -- exiting after 11 iterations
Time passed: 222421.70310020447 ms (pynndescent dim=(1183514, 25))
Thu May 18 11:47:03 2023 Building RP forest with 32 trees
Thu May 18 11:47:27 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
         7  /  20
         8  /  20
         9  /  20
         10  /  20
         11  /  20
        Stopping threshold met -- exiting after 11 iterations
Time passed: 230022.03941345215 ms (pynndescent dim=(1183514, 25))
NNDescent(
        data=Matrix<float>(n_rows=1183514, n_cols=25),
        metric=dot,
        n_neighbors=30,
        n_trees=32,
        leaf_size=30,
        pruning_degree_multiplier=1.5,
        diversify_prob=1,
        tree_init=1,
        seed=-1,
        low_memory=1,
        max_candidates=30,
        n_iters=20,
        delta=0.001,
        n_threads=4,
        compressed=0,
        parallel_batch_queries=0,
        verbose=1,
        algorithm=nnd,

        angular_trees=1,
)
Time passed: 0 ms (Constructor)
2023-05-18 11:50:54 Building RP forest with 32 trees
Time passed: 18487 ms (make forest)
2023-05-18 11:51:13 Update Graph by  RP forest
Time passed: 271 ms (make leaf array)
Time passed: 24300 ms (update by leaf array)
Time passed: 0 ms (update graph by rp-tree forest)
Time passed: 300 ms (random init neighbours)
2023-05-18 11:51:38 NN descent for 20 iterations
Time passed: 0 ms (nn descent: init)
2023-05-18 11:51:38     1  /  20
Time passed: 4913 ms (sample_candidates)
Time passed: 9497 ms (generate_graph_updates)
2023-05-18 11:52:00             30394512 updates applied
Time passed: 8032 ms (apply apply_graph_updates updates)
2023-05-18 11:52:01     2  /  20
Time passed: 4811 ms (sample_candidates)
Time passed: 11395 ms (generate_graph_updates)
2023-05-18 11:52:25             22028879 updates applied
Time passed: 7443 ms (apply apply_graph_updates updates)
2023-05-18 11:52:25     3  /  20
Time passed: 4967 ms (sample_candidates)
Time passed: 9334 ms (generate_graph_updates)
2023-05-18 11:52:44             8656058 updates applied
Time passed: 4191 ms (apply apply_graph_updates updates)
2023-05-18 11:52:44     4  /  20
Time passed: 3986 ms (sample_candidates)
Time passed: 6076 ms (generate_graph_updates)
2023-05-18 11:52:57             3138820 updates applied
Time passed: 3113 ms (apply apply_graph_updates updates)
2023-05-18 11:52:57     5  /  20
Time passed: 4042 ms (sample_candidates)
Time passed: 2954 ms (generate_graph_updates)
2023-05-18 11:53:07             1090082 updates applied
Time passed: 1786 ms (apply apply_graph_updates updates)
2023-05-18 11:53:07     6  /  20
Time passed: 3572 ms (sample_candidates)
Time passed: 1339 ms (generate_graph_updates)
2023-05-18 11:53:13             511277 updates applied
Time passed: 741 ms (apply apply_graph_updates updates)
2023-05-18 11:53:13     7  /  20
Time passed: 3764 ms (sample_candidates)
Time passed: 1045 ms (generate_graph_updates)
2023-05-18 11:53:18             299398 updates applied
Time passed: 541 ms (apply apply_graph_updates updates)
2023-05-18 11:53:18     8  /  20
Time passed: 3866 ms (sample_candidates)
Time passed: 625 ms (generate_graph_updates)
2023-05-18 11:53:24             151591 updates applied
Time passed: 195 ms (apply apply_graph_updates updates)
2023-05-18 11:53:24     9  /  20
Time passed: 3470 ms (sample_candidates)
Time passed: 280 ms (generate_graph_updates)
2023-05-18 11:53:28             87783 updates applied
Time passed: 122 ms (apply apply_graph_updates updates)
2023-05-18 11:53:28     10  /  20
Time passed: 3079 ms (sample_candidates)
Time passed: 242 ms (generate_graph_updates)
2023-05-18 11:53:32             48795 updates applied
Time passed: 66 ms (apply apply_graph_updates updates)
2023-05-18 11:53:32     11  /  20
Time passed: 4013 ms (sample_candidates)
Time passed: 157 ms (generate_graph_updates)
2023-05-18 11:53:36             30040 updates applied
Time passed: 53 ms (apply apply_graph_updates updates)
2023-05-18 11:53:36 Stopping threshold met -- exiting after 10 iterations
2023-05-18 11:53:36 NN descent done.
Time passed: 54 ms (nn_descent)
Time passed: 1104 ms (heapsort)
Time passed: 163947.74293899536 ms (nndescent dim=(1183514, 25))

glove25: Accuracy nndescent vs pynndescent
Average accuracy of 0.47117752726203527

Benchmarking glove50 ...
***********************

Thu May 18 11:54:14 2023 Building RP forest with 32 trees
Thu May 18 11:54:43 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
         7  /  20
         8  /  20
         9  /  20
         10  /  20
        Stopping threshold met -- exiting after 10 iterations
Time passed: 254919.65913772583 ms (pynndescent dim=(1183514, 50))
Thu May 18 11:58:30 2023 Building RP forest with 32 trees
Thu May 18 11:58:54 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
         7  /  20
         8  /  20
         9  /  20
         10  /  20
        Stopping threshold met -- exiting after 10 iterations
Time passed: 215752.30383872986 ms (pynndescent dim=(1183514, 50))
NNDescent(
        data=Matrix<float>(n_rows=1183514, n_cols=50),
        metric=dot,
        n_neighbors=30,
        n_trees=32,
        leaf_size=30,
        pruning_degree_multiplier=1.5,
        diversify_prob=1,
        tree_init=1,
        seed=-1,
        low_memory=1,
        max_candidates=30,
        n_iters=20,
        delta=0.001,
        n_threads=4,
        compressed=0,
        parallel_batch_queries=0,
        verbose=1,
        algorithm=nnd,

        angular_trees=1,
)
Time passed: 0 ms (Constructor)
2023-05-18 12:02:08 Building RP forest with 32 trees
Time passed: 25263 ms (make forest)
2023-05-18 12:02:33 Update Graph by  RP forest
Time passed: 281 ms (make leaf array)
Time passed: 22474 ms (update by leaf array)
Time passed: 0 ms (update graph by rp-tree forest)
Time passed: 308 ms (random init neighbours)
2023-05-18 12:02:56 NN descent for 20 iterations
Time passed: 0 ms (nn descent: init)
2023-05-18 12:02:56     1  /  20
Time passed: 4646 ms (sample_candidates)
Time passed: 9099 ms (generate_graph_updates)
2023-05-18 12:03:16             25217744 updates applied
Time passed: 6312 ms (apply apply_graph_updates updates)
2023-05-18 12:03:16     2  /  20
Time passed: 4857 ms (sample_candidates)
Time passed: 13775 ms (generate_graph_updates)
2023-05-18 12:03:43             19860091 updates applied
Time passed: 7455 ms (apply apply_graph_updates updates)
2023-05-18 12:03:43     3  /  20
Time passed: 4663 ms (sample_candidates)
Time passed: 10613 ms (generate_graph_updates)
2023-05-18 12:04:04             9768575 updates applied
Time passed: 5368 ms (apply apply_graph_updates updates)
2023-05-18 12:04:04     4  /  20
Time passed: 4245 ms (sample_candidates)
Time passed: 7123 ms (generate_graph_updates)
2023-05-18 12:04:19             4087265 updates applied
Time passed: 3178 ms (apply apply_graph_updates updates)
2023-05-18 12:04:19     5  /  20
Time passed: 3759 ms (sample_candidates)
Time passed: 4103 ms (generate_graph_updates)
2023-05-18 12:04:29             1504228 updates applied
Time passed: 1682 ms (apply apply_graph_updates updates)
2023-05-18 12:04:29     6  /  20
Time passed: 3416 ms (sample_candidates)
Time passed: 2365 ms (generate_graph_updates)
2023-05-18 12:04:36             623611 updates applied
Time passed: 762 ms (apply apply_graph_updates updates)
2023-05-18 12:04:36     7  /  20
Time passed: 3250 ms (sample_candidates)
Time passed: 1193 ms (generate_graph_updates)
2023-05-18 12:04:41             281366 updates applied
Time passed: 335 ms (apply apply_graph_updates updates)
2023-05-18 12:04:41     8  /  20
Time passed: 3143 ms (sample_candidates)
Time passed: 662 ms (generate_graph_updates)
2023-05-18 12:04:46             134981 updates applied
Time passed: 171 ms (apply apply_graph_updates updates)
2023-05-18 12:04:46     9  /  20
Time passed: 3140 ms (sample_candidates)
Time passed: 361 ms (generate_graph_updates)
2023-05-18 12:04:50             66891 updates applied
Time passed: 88 ms (apply apply_graph_updates updates)
2023-05-18 12:04:50     10  /  20
Time passed: 3237 ms (sample_candidates)
Time passed: 219 ms (generate_graph_updates)
2023-05-18 12:04:54             30586 updates applied
Time passed: 44 ms (apply apply_graph_updates updates)
2023-05-18 12:04:54 Stopping threshold met -- exiting after 9 iterations
2023-05-18 12:04:54 NN descent done.
Time passed: 46 ms (nn_descent)
Time passed: 1021 ms (heapsort)
Time passed: 167623.62480163574 ms (nndescent dim=(1183514, 50))

glove50: Accuracy nndescent vs pynndescent
Average accuracy of 0.5815169064328767

Benchmarking glove100 ...
***********************

Thu May 18 12:05:27 2023 Building RP forest with 32 trees
Thu May 18 12:05:57 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
         7  /  20
         8  /  20
         9  /  20
         10  /  20
         11  /  20
        Stopping threshold met -- exiting after 11 iterations
Time passed: 263264.2102241516 ms (pynndescent dim=(1183514, 100))
Thu May 18 12:09:51 2023 Building RP forest with 32 trees
Thu May 18 12:10:23 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
         7  /  20
         8  /  20
         9  /  20
         10  /  20
         11  /  20
        Stopping threshold met -- exiting after 11 iterations
Time passed: 280100.3987789154 ms (pynndescent dim=(1183514, 100))
NNDescent(
        data=Matrix<float>(n_rows=1183514, n_cols=100),
        metric=dot,
        n_neighbors=30,
        n_trees=32,
        leaf_size=30,
        pruning_degree_multiplier=1.5,
        diversify_prob=1,
        tree_init=1,
        seed=-1,
        low_memory=1,
        max_candidates=30,
        n_iters=20,
        delta=0.001,
        n_threads=4,
        compressed=0,
        parallel_batch_queries=0,
        verbose=1,
        algorithm=nnd,

        angular_trees=1,
)
Time passed: 0 ms (Constructor)
2023-05-18 12:14:33 Building RP forest with 32 trees
Time passed: 50669 ms (make forest)
2023-05-18 12:15:24 Update Graph by  RP forest
Time passed: 321 ms (make leaf array)
Time passed: 26011 ms (update by leaf array)
Time passed: 0 ms (update graph by rp-tree forest)
Time passed: 329 ms (random init neighbours)
2023-05-18 12:15:50 NN descent for 20 iterations
Time passed: 0 ms (nn descent: init)
2023-05-18 12:15:50     1  /  20
Time passed: 5137 ms (sample_candidates)
Time passed: 11734 ms (generate_graph_updates)
2023-05-18 12:16:14             26095320 updates applied
Time passed: 6544 ms (apply apply_graph_updates updates)
2023-05-18 12:16:14     2  /  20
Time passed: 5413 ms (sample_candidates)
Time passed: 17414 ms (generate_graph_updates)
2023-05-18 12:16:46             19638475 updates applied
Time passed: 8560 ms (apply apply_graph_updates updates)
2023-05-18 12:16:46     3  /  20
Time passed: 5266 ms (sample_candidates)
Time passed: 16221 ms (generate_graph_updates)
2023-05-18 12:17:15             10238488 updates applied
Time passed: 7198 ms (apply apply_graph_updates updates)
2023-05-18 12:17:15     4  /  20
Time passed: 5412 ms (sample_candidates)
Time passed: 11930 ms (generate_graph_updates)
2023-05-18 12:17:39             6466762 updates applied
Time passed: 5638 ms (apply apply_graph_updates updates)
2023-05-18 12:17:39     5  /  20
Time passed: 6049 ms (sample_candidates)
Time passed: 8037 ms (generate_graph_updates)
2023-05-18 12:17:56             2558969 updates applied
Time passed: 2237 ms (apply apply_graph_updates updates)
2023-05-18 12:17:56     6  /  20
Time passed: 4358 ms (sample_candidates)
Time passed: 4123 ms (generate_graph_updates)
2023-05-18 12:18:06             1116978 updates applied
Time passed: 1125 ms (apply apply_graph_updates updates)
2023-05-18 12:18:06     7  /  20
Time passed: 4070 ms (sample_candidates)
Time passed: 2630 ms (generate_graph_updates)
2023-05-18 12:18:13             529429 updates applied
Time passed: 595 ms (apply apply_graph_updates updates)
2023-05-18 12:18:14     8  /  20
Time passed: 4070 ms (sample_candidates)
Time passed: 1504 ms (generate_graph_updates)
2023-05-18 12:18:20             255580 updates applied
Time passed: 292 ms (apply apply_graph_updates updates)
2023-05-18 12:18:20     9  /  20
Time passed: 3796 ms (sample_candidates)
Time passed: 916 ms (generate_graph_updates)
2023-05-18 12:18:25             128298 updates applied
Time passed: 158 ms (apply apply_graph_updates updates)
2023-05-18 12:18:25     10  /  20
Time passed: 3842 ms (sample_candidates)
Time passed: 525 ms (generate_graph_updates)
2023-05-18 12:18:30             63296 updates applied
Time passed: 75 ms (apply apply_graph_updates updates)
2023-05-18 12:18:30     11  /  20
Time passed: 3869 ms (sample_candidates)
Time passed: 291 ms (generate_graph_updates)
2023-05-18 12:18:35             31406 updates applied
Time passed: 45 ms (apply apply_graph_updates updates)
2023-05-18 12:18:35 Stopping threshold met -- exiting after 10 iterations
2023-05-18 12:18:35 NN descent done.
Time passed: 48 ms (nn_descent)
Time passed: 1050 ms (heapsort)
Time passed: 243194.85759735107 ms (nndescent dim=(1183514, 100))

glove100: Accuracy nndescent vs pynndescent
Average accuracy of 0.6136039511713984

Benchmarking glove200 ...
***********************

Thu May 18 12:19:21 2023 Building RP forest with 32 trees
Thu May 18 12:20:18 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
         7  /  20
         8  /  20
         9  /  20
         10  /  20
         11  /  20
         12  /  20
         13  /  20
        Stopping threshold met -- exiting after 13 iterations
Time passed: 390037.08958625793 ms (pynndescent dim=(1183514, 200))
Thu May 18 12:25:52 2023 Building RP forest with 32 trees
Thu May 18 12:26:38 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
         7  /  20
         8  /  20
         9  /  20
         10  /  20
         11  /  20
         12  /  20
         13  /  20
        Stopping threshold met -- exiting after 13 iterations
Time passed: 339935.6288909912 ms (pynndescent dim=(1183514, 200))
NNDescent(
        data=Matrix<float>(n_rows=1183514, n_cols=200),
        metric=dot,
        n_neighbors=30,
        n_trees=32,
        leaf_size=30,
        pruning_degree_multiplier=1.5,
        diversify_prob=1,
        tree_init=1,
        seed=-1,
        low_memory=1,
        max_candidates=30,
        n_iters=20,
        delta=0.001,
        n_threads=4,
        compressed=0,
        parallel_batch_queries=0,
        verbose=1,
        algorithm=nnd,

        angular_trees=1,
)
Time passed: 0 ms (Constructor)
2023-05-18 12:31:33 Building RP forest with 32 trees
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
tree depth limit reached
Time passed: 106021 ms (make forest)
2023-05-18 12:33:19 Update Graph by  RP forest
Time passed: 378 ms (make leaf array)
Time passed: 25865 ms (update by leaf array)
Time passed: 0 ms (update graph by rp-tree forest)
Time passed: 339 ms (random init neighbours)
2023-05-18 12:33:46 NN descent for 20 iterations
Time passed: 0 ms (nn descent: init)
2023-05-18 12:33:46     1  /  20
Time passed: 4979 ms (sample_candidates)
Time passed: 13871 ms (generate_graph_updates)
2023-05-18 12:34:10             25529899 updates applied
Time passed: 5345 ms (apply apply_graph_updates updates)
2023-05-18 12:34:10     2  /  20
Time passed: 5356 ms (sample_candidates)
Time passed: 24564 ms (generate_graph_updates)
2023-05-18 12:34:48             20499195 updates applied
Time passed: 7541 ms (apply apply_graph_updates updates)
2023-05-18 12:34:48     3  /  20
Time passed: 5310 ms (sample_candidates)
Time passed: 22922 ms (generate_graph_updates)
2023-05-18 12:35:24             9691672 updates applied
Time passed: 7619 ms (apply apply_graph_updates updates)
2023-05-18 12:35:25     4  /  20
Time passed: 5259 ms (sample_candidates)
Time passed: 17926 ms (generate_graph_updates)
2023-05-18 12:35:53             5270823 updates applied
Time passed: 5331 ms (apply apply_graph_updates updates)
2023-05-18 12:35:54     5  /  20
Time passed: 4975 ms (sample_candidates)
Time passed: 11471 ms (generate_graph_updates)
2023-05-18 12:36:13             2923285 updates applied
Time passed: 2836 ms (apply apply_graph_updates updates)
2023-05-18 12:36:13     6  /  20
Time passed: 4619 ms (sample_candidates)
Time passed: 7037 ms (generate_graph_updates)
2023-05-18 12:36:27             1518577 updates applied
Time passed: 1372 ms (apply apply_graph_updates updates)
2023-05-18 12:36:27     7  /  20
Time passed: 4468 ms (sample_candidates)
Time passed: 4403 ms (generate_graph_updates)
2023-05-18 12:36:37             781629 updates applied
Time passed: 704 ms (apply apply_graph_updates updates)
2023-05-18 12:36:37     8  /  20
Time passed: 4246 ms (sample_candidates)
Time passed: 2994 ms (generate_graph_updates)
2023-05-18 12:36:45             406715 updates applied
Time passed: 394 ms (apply apply_graph_updates updates)
2023-05-18 12:36:45     9  /  20
Time passed: 4150 ms (sample_candidates)
Time passed: 1850 ms (generate_graph_updates)
2023-05-18 12:36:52             215837 updates applied
Time passed: 210 ms (apply apply_graph_updates updates)
2023-05-18 12:36:52     10  /  20
Time passed: 4218 ms (sample_candidates)
Time passed: 1165 ms (generate_graph_updates)
2023-05-18 12:36:58             115179 updates applied
Time passed: 118 ms (apply apply_graph_updates updates)
2023-05-18 12:36:58     11  /  20
Time passed: 4146 ms (sample_candidates)
Time passed: 663 ms (generate_graph_updates)
2023-05-18 12:37:03             61685 updates applied
Time passed: 72 ms (apply apply_graph_updates updates)
2023-05-18 12:37:03     12  /  20
Time passed: 4134 ms (sample_candidates)
Time passed: 400 ms (generate_graph_updates)
2023-05-18 12:37:08             33103 updates applied
Time passed: 39 ms (apply apply_graph_updates updates)
2023-05-18 12:37:08 Stopping threshold met -- exiting after 11 iterations
2023-05-18 12:37:08 NN descent done.
Time passed: 66 ms (nn_descent)
Time passed: 989 ms (heapsort)
Time passed: 336448.7202167511 ms (nndescent dim=(1183514, 200))

glove200: Accuracy nndescent vs pynndescent
Average accuracy of 0.6724247171276947

Benchmarking mnist ...
***********************

Thu May 18 12:37:41 2023 Building RP forest with 21 trees
Thu May 18 12:37:45 2023 NN descent for 16 iterations
         1  /  16
         2  /  16
         3  /  16
         4  /  16
        Stopping threshold met -- exiting after 4 iterations
Time passed: 13045.414686203003 ms (pynndescent dim=(60000, 784))
Thu May 18 12:37:55 2023 Building RP forest with 21 trees
Thu May 18 12:37:58 2023 NN descent for 16 iterations
         1  /  16
         2  /  16
         3  /  16
        Stopping threshold met -- exiting after 3 iterations
Time passed: 12838.75846862793 ms (pynndescent dim=(60000, 784))
NNDescent(
        data=Matrix<float>(n_rows=60000, n_cols=784),
        metric=euclidean,
        n_neighbors=30,
        n_trees=21,
        leaf_size=30,
        pruning_degree_multiplier=1.5,
        diversify_prob=1,
        tree_init=1,
        seed=-1,
        low_memory=1,
        max_candidates=30,
        n_iters=16,
        delta=0.001,
        n_threads=4,
        compressed=0,
        parallel_batch_queries=0,
        verbose=1,
        algorithm=nnd,

        angular_trees=0,
)
Time passed: 0 ms (Constructor)
2023-05-18 12:38:07 Building RP forest with 21 trees
Time passed: 3392 ms (make forest)
2023-05-18 12:38:11 Update Graph by  RP forest
Time passed: 2 ms (make leaf array)
Time passed: 1309 ms (update by leaf array)
Time passed: 0 ms (update graph by rp-tree forest)
Time passed: 5 ms (random init neighbours)
2023-05-18 12:38:12 NN descent for 16 iterations
Time passed: 0 ms (nn descent: init)
2023-05-18 12:38:12     1  /  16
Time passed: 237 ms (sample_candidates)
Time passed: 1667 ms (generate_graph_updates)
2023-05-18 12:38:14             284966 updates applied
Time passed: 429 ms (apply apply_graph_updates updates)
2023-05-18 12:38:14     2  /  16
Time passed: 253 ms (sample_candidates)
Time passed: 2987 ms (generate_graph_updates)
2023-05-18 12:38:18             47752 updates applied
Time passed: 611 ms (apply apply_graph_updates updates)
2023-05-18 12:38:18     3  /  16
Time passed: 247 ms (sample_candidates)
Time passed: 626 ms (generate_graph_updates)
2023-05-18 12:38:19             1184 updates applied
Time passed: 154 ms (apply apply_graph_updates updates)
2023-05-18 12:38:19 Stopping threshold met -- exiting after 2 iterations
2023-05-18 12:38:19 NN descent done.
Time passed: 0 ms (nn_descent)
Time passed: 58 ms (heapsort)
Time passed: 12019.755363464355 ms (nndescent dim=(60000, 784))

mnist: Accuracy nndescent vs pynndescent
Average accuracy of 0.9971605555555555

Benchmarking nytimes ...
***********************

Thu May 18 12:38:22 2023 Building RP forest with 28 trees
Thu May 18 12:38:33 2023 NN descent for 18 iterations
         1  /  18
         2  /  18
         3  /  18
         4  /  18
         5  /  18
         6  /  18
         7  /  18
         8  /  18
         9  /  18
         10  /  18
         11  /  18
        Stopping threshold met -- exiting after 11 iterations
Time passed: 80646.22378349304 ms (pynndescent dim=(290000, 256))
Thu May 18 12:39:43 2023 Building RP forest with 28 trees
Thu May 18 12:39:54 2023 NN descent for 18 iterations
         1  /  18
         2  /  18
         3  /  18
         4  /  18
         5  /  18
         6  /  18
         7  /  18
         8  /  18
         9  /  18
         10  /  18
         11  /  18
        Stopping threshold met -- exiting after 11 iterations
Time passed: 97178.30204963684 ms (pynndescent dim=(290000, 256))
NNDescent(
        data=Matrix<float>(n_rows=290000, n_cols=256),
        metric=dot,
        n_neighbors=30,
        n_trees=28,
        leaf_size=30,
        pruning_degree_multiplier=1.5,
        diversify_prob=1,
        tree_init=1,
        seed=-1,
        low_memory=1,
        max_candidates=30,
        n_iters=18,
        delta=0.001,
        n_threads=4,
        compressed=0,
        parallel_batch_queries=0,
        verbose=1,
        algorithm=nnd,

        angular_trees=1,
)
Time passed: 0 ms (Constructor)
2023-05-18 12:41:20 Building RP forest with 28 trees
Time passed: 10966 ms (make forest)
2023-05-18 12:41:31 Update Graph by  RP forest
Time passed: 53 ms (make leaf array)
Time passed: 5660 ms (update by leaf array)
Time passed: 0 ms (update graph by rp-tree forest)
Time passed: 46 ms (random init neighbours)
2023-05-18 12:41:37 NN descent for 18 iterations
Time passed: 0 ms (nn descent: init)
2023-05-18 12:41:37     1  /  18
Time passed: 1119 ms (sample_candidates)
Time passed: 3819 ms (generate_graph_updates)
2023-05-18 12:41:43             5237991 updates applied
Time passed: 1392 ms (apply apply_graph_updates updates)
2023-05-18 12:41:43     2  /  18
Time passed: 1222 ms (sample_candidates)
Time passed: 6615 ms (generate_graph_updates)
2023-05-18 12:41:54             3952301 updates applied
Time passed: 2531 ms (apply apply_graph_updates updates)
2023-05-18 12:41:54     3  /  18
Time passed: 1282 ms (sample_candidates)
Time passed: 6893 ms (generate_graph_updates)
2023-05-18 12:42:05             1659592 updates applied
Time passed: 2899 ms (apply apply_graph_updates updates)
2023-05-18 12:42:05     4  /  18
Time passed: 1227 ms (sample_candidates)
Time passed: 5239 ms (generate_graph_updates)
2023-05-18 12:42:14             714553 updates applied
Time passed: 1988 ms (apply apply_graph_updates updates)
2023-05-18 12:42:14     5  /  18
Time passed: 1192 ms (sample_candidates)
Time passed: 2687 ms (generate_graph_updates)
2023-05-18 12:42:19             405628 updates applied
Time passed: 1142 ms (apply apply_graph_updates updates)
2023-05-18 12:42:19     6  /  18
Time passed: 1422 ms (sample_candidates)
Time passed: 1579 ms (generate_graph_updates)
2023-05-18 12:42:23             179543 updates applied
Time passed: 368 ms (apply apply_graph_updates updates)
2023-05-18 12:42:23     7  /  18
Time passed: 1429 ms (sample_candidates)
Time passed: 899 ms (generate_graph_updates)
2023-05-18 12:42:25             92643 updates applied
Time passed: 180 ms (apply apply_graph_updates updates)
2023-05-18 12:42:25     8  /  18
Time passed: 1613 ms (sample_candidates)
Time passed: 419 ms (generate_graph_updates)
2023-05-18 12:42:27             36241 updates applied
Time passed: 61 ms (apply apply_graph_updates updates)
2023-05-18 12:42:27     9  /  18
Time passed: 1135 ms (sample_candidates)
Time passed: 234 ms (generate_graph_updates)
2023-05-18 12:42:29             18250 updates applied
Time passed: 33 ms (apply apply_graph_updates updates)
2023-05-18 12:42:29     10  /  18
Time passed: 1108 ms (sample_candidates)
Time passed: 136 ms (generate_graph_updates)
2023-05-18 12:42:30             9728 updates applied
Time passed: 18 ms (apply apply_graph_updates updates)
2023-05-18 12:42:30     11  /  18
Time passed: 1098 ms (sample_candidates)
Time passed: 89 ms (generate_graph_updates)
2023-05-18 12:42:32             6235 updates applied
Time passed: 15 ms (apply apply_graph_updates updates)
2023-05-18 12:42:32 Stopping threshold met -- exiting after 10 iterations
2023-05-18 12:42:32 NN descent done.
Time passed: 23 ms (nn_descent)
Time passed: 281 ms (heapsort)
Time passed: 71759.43326950073 ms (nndescent dim=(290000, 256))

nytimes: Accuracy nndescent vs pynndescent
Average accuracy of 0.8138772413793104

Benchmarking sift ...
***********************

Thu May 18 12:42:41 2023 Building RP forest with 32 trees
Thu May 18 12:43:14 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
        Stopping threshold met -- exiting after 6 iterations
Time passed: 159478.11126708984 ms (pynndescent dim=(1000000, 128))
Thu May 18 12:45:22 2023 Building RP forest with 32 trees
Thu May 18 12:45:53 2023 NN descent for 20 iterations
         1  /  20
         2  /  20
         3  /  20
         4  /  20
         5  /  20
         6  /  20
        Stopping threshold met -- exiting after 6 iterations
Time passed: 153373.0854988098 ms (pynndescent dim=(1000000, 128))
NNDescent(
        data=Matrix<float>(n_rows=1000000, n_cols=128),
        metric=euclidean,
        n_neighbors=30,
        n_trees=32,
        leaf_size=30,
        pruning_degree_multiplier=1.5,
        diversify_prob=1,
        tree_init=1,
        seed=-1,
        low_memory=1,
        max_candidates=30,
        n_iters=20,
        delta=0.001,
        n_threads=4,
        compressed=0,
        parallel_batch_queries=0,
        verbose=1,
        algorithm=nnd,

        angular_trees=0,
)
Time passed: 0 ms (Constructor)
2023-05-18 12:47:57 Building RP forest with 32 trees
Time passed: 28350 ms (make forest)
2023-05-18 12:48:25 Update Graph by  RP forest
Time passed: 196 ms (make leaf array)
Time passed: 22256 ms (update by leaf array)
Time passed: 0 ms (update graph by rp-tree forest)
Time passed: 192 ms (random init neighbours)
2023-05-18 12:48:48 NN descent for 20 iterations
Time passed: 0 ms (nn descent: init)
2023-05-18 12:48:48     1  /  20
Time passed: 3683 ms (sample_candidates)
Time passed: 7562 ms (generate_graph_updates)
2023-05-18 12:49:04             10324600 updates applied
Time passed: 4895 ms (apply apply_graph_updates updates)
2023-05-18 12:49:04     2  /  20
Time passed: 4411 ms (sample_candidates)
Time passed: 13233 ms (generate_graph_updates)
2023-05-18 12:49:29             3612762 updates applied
Time passed: 6773 ms (apply apply_graph_updates updates)
2023-05-18 12:49:29     3  /  20
Time passed: 3895 ms (sample_candidates)
Time passed: 7918 ms (generate_graph_updates)
2023-05-18 12:49:46             594135 updates applied
Time passed: 4446 ms (apply apply_graph_updates updates)
2023-05-18 12:49:46     4  /  20
Time passed: 3920 ms (sample_candidates)
Time passed: 2036 ms (generate_graph_updates)
2023-05-18 12:49:53             112813 updates applied
Time passed: 558 ms (apply apply_graph_updates updates)
2023-05-18 12:49:53     5  /  20
Time passed: 3398 ms (sample_candidates)
Time passed: 535 ms (generate_graph_updates)
2023-05-18 12:49:57             23264 updates applied
Time passed: 67 ms (apply apply_graph_updates updates)
2023-05-18 12:49:57 Stopping threshold met -- exiting after 4 iterations
2023-05-18 12:49:57 NN descent done.
Time passed: 41 ms (nn_descent)
Time passed: 833 ms (heapsort)
Time passed: 121601.31025314331 ms (nndescent dim=(1000000, 128))

sift: Accuracy nndescent vs pynndescent
Average accuracy of 0.9735866999999999

# Benchmark test pynndescent vs nndescent
Data set  | pynndescent [ms] | nndescent [ms] | ratio | accuracy
----------|------------------|----------------|-------|---------
coil20    |            396.7 |          385.9 | 0.973 |  1.000
fmnist    |          12312.6 |        10695.1 | 0.869 |  0.997
glove25   |         230022.0 |       163947.7 | 0.713 |  0.471
glove50   |         215752.3 |       167623.6 | 0.777 |  0.582
glove100  |         280100.4 |       243194.9 | 0.868 |  0.614
glove200  |         339935.6 |       336448.7 | 0.990 |  0.672
mnist     |          12838.8 |        12019.8 | 0.936 |  0.997
nytimes   |          97178.3 |        71759.4 | 0.738 |  0.814
sift      |         153373.1 |       121601.3 | 0.793 |  0.974



>>> print(reverse_graph)
  (0, 0)        1.1920929e-07
  (4, 0)        73.0
  (1, 1)        1.1920929e-07
  (4, 1)        128.0
  (2, 2)        1.1920929e-07
  (3, 2)        1.0
  (7, 2)        221.0
  (3, 3)        1.1920929e-07
  (2, 3)        1.0
  (5, 3)        64.0
  (4, 4)        1.1920929e-07
  (0, 4)        73.0
  (6, 4)        100.0
  (1, 4)        128.0
  (5, 5)        1.1920929e-07
  (3, 5)        64.0
  (7, 5)        72.0
  (6, 6)        1.1920929e-07
  (10, 6)       100.0
  (4, 6)        100.0
  (7, 7)        1.1920929e-07
  (8, 7)        5.0
  (5, 7)        72.0
  (8, 8)        1.1920929e-07
  (7, 8)        5.0
  (9, 8)        8.0
  (9, 9)        1.1920929e-07
  (8, 9)        8.0
  (14, 9)       104.0
  (10, 10)      1.1920929e-07
  (13, 10)      49.0
  (6, 10)       100.0
  (0, 10)       289.0
  (11, 11)      1.1920929e-07
  (15, 11)      34.0
  (5, 11)       325.0
  (12, 12)      1.1920929e-07
  (6, 12)       340.0
  (14, 12)      442.0
  (13, 13)      1.1920929e-07
  (10, 13)      49.0
  (12, 13)      485.0
  (14, 14)      1.1920929e-07
  (9, 14)       104.0
  (15, 14)      226.0
  (15, 15)      1.1920929e-07
  (11, 15)      34.0
  (14, 15)      226.0
>>> self._is_sparse:
  File "<stdin>", line 1
    self._is_sparse:




rp_tree=
[
    2: [0, 4, 10, 13],
    0: [2, 7, 8, 9, 12, 14],
    1: [0, 1, 4, 6, 10, 13],
    2: [3, 5, 11, 15],
    0: [2, 3, 5, 7, 8, 9, 11, 14, 15],
    1: [0, 1, 4, 6, 10, 12, 13],
    0: [0, 1, 4],
    1: [2, 3, 5, 7, 8, 9, 14],
    2: [6, 10, 12, 13],
    3: [11, 15],
    0: [2, 3, 5, 11],
    1: [7, 8, 9, 14, 15],
    2: [0, 1, 4, 6, 10, 12, 13],
    0: [2, 3, 5, 11],
    1: [1, 4, 6, 7, 8, 9, 12, 14, 15],
    2: [0, 10, 13],
    0: [2, 3, 5, 7, 8, 9, 11, 14, 15],
    1: [0, 1, 4, 6, 10, 12, 13],

leaf_array=[
 [1, 2, 3, 5, 6, 7, 8, 12, -1, -1],
 [0, 4, 10, 13, -1, -1, -1, -1, -1, -1],
 [2, 7, 8, 9, 12, 14, -1, -1, -1, -1],
 [0, 1, 4, 6, 10, 13, -1, -1, -1, -1],
 [3, 5, 11, 15, -1, -1, -1, -1, -1, -1],
 [2, 3, 5, 7, 8, 9, 11, 14, 15, -1],
 [0, 1, 4, 6, 10, 12, 13, -1, -1, -1],
 [0, 1, 4, -1, -1, -1, -1, -1, -1, -1],
 [2, 3, 5, 7, 8, 9, 14, -1, -1, -1],
 [6, 10, 12, 13, -1, -1, -1, -1, -1, -1],
 [11, 15, -1, -1, -1, -1, -1, -1, -1, -1],
 [2, 3, 5, 11, -1, -1, -1, -1, -1, -1],
 [7, 8, 9, 14, 15, -1, -1, -1, -1, -1],
 [0, 1, 4, 6, 10, 12, 13, -1, -1, -1],
 [2, 3, 5, 11, -1, -1, -1, -1, -1, -1],
 [1, 4, 6, 7, 8, 9, 12, 14, 15, -1],
 [0, 10, 13, -1, -1, -1, -1, -1, -1, -1],
 [2, 3, 5, 7, 8, 9, 11, 14, 15, -1],
 [0, 1, 4, 6, 10, 12, 13, -1, -1, -1]]


leaf_array=[
    [9, 11, 14, 15, -1, -1, -1, -1, -1, -1],
    [1, 2, 3, 5, 6, 7, 8, 12, -1, -1],
    [0, 4, 10, 13, -1, -1, -1, -1, -1, -1],
    [2, 7, 8, 9, 12, 14, -1, -1, -1, -1],
    [0, 1, 4, 6, 10, 13, -1, -1, -1, -1],
    [3, 5, 11, 15, -1, -1, -1, -1, -1, -1],
    [2, 3, 5, 7, 8, 9, 11, 14, 15, -1],
    [0, 1, 4, 6, 10, 12, 13, -1, -1, -1],
    [0, 1, 4, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 5, 7, 8, 9, 14, -1, -1, -1],
    [6, 10, 12, 13, -1, -1, -1, -1, -1, -1],
    [11, 15, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 5, 11, -1, -1, -1, -1, -1, -1],
    [7, 8, 9, 14, 15, -1, -1, -1, -1, -1],
    [0, 1, 4, 6, 10, 12, 13, -1, -1, -1],
    [2, 3, 5, 11, -1, -1, -1, -1, -1, -1],
    [1, 4, 6, 7, 8, 9, 12, 14, 15, -1],
    [0, 10, 13, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 5, 7, 8, 9, 11, 14, 15, -1],
    [0, 1, 4, 6, 10, 12, 13, -1, -1, -1]
]


indices=[0, 1, 4] ,n_leaves=1
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1]]
indices=[2, 3, 5, 7, 8, 9, 14] ,n_leaves=2
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1]]
indices=[6, 10, 12, 13] ,n_leaves=3
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1], [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1]]
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1], [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1], [of=-1400, hp=[-50, 20], id=[], lr=1,2]]
indices=[11, 15] ,n_leaves=4
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1], [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1], [of=-1400, hp=[-50, 20], id=[], lr=1,2], [of=3.40282e+38, hp=[], id=[11, 15], lr=-1,-1]]
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1], [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1], [of=-1400, hp=[-50, 20], id=[], lr=1,2], [of=3.40282e+38, hp=[], id=[11, 15], lr=-1,-1], [of=1089, hp=[13, 11], id=[], lr=3,4]]
nodes=
[
    [of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1],
    [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1],
    [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1],
    [of=-1400, hp=[-50, 20], id=[], lr=1,2],
    [of=3.40282e+38, hp=[], id=[11, 15], lr=-1,-1],
    [of=1089, hp=[13, 11], id=[], lr=3,4],
    [of=200, hp=[8, 6], id=[], lr=3,5]]
tree=
Tree(leaf_size=10, n_leaves=4, n_nodes=7,
└──[of=200, hp=[8, 6], id=[], lr=3,5]
    ├──[of=-1400, hp=[-50, 20], id=[], lr=1,2]
    │   ├──[of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1]
    │   └──[of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1]
    └──[of=1089, hp=[13, 11], id=[], lr=3,4]
        ├──[of=-1400, hp=[-50, 20], id=[], lr=1,2]
        │   ├──[of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1]
        │   └──[of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1]
        └──[of=3.40282e+38, hp=[], id=[11, 15], lr=-1,-1]
)







add_leaf:
indices=[0, 1, 4] ,n_leaves=1
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1]]

add_left_subtree:

add_leaf:
indices=[2, 3, 5, 7, 8, 9, 14] ,n_leaves=2
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1]]

add_left_subtree:

add_leaf:
indices=[6, 10, 12, 13] ,n_leaves=3
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1], [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1]]

add_right_subtree:

add_connecting_node:
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1], [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1], [of=-1400, hp=[-50, 20], id=[], lr=1,2]]

add_left_subtree:

add_leaf:
indices=[11, 15] ,n_leaves=4
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1], [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1], [of=-1400, hp=[-50, 20], id=[], lr=1,2], [of=3.40282e+38, hp=[], id=[11, 15], lr=-1,-1]]

add_right_subtree:

add_connecting_node:
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1], [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1], [of=-1400, hp=[-50, 20], id=[], lr=1,2], [of=3.40282e+38, hp=[], id=[11, 15], lr=-1,-1], [of=1089, hp=[13, 11], id=[], lr=3,4]]

add_right_subtree:

add_connecting_node:
nodes=[[of=3.40282e+38, hp=[], id=[0, 1, 4], lr=-1,-1], [of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1], [of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1], [of=-1400, hp=[-50, 20], id=[], lr=1,2], [of=3.40282e+38, hp=[], id=[11, 15], lr=-1,-1], [of=1089, hp=[13, 11], id=[], lr=3,4], [of=200, hp=[8, 6], id=[], lr=3,5]]
tree=
Tree(leaf_size=10, n_leaves=4, n_nodes=7,
└──[of=200, hp=[8, 6], id=[], lr=3,5]
    ├──[of=-1400, hp=[-50, 20], id=[], lr=1,2]
    │   ├──[of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1]
    │   └──[of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1]
    └──[of=1089, hp=[13, 11], id=[], lr=3,4]
        ├──[of=-1400, hp=[-50, 20], id=[], lr=1,2]
        │   ├──[of=3.40282e+38, hp=[], id=[2, 3, 5, 7, 8, 9, 14], lr=-1,-1]
        │   └──[of=3.40282e+38, hp=[], id=[6, 10, 12, 13], lr=-1,-1]
        └──[of=3.40282e+38, hp=[], id=[11, 15], lr=-1,-1]
