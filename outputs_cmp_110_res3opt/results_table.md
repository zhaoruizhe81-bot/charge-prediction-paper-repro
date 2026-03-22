| family | model | variant | source | accuracy | f1_macro | f1_micro | f1_weighted |
|---|---|---|---|---|---|---|---|
| ML | svm | base | ml_baselines | 0.792138 | 0.782881 | 0.792138 | 0.790053 |
| ML | sgd | base | ml_baselines | 0.803309 | 0.785663 | 0.803309 | 0.800561 |
| ML | pa | base | ml_baselines | 0.791148 | 0.779638 | 0.791148 | 0.788369 |
| ML | svm | hier_fusion_opt | ml_baselines+weighted | 0.796521 | 0.782331 | 0.796521 | 0.794484 |
| ML | sgd | hier_fusion | ml_baselines | 0.804299 | 0.786613 | 0.804299 | 0.801368 |
| ML | pa | hier_fusion | ml_baselines | 0.792986 | 0.783382 | 0.792986 | 0.789995 |
| DL | fc | flat | deep_hierarchical_fc_c2r_tune1 | 0.829327 | 0.794125 | 0.829327 | 0.824127 |
| DL | fc | hier_fusion | deep_hierarchical_fc_c2r_tune1 | 0.829468 | 0.795024 | 0.829468 | 0.824404 |
| DL | rcnn | flat | deep_hierarchical_rcnn_c2r | 0.812500 | 0.750947 | 0.812500 | 0.804401 |
| DL | rcnn | hier_fusion_opt | deep_hierarchical_rcnn_c2r+sweep | 0.815469 | 0.759683 | 0.815469 | 0.808626 |