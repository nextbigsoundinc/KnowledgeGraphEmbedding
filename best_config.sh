# CoCoE
bash run_v2.sh train CoCoE FB15k-237 0 20 128 256 200 9.0 1.0 0.003 100000 16 -de -dr -r 0.00001
bash run_v2.sh train CoCoE wn18 1 28 128 256 200 9.0 1.0 0.003 80000 8 -de -dr -r 0.00001
# TransE
bash run.sh train TransE FB15k-237 1 21 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE wn18 0 23 512 1024 500 12.0 0.5 0.0001 80000 8
# ComplEx    #
bash run.sh train ComplEx FB15k-237 0 25 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train ComplEx wn18 1 26 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
# RotatE
bash run.sh train RotatE FB15k-237 1 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
bash run.sh train RotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de
# pRotatE
bash run.sh train pRotatE FB15k-237 1 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train pRotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
# DistMult
bash run.sh train DistMult FB15k-237 1 0 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001
bash run.sh train DistMult wn18 0 0 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001



#bash run.sh train ConvE FB15k-237 0 1 128 256 200 9.0 1.0 0.000002 100000 16
#bash run.sh train ConvE FB15k-237 0 2 128 256 200 9.0 1.0 0.000005 100000 16
#bash run.sh train ConvE FB15k-237 0 3 128 256 200 9.0 1.0 0.00001 100000 16
#bash run.sh train ConvE FB15k-237 1 4 128 256 200 9.0 1.0 0.00002 100000 16
#bash run.sh train ConvE FB15k-237 1 5 128 256 200 9.0 1.0 0.00005 100000 16
#bash run.sh train ConvE FB15k-237 1 6 128 256 200 9.0 1.0 0.0001 100000 16
#
#bash run.sh train ConvE FB15k-237 0 10 128 256 200 9.0 1.0 0.0001 100000 16
#
#bash run.sh train ConvE FB15k-237 0 12 128 256 200 9.0 1.0 0.003 10000 128
#
#
#bash run.sh train CoCo FB15k-237 0 13 128 256 200 9.0 1.0 0.003 10000 128
#bash run.sh train CoCo FB15k-237 0 14 128 256 200 9.0 1.0 0.001 10000 128
#bash run.sh train CoCo FB15k-237 0 15 128 256 200 9.0 1.0 0.002 10000 128
#bash run.sh train CoCo FB15k-237 0 16 128 256 200 9.0 1.0 0.004 10000 128
#bash run.sh train CoCo FB15k-237 0 17 128 256 200 9.0 1.0 0.001 20000 128
#
#bash run.sh train ConvE FB15k-237 0 18 128 256 200 9.0 1.0 0.003 20000 128
#
#bash run.sh train CoCo2 FB15k-237 0 19 128 256 200 9.0 1.0 0.001 40000 128
#
#bash run.sh train ConvE FB15k-237 0 20 128 256 200 9.0 1.0 0.003 40000 128
#
#
## TransE
#bash run.sh train TransE FB15k 0 22 1024 256 1000 24.0 1.0 0.0001 150000 16
#bash run.sh train TransE FB15k-237 1 21 1024 256 1000 9.0 1.0 0.00005 100000 16
#bash run.sh train TransE wn18 0 23 512 1024 500 12.0 0.5 0.0001 80000 8
## ComplEx
#bash run.sh train ComplEx FB15k 0 24 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002     #
#bash run.sh train ComplEx FB15k-237 0 25 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
#bash run.sh train ComplEx wn18 1 26 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
## ConvE
#bash run.sh train ConvE FB15k 1 27 128 256 200 9.0 1.0 0.003 40000 128
#bash run.sh train ConvE FB15k-237 0 20 128 256 200 9.0 1.0 0.003 40000 128
#bash run.sh train ConvE wn18 1 28 128 256 200 9.0 1.0 0.003 40000 128
#
#
## CoCo2
#bash run.sh train CoCo2 FB15k 0 29 1024 256 200 9.0 1.0 0.003 40000 16
#bash run.sh train CoCo2 FB15k 1 30 1024 256 200 9.0 1.0 0.001 40000 16
#bash run.sh train CoCo2 FB15k 0 31 1024 256 200 9.0 1.0 0.0005 40000 16
#bash run.sh train CoCo2 FB15k 1 32 1024 256 200 9.0 1.0 0.0002 40000 16
#
#
#bash run.sh train CoCo2 FB15k-237 0 33 1024 256 200 9.0 1.0 0.003 40000 16
#bash run.sh train CoCo2 FB15k-237 0 34 1024 256 200 9.0 1.0 0.001 40000 16
#bash run.sh train CoCo2 FB15k-237 1 35 1024 256 200 9.0 1.0 0.0005 40000 16
#bash run.sh train CoCo2 FB15k-237 1 36 1024 256 200 9.0 1.0 0.0002 40000 16
#
#
#bash run.sh train CoCo2 wn18 1 37 1024 256 200 9.0 1.0 0.003 40000 16
#bash run.sh train CoCo2 wn18 1 38 1024 256 200 9.0 1.0 0.001 40000 16
#bash run.sh train CoCo2 wn18 1 39 1024 256 200 9.0 1.0 0.0005 40000 16
#bash run.sh train CoCo2 wn18 1 40 1024 256 200 9.0 1.0 0.0002 40000 16
#
#
#
#bash run.sh train ComplEx wn18 0 41 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
#
#bash run_v2.sh train model dataset 0 80 1024 256 200
#
#
#
## bash run.sh train RotatE FB15k    0       0      1024        256               1000         24.0    1.0   0.0001 150000         16               -de
##               1     2      3       4      5        6          7                   8          9       10     11     12           13
##              mode model  dataset  GPU  saveid    batchsize   neg_sample_size  hidden_dim    gamma   alpha   lr    Max_steps  test_batchsize
#
#
#bash run.sh train ConvE FB15k-237 1 0 128 256 200 9.0 1.0 0.00002 100000 16
#
## Best Configuration for RotatE
##
#bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 -de                  #
#bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
#bash run.sh train RotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de
#bash run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
#bash run.sh train RotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
#bash run.sh train RotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
#bash run.sh train RotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
#bash run.sh train RotatE YAGO3-10 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de
##
## Best Configuration for pRotatE
##
#bash run.sh train pRotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16
#bash run.sh train pRotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
#bash run.sh train pRotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
#bash run.sh train pRotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
#bash run.sh train pRotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#bash run.sh train pRotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#bash run.sh train pRotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
##
## Best Configuration for TransE
##
#bash run.sh train TransE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16                    #
#bash run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
#bash run.sh train TransE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
#bash run.sh train TransE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
#bash run.sh train TransE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#bash run.sh train TransE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#bash run.sh train TransE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
##
## Best Configuration for ComplEx
##
#bash run.sh train ComplEx FB15k 0 0 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002     #
#bash run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
#bash run.sh train ComplEx wn18 0 0 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
#bash run.sh train ComplEx wn18rr 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
#bash run.sh train ComplEx countries_S1 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
#bash run.sh train ComplEx countries_S2 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
#bash run.sh train ComplEx countries_S3 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
##
## Best Configuration for DistMult
##
#bash run.sh train DistMult FB15k 0 0 1024 256 2000 500.0 1.0 0.001 150000 16 -r 0.000002
#bash run.sh train DistMult FB15k-237 0 0 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001
#bash run.sh train DistMult wn18 0 0 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001
#bash run.sh train DistMult wn18rr 0 0 512 1024 1000 200.0 1.0 0.002 80000 8 -r 0.000005
#bash run.sh train DistMult countries_S1 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
#bash run.sh train DistMult countries_S2 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
#bash run.sh train DistMult countries_S3 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
##
