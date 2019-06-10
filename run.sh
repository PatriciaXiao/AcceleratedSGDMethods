# for i in 1 2 3 4 5 6 7 8 9 10
for i in 1 2 3 4 5
do
    for lr in 0.03 0.003 0.0003 # 0.3 not work for all
    do
        python my_cnn.py -e 6 -a standard -lr $lr -d CIFAR -fo $i -logs
        python my_cnn.py -e 6 -a AGD -lr $lr -d CIFAR -fo $i -logs
        python my_cnn.py -e 6 -a GOSE -lr $lr -d CIFAR -fo $i -logs
        python my_cnn.py -e 6 -a ANCM -lr $lr -d CIFAR -fo $i -logs
        python my_cnn.py -e 6 -a combined -lr $lr -d CIFAR -fo $i -logs
    done
    for lr in 0.02 0.002 0.0002 # 0.2 not work for all
    do
        python my_cnn.py -e 6 -a standard -lr $lr -fo $i -logs
        python my_cnn.py -e 6 -a AGD -lr $lr -fo $i -logs
        python my_cnn.py -e 6 -a GOSE -lr $lr -fo $i -logs
        python my_cnn.py -e 6 -a ANCM -lr $lr -fo $i -logs
        python my_cnn.py -e 6 -a combined -lr $lr -fo $i -logs
    done
done