# for i in 1 2 3 4 5 6 7 8 9 10
for i in 1 2 3
do
    for lr in 0.3 0.03 0.003 0.0003
    do
        python my_cnn.py -e 6 -a standard -lr $lr -d CIFAR -fo $i
        python my_cnn.py -e 6 -a AGD -lr $lr -d CIFAR -fo $i
        python my_cnn.py -e 6 -a GOSE -lr $lr -d CIFAR -fo $i
        python my_cnn.py -e 6 -a ANCM -lr $lr -d CIFAR -fo $i
        python my_cnn.py -e 6 -a combined -lr $lr -d CIFAR -fo $i
    done
    for lr in 0.2 0.02 0.002 0.0002
    do
        python my_cnn.py -e 6 -a standard -lr $lr -fo $i
        python my_cnn.py -e 6 -a AGD -lr $lr -fo $i
        python my_cnn.py -e 6 -a GOSE -lr $lr -fo $i
        python my_cnn.py -e 6 -a ANCM -lr $lr -fo $i
        python my_cnn.py -e 6 -a combined -lr $lr -fo $i
    done
done