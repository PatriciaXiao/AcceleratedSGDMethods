for lr in 0.3 0.03 0.003 0.0003
do
    python my_cnn.py -e 6 -a standard -fo $lr
    python my_cnn.py -e 6 -a AGD -fo $lr
    python my_cnn.py -e 6 -a GOSE -fo $lr
    python my_cnn.py -e 6 -a ANCM -fo $lr
    python my_cnn.py -e 6 -a combined -fo $lr
done