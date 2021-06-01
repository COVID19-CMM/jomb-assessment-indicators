#!/usr/bin/bash

N=500

python simulate.py --indicador 1 --metodo a -n $N --umbral 0 8000
python simulate.py --indicador 1 --metodo b -n $N --umbral 0 120
python simulate.py --indicador 1 --metodo c -n $N --umbral 0 1
python simulate.py --indicador 1 --metodo d -n $N --umbral 0 100
python simulate.py --indicador 2 --metodo a -n $N --umbral 0 3500
python simulate.py --indicador 2 --metodo b -n $N --umbral 0 48
python simulate.py --indicador 2 --metodo c -n $N --umbral 0 1
python simulate.py --indicador 2 --metodo d -n $N --umbral 0 300
