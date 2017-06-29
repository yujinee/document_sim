#!/bin/bash

for i in $( ls ./pdf ); do
  echo dir: $i
  for j in $( ls ./pdf/$i ); do
    echo item: $j
    pdftotext -layout pdf/$i/$j 1.txt
    python makedict.py > tmpdict
    cp tmpdict dictionary.dict
    echo finished
  done
done

