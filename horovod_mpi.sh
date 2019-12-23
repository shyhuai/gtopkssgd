#!/bin/bash
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
nwpernode=4
nstepsupdate=1
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
#PY=python
PY=/usr/local/bin/python
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include em1 \
    $PY horovod_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode 
