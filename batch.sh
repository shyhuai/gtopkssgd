#dnns=( "resnet20" "cifar10flnet" "mnistflnet" ) #"vgg16" "resnet110" )
dnns=( "resnet20" )
#compressors=( "gtopk" "topk" ) 
#compressors=( "gtopkr" ) 
#compressors=( "topk" ) 
#compressors=( "none" "topk" "gtopkr" ) 
compressors=( "gtopkr" ) 
#ns=( "32" "16" "8" "4" )
ns=( "32" )
density=0.001
for dnn in "${dnns[@]}"
do
    for nworkers in "${ns[@]}"
    do
        for compressor in "${compressors[@]}"
        do
            if [ "$compressor" = "none" ]; then 
                dnn=$dnn nworkers=$nworkers ./horovod_mpi.sh
            else
                dnn=$dnn density=$density nworkers=$nworkers compressor=$compressor ./gtopk_mpi.sh
            fi
        done
    done
done
