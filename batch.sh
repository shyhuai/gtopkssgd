#dnns=( "resnet20" "cifar10flnet" "mnistflnet" ) #"vgg16" "resnet110" )
#dnns=( "vgg16" )
dnns=( "resnet20" )
#compressors=( "gtopk" "topk" ) 
compressors=( "topk" ) 
#compressors=( "topkdense" ) 
#compressors=( "none" "topk" "gtopkr" ) 
#compressors=( "gtopkr" ) 
#ns=( "32" "16" "8" "4" )
ns=( "32" )
density=0.001
lr=0.141
#bs=32
for dnn in "${dnns[@]}"
do
    for nworkers in "${ns[@]}"
    do
        for compressor in "${compressors[@]}"
        do
            if [ "$compressor" = "none" ]; then 
                lr=$lr dnn=$dnn nworkers=$nworkers ./horovod_mpi.sh
            else
                lr=$lr dnn=$dnn density=$density nworkers=$nworkers compressor=$compressor ./gtopk_mpi.sh
            fi
        done
    done
done
