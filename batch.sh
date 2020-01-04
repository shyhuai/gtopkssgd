dnns=( "resnet20" "vgg16" "resnet110" )
#dnns=( "resnet110" )
#compressors=( "gtopk" "topk" ) 
#compressors=( "gtopkr" ) 
#compressors=( "topk" ) 
#compressors=( "none" "topk" "gtopkr" ) 
compressors=( "none" ) 
#ns=( "32" "16" "8" "4" )
ns=( "8" )
density=0.001
lr=0.2
for dnn in "${dnns[@]}"
do
    for nworkers in "${ns[@]}"
    do
        for compressor in "${compressors[@]}"
        do
            if [ "$compressor" = "none" ]; then 
                lr=$lr dnn=$dnn nworkers=$nworkers ./horovod_mpi.sh
            else
                dnn=$dnn density=$density nworkers=$nworkers compressor=$compressor ./gtopk_mpi.sh
            fi
        done
    done
done
