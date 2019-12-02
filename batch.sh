dnns=( "resnet20" "vgg16" )
compressors=( "gtopk" "topk" ) 
ns=( "32" "16" "8" "4" )
density=0.001
for dnn in "${dnns[@]}"
do
    for nworkers in "${ns[@]}"
    do
        for compressor in "${compressors[@]}"
        do
            dnn=$dnn density=$density nworkers=$nworkers compressor=$compressor ./gtopk_mpi.sh
        done
    done
done
