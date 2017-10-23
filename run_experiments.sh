# assigning the experiment a random ID
exp_id=1000
while [ -d $exp_id ]
do
    exp_id=$(($exp_id+1))
done
mkdir $exp_id


# getting all the parameters
runs=$1
data_path=$2
target_classes_path=$3
k=$4
max_queries=$5
init_size=$6

# in the following "i" is index of each run (run_id)
for ((i=1;i<=$runs;i++))
do
    mkdir "$exp_id/saved_models"
    mkdir "$exp_id/$i"
    run_model_path="$exp_id/saved_models/init-$i.ckpt"
    run_results_path="$exp_id/$i/results.dat"
    
    python run_querying_scr.py \
	$exp_id \
	$i \
	$data_path \
	$target_classes_path \
	$run_model_path \
	$k \
	$max_queries \
	$init_size
done