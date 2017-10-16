# getting all the parameters
runs=$1
data_path=$2
model_save_path=$3
results_save_path=$4
k=$5
max_queries=$6
init_size=$7

for ((i=1;i<=$runs-1;i++))
do
    iter_model_path="$model_save_path-$i.ckpt"
    iter_results_path="$results_save_path-$i.dat"

    python run_querying_scr.py \
	$data_path \
	$iter_model_path \
	$iter_results_path \
	$k \
	$max_queries \
	$init_size

done