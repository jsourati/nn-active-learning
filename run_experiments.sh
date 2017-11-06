# getting all the parameters
root_dir=$1
par_temp=$2
data_dir=$3
target_classes=$4
run_id=$5
run_num=$6
nqueries=$7
chunk=$8
optpars=$9

echo "Input Directories"
echo "================================="
echo "Root dir: $root_dir"
echo "Par. template: $par_temp"
echo "Data dir: $data_dir"
echo "Target classes: $target_classes"
echo "Input Parameters"
echo "================================="
echo "Run ID: $run_id"
echo "# Runs: $run_num"
echo "# queries: $nqueries"
echo "Iter-chunk: $chunk"
echo "Parameters: $optpars"
echo "================================="

# if this experiment does not exist, 
# create one and set the parameters
if [ ! -d $root_dir ]; then

    echo "Creating experiment in:"
    echo $root_dir

    # creating the experiment
    python -c "from expr_handler import create_expr; create_expr('$root_dir', '$data_dir', '$target_classes')"

    
    # setting up the parameters
    python -c "from expr_handler import set_parameters; set_parameters('$par_temp', '$root_dir', '$optpars')"
fi

# checking if the given run-ID is a number
# or a plust ('+') as a sign to add a run
re='^[0-9]+$'
if [[ $run_id =~ $re ]]; then
    # if so, do not run more tan once
    run_num=1
fi

for ((i=1;i<=$run_num;i++))
do
    # and now run the experiment
    M=(fi random entropy rep-entropy)
    
    # if it's a new run create it
    if ! [[ $run_id =~ $re ]]; then
	# creating the new run in the experiment
	python -c "from expr_handler import create_run; create_run('$root_dir')"
	
	# assiginig the run ID 
	run_id=`python -c "import AL;E=AL.Experiment('$root_dir');print(len(E.get_runs())-1)"`
	echo ""
	echo "This new run's ID is $run_id"
    fi

    # for each method run the querying iterations
    # chunk-by-chunk
    for method in ${M[@]}
    do
	Q=0
	while [ $Q -lt $(($nqueries)) ]
	do
	    echo "Running method $method"
	    python expr_handler.py \
		$root_dir \
		$run_id \
		$method \
		$chunk
	    
	    # update the counter
	    Q=$(($Q+$chunk))
	done
    done
    # if the iteration is supposed to repeat
    # turn "run_id" back to '+' symbol
    run_id=+
done
