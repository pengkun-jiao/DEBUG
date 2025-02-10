experiment_name="debug"

config='pacs_uresnet18_gsum'

# domains = ['art_painting', 'cartoon', 'sketch', 'photo']

source='art_painting'
nohup python shell_train.py --gpu 0 --config $config --name $experiment_name --source $source  > output1 2>&1   &
source='cartoon'
nohup python shell_train.py --gpu 1 --config $config --name $experiment_name --source $source  > output3 2>&1   &
source='sketch'
nohup python shell_train.py --gpu 0 --config $config --name $experiment_name --source $source  > output2 2>&1   &
source='photo'
nohup python shell_train.py --gpu 1 --config $config --name $experiment_name --source $source  > output4 2>&1   &



