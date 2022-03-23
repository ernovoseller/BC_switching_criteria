#! /bin/sh

python setup.py install

cd run_policies
python eval_policies.py --max_episodes 5 --seed 10 --tier 3 --start_idx 199 --end_idx 200
cd ../

