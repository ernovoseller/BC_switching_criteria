#! /bin/sh

python setup.py install

cd run_policies
python analytic.py oracle --max_episodes 2000 --seed 1336 --tier 3
cd ../

