source ~/morai-student/env.sh && export PYTHONPATH=$PYTHONPATH:$HOME/morai-student/pylib
echo "== forced stop $(date +%T)"
S1EXTRA="+callbacks.traj_early_stop.min_epochs=0 +callbacks.traj_early_stop.patience=1 +callbacks.traj_early_stop.rel_improve=1.0" STAGES="s1 sgate" bash scripts/student_dryrun.sh $RUNS/dry_es $D 40
echo "== rule as is $(date +%T)"
STAGES="s1" bash scripts/student_dryrun.sh $RUNS/dry_es2 $D 40
echo "== pytest $(date +%T)"
python -m pytest -q tests 2>&1 | tail -5
echo "== done $(date +%T)"
