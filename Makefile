PY := .venv/bin/python -E -s   # -E: ignore PYTHONPATH (ROS on it breaks imports)
.PHONY: download prepare selftest
download: ; $(PY) -c "from huggingface_hub import snapshot_download as s; s('dgural/bdd100k', repo_type='dataset', revision='c2e7f266756bcd07b87f1a45a35937c8eac20241', local_dir='data/raw', max_workers=16, allow_patterns=['data/*', 'samples.json'])"
prepare:  ; $(PY) prepare.py
selftest: ; $(PY) selftest.py
.PHONY: selftest_eval baseline start stop status
selftest_eval: ; $(PY) selftest_eval.py
baseline: ; for s in day_test night_test dusk_test; do $(PY) evaluate.py --weights weights/yolo11s.pt --coco --split $$s || exit 1; done
# Whole queue in the background; survives closing the terminal. `make start` again resumes.
start:  ; mkdir -p runs && JOBS="$(JOBS)" setsid nohup ./queue.sh >> runs/queue.out 2>&1 & echo started
stop:   ; pkill -TERM -f "^bash ./queue.sh" ; pkill -TERM -f "^.venv/bin/python -E -s -u (train|evaluate).py" ; echo stopped
status: ; ls runs/*/DONE 2>/dev/null; grep -a -o "[0-9]*/40 .\{0,40\}" runs/queue.out | tail -1; ls runs/eval/*.json 2>/dev/null | wc -l; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
