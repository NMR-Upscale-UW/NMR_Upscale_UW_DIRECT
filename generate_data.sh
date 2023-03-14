rm -r data/400MHz/*.csv
python preprocessing/nmrgenerate.py \
--low_resolution 60 \
--high_resolution 400 \
--num_spec 100 \
--data_dir "./data" \
