# model=Deepseek-Coder-chat
model=Llama3-chat
scale=8b

for DATASET in WikiTQ; do
        for indexer in md dict list pd db; do
                for SPLIT in test; do
                        data_file=./dataset/$DATASET/$SPLIT.list.json

                        if { [ $indexer = "list" ]; } || { [ $indexer = "dict" ]; } || { [ $indexer = "pd" ]; } || { [ $indexer = "db" ]; }; then
                                agent=pl
                                echo "table indexer is $indexer, with $agent"
                        fi
                        if { [ $indexer = "csv" ]; } ||  { [ $indexer = "md" ]; } || { [ $indexer = "text" ]; } || { [ $indexer = "tsv" ]; } || { [ $indexer = "tuple" ]; } || { [ $indexer = "htmlns" ]; } || { [ $indexer = "html" ]; }; then
                                agent=nl
                                echo "table indexer is $indexer, with $agent"
                        fi

                        python3 ./reason/inference.py \
                                --llm_name_or_path ./model/$model/$scale \
                                --config_path ./config/$model.json \
                                --prompt_path ./inference/prompts/$model \
                                --data_file $data_file \
                                --dump_path ./inference/results/$DATASET/$model/$scale/$agent/$indexer/$SPLIT.json \
                                --prompt_type $indexer \
                                # --data_size 128 \
                                # --save_n true \
                done
        done
done