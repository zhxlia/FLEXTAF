
for i in {low..high..interval}
do
        python3 ./classify/multi_label_classify.py \
                --model_name_or_path ./classify/ckpt/Llama3-chat/8b/checkpoint-$i \
                --tokenizer_name_or_path ./model/Electra/large \
                --test_data_file ./reason/results/WikiTQ/Llama3-chat/8b/compare/compare.test.json \
                --dump_path ./classify/result/Electra/Llama3-chat/8b/test.c$i.json \
                --representation_num 5 \
                --output_log true \
                --marked true
done
