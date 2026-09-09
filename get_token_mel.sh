python tools/visualize_token_mel.py \
    --checkpoint /data1/xiaoqihezuo/hanxingwen/LITs/TEMP_REPO/logs/train/en-zh-pinyin_bs24_lr0.0001_pupu16k_mel_20260722_225057/runs/2026-07-22_22-50-57/checkpoints/checkpoint_epoch=1299.ckpt \
    --input_txt ./cmos_raw_text/en_cmos.txt \
    --model_lang en-zh-dict \
    --spk_id 2 \
    --output_dir ./infer_mel/shahenda