python mfa_zh354_duration/run.py \
  --run-name chuanyin_biaobei \
  --data-txt /data1/xiaoqihezuo/hanxingwen/LITs/process_data/CN9000_hanzi_filelist.txt \
  --wav-dir /data1/xiaoqihezuo/Dataset/zh/female/CN9000/wav \
  --mfa-dict ~/Documents/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict \
  --mfa-acoustic ~/Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip \
  --limit 10000 --align --stats