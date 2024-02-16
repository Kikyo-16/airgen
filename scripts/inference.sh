CUDA_VISIBLE_DEVICES=0 python inference.py --output_folder="demo/output/" \
                                            --model_path="model_weights/drums.pth" \
                                            --mode=drums \
                                            --audio_path="test.mp3" \
                                            --midi_path="test.mid" \
                                            --chord_path="chord_audio.txt" \
                                            --beat_path="beat_audio.txt" \
                                            --drums_path="drums.mp3" \
                                            --onset=0
