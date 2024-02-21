# AIRGen

Offical repository of [Arrange, Inpaint, and Refine: Steerable Long-term Music Audio Generation and Editing via Content-based Controls](https://arxiv.org/abs/2402.09508). You can visit our demo page [here](https://kikyo-16.github.io/AIR/).


## Quick Start

### Requirements
The project requires Python 3.11. See `requirements.txt`.

    conda create -n <environment-name> --file requirements.txt

### Inference

    python inference.py --output_folder=your_output_folder \
                        --model_path=your_model_weight_path \
                        --mode=fix_drums \
                        --audio_path=your_audio_path \
                        --midi_path=your_midi_path \
                        --chord_path=your_chord_path \
                        --beat_path=your_beat_path \
                        --drums_path=your_drums_path \
                        --onset=your_onset
See `demo_fm` folder for the input data format. Regarding `mode`, you can choose one from `{fix_drums, edit_drums, piano, chord}`. `Onset` should be a number indicating the starting second of the input audio.


### Model Weights
We provide with model weights (3.75M) with `m=10`. Download them via [drums](https://drive.google.com/file/d/163XEruv0vO9Pz24zsXz5qQz8lrCUZASD/view?usp=sharing), [chord](https://drive.google.com/file/d/1P2lKlA8s9T7FKwvdR-6kZBQ5aAiuIMcg/view?usp=sharing), and [piano](https://drive.google.com/file/d/1f4rMeSceAOQ_GBncEOXfMYkbN61Rui1K/view?usp=sharing). 

# How to cite
    @misc{lin2024arrange,
      title={Arrange, Inpaint, and Refine: Steerable Long-term Music Audio Generation and Editing via Content-based Controls}, 
      author={Liwei Lin and Gus Xia and Yixiao Zhang and Junyan Jiang},
      year={2024},
      eprint={2402.09508},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
    }
