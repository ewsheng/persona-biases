# UnitPersonaBias
This is the repository to accompany the paper [Revealing Persona Biases in Dialogue Systems](https://arxiv.org/abs/2104.08728).
```
@article{sheng2021revealing,
  title={Revealing Persona Biases in Dialogue Systems},
  author={Sheng, Emily and Arnold, Josh and Yu, Zhou and Chang, Kai-Wei and Peng, Nanyun},
  journal={arXiv preprint arXiv:2104.08728},
  year={2021}
}
```

For questions, please contact Emily at ewsheng@gmail.

## Setup
```
conda create --name personabias python==3.7
conda activate personabias
pip install -r requirements.txt
```

### Offensiveness Test Setup
To run the offensiveness bias test on the Blended Skill Talk dataset, you'll have to download the dataset [here](https://parl.ai/projects/bst/) (in `bst.pkl` format).

To run this test on the RealToxicityPrompts dataset, you'll have to download the dataset [here](https://allenai.org/data/real-toxicity-prompts). 

### DialoGPT Setup
The persona-DialoGPT model we describe in the original paper (DialoGPT-medium fine-tuned on PersonaChat) can be found [here](https://drive.google.com/file/d/19TNVr1a4jDVOKHpkr5cUxLw3vBxlsrVL/view?usp=sharing).

If you wish to fine-tune your own model, you can follow the instructions [below](#fine-tuning-dialogpt-personas).


## Running Persona Bias Test Cases

### Offensiveness Biases

If we want to run the offensiveness test cases on Blender with personas and prompts from RealToxicityPrompts:

```
python offensiveness.py \
--model_type blender \
--include_persona \
--generator rtp
```

See `python offensiveness.py -h` for more options.

### Harmful Affirmation/Refutation Biases

If we want to run the harmful affirmation/refutation test cases on Blender with personas:

```
python harmful_affirm_refute.py \
--model_type blender \
--include_persona
```

See `python harmful_affirm_refute.py -h` for more options.

### Occupational Association Biases

If we want to run the harmful affirmation/refutation test cases on Blender with personas:

```
python occupational_assoc.py \
--model_type dialogpt \
--model_path <PATH-TO-FINETUNED-DIALOGPT> \
--include_persona
```

See `python occupational_assoc.py -h` for more options.

### Gendered Coreference Biases

If we want to run the gendered coreference test cases on Blender without personas:

```
python gendered_coref.py \
--model_type blender
```

See `python gendered_coref.py -h` for more options.


## Fine-tuning DialoGPT Personas

To fine-tune DialoGPT on the PersonaChat dataset, we must download [PersonaChat](https://github.com/facebookresearch/ParlAI/tree/master/projects/personachat) by running the following:
```
parlai display_data --task personachat --datatype train
parlai display_data --task personachat --datatype valid
```
This will download `train_self_original.txt` and `valid_self_original.txt` files and print out the path to these files.

Next, run ```run_language_modeling.py``` to start the fine-tuning process.
For example:
```
python run_language_modeling.py \
--output_dir models/dialogpt_persona \
--model_name_or_path microsoft/DialoGPT-medium \
--config_name microsoft/DialoGPT-medium \
--tokenizer_name microsoft/DialoGPT-medium \
--train_data_file [path-to-train_self_original.txt] \
--eval_data_file [path-to-valid_self_original.txt] \
--do_eval \
--line_by_line \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--num_train_epochs 5 \
--learning_rate 5e-6 \
--do_train \
--block_size 128 
```
This fine-tunes DialoGPT on the PersonaChat data by concatenating all personas and different dialogue turns in the following format:
```
your persona: [personaA] your persona: [personaB] your persona: [personaC] your persona: [personaD] [EOS] [PERSON1 TEXT] [EOS] [PERSON2 TEXT] [EOS]
```
