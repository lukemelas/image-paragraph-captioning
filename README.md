## Training for Diversity in Image Paragraph Captioning

This repository includes a PyTorch implementation of [Training for Diversity in Image Paragraph Captioning](). Our code is based on Ruotian Luo's implementation of [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563), available [here.](https://github.com/ruotianluo/self-critical.pytorch). 

### Requirements
* Python 2.7 (because coco-caption does not support Python 3)
* PyTorch 0.4 (with torchvision)
* cider (already included as a submodule)
* coco-caption (already included as a submodule)

If training from scratch, you also need:
* spacy (to tokenize words)
* h5py (to store features)
* scikit-image (to process images) 

To clone this repository with submodules, use:
* `git clone --recurse-submodules https://github.com/lukemelas/image-paragraph-captioning`

### Train your own network
#### Download and preprocess cations

* Download captions:
  *  Run `download.sh` in `data/raw_captions`
* Preprocess captions for training (part 1): 
  * Download `spacy` English tokenizer with `python -m spacy download en`
  * First, convert the text into tokens: 'scripts/prepro\_text.py'
  * Next, preprocess the tokens into a vocabulary (and map infrequent words to an `UNK` token) with the following command. Note that image/vocab information is stored in `data/paratalk.json` and caption data is stored in `data/paratalk\_label.h5`
```bash
python scripts/prepro_labels.py --input_json data/captions/para_karpathy_format.json --output_json data/paratalk.json --output_h5 data/paratalk
```
* Preprocess captions into a coco-captions format for calculating CIDER/BLEU/etc: 
  *  Run `scripts/prepro\_captions.py`
  *  There should be 14,575/2487/2489 images and annotations in the train/val/test splits
  *  Uncomment line 44 (`(Spice(), "SPICE")`) in `coco-caption/pycocoevalcap/eval.py` to disable Spice testing
* Preprocess ngrams for self-critical training:
```bash
python scripts/prepro_ngrams.py --input_json data/captions/para_karpathy_format.json --dict_json data/paratalk.json --output_pkl data/para_train --split train
```
* Extract image features using an object detector
  * We make pre-processed features widely available:
    * Download and extract `parabu_fc` and `parabu_att` from [here](https://drive.google.com/drive/folders/1lgdHmU6osXt4BObnhHS6tPqnkedwnHLD?usp=sharing) into `data/bu_data` 
  * Or generate the features yourself:
    * Download the [Visual Genome Dataset](https://visualgenome.org/api/v0/api_home.html)
    * Apply the bottom-up attention object detector [here](https://github.com/peteanderson80/bottom-up-attention) made by Peter Anderson.
    * Use `scripts/make_bu_data.py` to convert the image features to `.npz` files for faster data loading

#### Train the network

As explained in [Self-Critical Sequence Training](https://arxiv.org/abs/1612.00563), training occurs in two steps:
1. The model is trained with a cross-entropy loss (30 epochs)
2. The model is trained with a self-critical loss (30+ epochs)

Training hyperparameters may be accessed with `python main.py --help`. 

For example, the following command trains with cross-entropy:
```bash 
python train.py --id xe --input_json data/paratalk.json --input_fc_dir data/parabu_fc --input_att_dir data/parabu_att --input_label_h5 data/paratalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_xe --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30
```

You can then copy the model:
```bash
./scripts/copy_model.sh xe sc
```

And train with self-critical:
```bash
./scripts/copy_model.sh xe sc
```

```bash
python train.py --id sc --input_json data/paratalk.json --input_fc_dir data/parabu_fc --input_att_dir data/parabu_att --input_label_h5 data/paratalk_label.h5 --batch_size 16 --learning_rate 5e-5 --start_from log_sc --checkpoint_path log_sc --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 28
```




<!--

```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

### Download COCO dataset and pre-extract the image features (Skip if you are using bottom-up feature)

Download the coco images from [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images. You should put the `train2014/` and `val2014/` in the same directory, denoted as `$IMAGE_ROOT`.

Then:

```
$ python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT
```


`prepro_feats.py` extract the resnet101 features (both fc feature and last conv feature) of each image. The features are saved in `data/cocotalk_fc` and `data/cocotalk_att`, and resulting files are about 200GB.

(Check the prepro scripts for more options, like other resnet models or other attention sizes.)

**Warning**: the prepro script will fail with the default MSCOCO data because one of their images is corrupted. See [this issue](https://github.com/karpathy/neuraltalk2/issues/4) for the fix, it involves manually replacing one image in the dataset.

### Download Bottom-up features (Skip if you are using resnet features)

Download pre-extracted feature from [link](https://github.com/peteanderson80/bottom-up-attention). You can either download adaptive one or fixed one.

For example:
```
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip

```

Then:

```bash
python script/make_bu_data.py --output_dir data/cocobu
```

This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`. If you want to use bottom-up feature, you can just follow the following steps and replace all cocotalk with cocobu.

### Start training

```bash
$ python train.py --id fc --caption_model fc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `save/`). We only save the best-performing checkpoint on validation and the latest checkpoint to save disk space.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

If you have tensorflow, the loss histories are automatically dumped into `--checkpoint_path`, and can be visualized using tensorboard.

The current command use scheduled sampling, you can also set scheduled_sampling_start to -1 to turn off scheduled sampling.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.

For more options, see `opts.py`. 

**A few notes on training.** To give you an idea, with the default settings one epoch of MS COCO images is about 11000 iterations. After 1 epoch of training results in validation loss ~2.5 and CIDEr score of ~0.68. By iteration 60,000 CIDEr climbs up to about ~0.84 (validation loss at about 2.4 (under scheduled sampling)).

### Train using self critical

First you should preprocess the dataset and get the cache for calculating cider score:
```
$ python scripts/prepro_ngrams.py --input_json .../dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

Then, copy the model from the pretrained model using cross entropy. (It's not mandatory to copy the model, just for back-up)
```
$ bash scripts/copy_model.sh fc fc_rl
```

Then
```bash
$ python train.py --id fc_rl --caption_model fc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-5 --start_from log_fc_rl --checkpoint_path log_fc_rl --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 30
```

You will see a huge boost on Cider score, : ).

**A few notes on training.** Starting self-critical training after 30 epochs, the CIDEr score goes up to 1.05 after 600k iterations (including the 30 epochs pertraining).

### Caption images after training

## Generate image captions

### Evaluate on raw images
Now place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

```bash
$ python eval.py --model model.pth --infos_path infos.pkl --image_folder blah --num_images 10
```

This tells the `eval` script to run up to 10 images from the given folder. If you have a big GPU you can speed up the evaluation by increasing `batch_size`. Use `--num_images -1` to process all images. The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```bash
$ cd vis
$ python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser and you should see your predicted captions.

### Evaluate on Karpathy's test split

```bash
$ python eval.py --dump_images 0 --num_images 5000 --model model.pth --infos_path infos.pkl --language_eval 1 
```

The defualt split to evaluate is test. The default inference method is greedy decoding (`--sample_max 1`), to sample from the posterior, set `--sample_max 0`.

**Beam Search**. Beam search can increase the performance of the search for greedy decoding sequence by ~5%. However, this is a little more expensive. To turn on the beam search, use `--beam_size N`, N should be greater than 1.

## Miscellanea
**Using cpu**. The code is currently defaultly using gpu; there is even no option for switching. If someone highly needs a cpu model, please open an issue; I can potentially create a cpu checkpoint and modify the eval.py to run the model on cpu. However, there's no point using cpu to train the model.

**Train on other dataset**. It should be trivial to port if you can create a file like `dataset_coco.json` for your own dataset.

**Live demo**. Not supported now. Welcome pull request.

## For more advanced features:

Checkout `ADVANCED.md`.

## Reference

If you find this repo useful, please consider citing (no obligation at all):

```
@article{luo2018discriminability,
  title={Discriminability objective for training descriptive captions},
  author={Luo, Ruotian and Price, Brian and Cohen, Scott and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:1803.04376},
  year={2018}
}
```

Of course, please cite the original paper of models you are using (You can find references in the model files).

## Acknowledgements

Thanks the original [neuraltalk2](https://github.com/karpathy/neuraltalk2) and awesome PyTorch team.

-->
