This is the repository of the paper entitiled ***Gotta Embed Them All! - Knowledge-aware Recommendations Combining Heterogeneous Multi-Modal Item Embeddings***

In this repo, we have uploaded all the files necessary to reproduce our experiments.
First, let us discuss the content and the structure of this repository.

## Structure of the repository

### Scraping audio/video (A/V) data folder
In the folder `scraping_av_data/`, we have have provided the scripts we used to scrape A/V data for both the datasets.

The scripts `scraper_img_dbbook.py` and `scraper_img_dbbook.py` retrieve images of movie posters and book covers for Movielens-1M and DBbook, respectively, startig from their DBpedia URL (available in the mappings publicly available).

Similarly, the script `scraper_trailer_ml1m.py` scrapes the movie trailer for movies in Movielens-1M. Note that, in this case, a lot of effort has been manually performed, since a lot of them are not on the same platform, or are temporarily removed.

Thanks to these scripts, we are able to get the extended versions of the two datasets; due to copyright law, we won't provide the raw files (images and videos), but we provide the mappings and the embeddings related to that A/V content.

### Embedding learning folder 
In the folder `embedding_learning/`, we have provided the files necessary to learn all the embeddings we use in our model, namely: `graphs`, `text`, `images`, `video`, `audio`.

In the `embedding_learning/graph/` folder there are the graphs for both the dataset, in both the forms `user-item` (no KGs) and `user-item-prop` (with KGs); of course, you can also find the script to run the Knowledge Graph Embeddings.

In the `embedding_learning/text/` folder, the script `learn_text_embs.py` is used to learn text embeddings through the technique we used (`SBERT` implementation available in `MiniLM`); this script uses the text available for the two dataset, but due to their size, we have provided a download link. Alternatively, you can find the plain text in the repository related to the dataset we have linked in the paper.
The script `build_user_profiles.py` is used to learn user embeddings starting from the text embedding, so it basically computes the centroid of the liked items for each users, as we described in our paper.

The `embedding_learning/images/` folder contains the script to learn the image embeddings. In this case, due to copyright issue, we cannot share the raw files (images), but we have provided mapping file (files `dbbook/ml1m_extended_mapping.tsv` in the folder `ml1m_or_dbbook_images/`) so that you can download them and run the scripts; we have considered the models `ViT` (used in our paper), `VGG` and `ResNet152`. Similarly to the `text` folder, we provided a script to learn user emeddings with the centroid technique.

The `embedding_learning/video/` folder contains the script to learn audio embeddings; also in this case, due to copyright, we cannot share raw files but provided mapping to the link of the trailer of movies; again, we provided a script to learn user embeddings with the centroid technique.

The `embedding_learning/audio/` folder contains the script to learn audio embeddings; also in this case, due to copyright, we cannot share raw files bur provided mapping (in this case, the audio is taken from the trailer of the movie, so this hold only for `ML1M`); again, we provided a script to learn user embeddings with the centroid technique.

### Embeddings folder
In this folder, we have put a link to downloadd ***ALL*** the embeddings (for both the datasets) we have learnt in our work, in order to replicate the experiments. We have put the link due to high size of the resulting file, and we will weekly check weather the link works (in case of error, we will update it).

### GETAll folder

This folder contains the source code of our model.

In the `data` folder we insert both `raw` data (`train.tsv` and `test.tsv` files), and the `embeddings` learnt in `.pkl` format, for each dataset and for each source; we have provided the link to download them also in this folder.

The `report` folder is used to save both the trained recommendation models in format `.pth` (folder `models`), and the prediction lists (in the folder `predictions`). 

Finally, the `src` contains the source code for **GETAll!** recommendation model.
We have implemented different classes, one for each number of sources used (from single source to five source), and the dataset handlet class; moreover, we provided two script, `train_movielens.py` and `train_dbbook.py`, already written so that it is possible to replicate the results we have presented in our paper. 

### Baseline Settings folder

In this folder we have provided the setting files we used to fine-tune the baselines considered in the paper.
Since we used two State-of-the-art recommendation framework ([RecBole](https://github.com/RUCAIBox/RecBole) and [Formal-Multimodal-Recsys](https://github.com/sisinflab/Formal-MultiMod-Rec/tree/main)), it is easy to provide these settings file and generate prediction lists using the same `test.tsv` file we used to replicate also these experiments.

### Evaluation folder
In this folder, have provided three files: since we used [ClayRS](https://swapuniba.github.io/ClayRS/) to evaluate the recommendation lists, we have provided the scripts to perform the evaluation of the prediction lists (those produced by GETAll, those produced by the baseline models, and those for the sentivity analysis, including all the needed statistical tests), with the names `eval_rq1_rq2.py`, `eval_rq3.py`, `ttest_rq1_rq2.py`, `ttest_rq3.py` file we used, and the `clayrs_req.txt` needed to perform the evaluation.

## Setting the environment

To replicate our experiment, here we provide some characteristics of our environment.
We have performed our experiments on a `Ubuntu 20.04.4 LTS (GNU/Linux 5.15.0-91-generic x86_64)` with NVIDIA GTX TITAN X (12 GB) as GPU; moreover, we provide here the output or the `nvcc --verison` command:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```

As for the `nvidia-smi` output:
```
NVIDIA-SMI 470.223.02   Driver Version: 470.223.02   CUDA Version: 11.4
```

Finally, we have performed all the experiments using `python3.8`.

## Learning source embeddings

Once raw files have been copied/download in the related directories, it is possible to run the scripts to learn the embeddings (of course, please edit the source file indicating the correct folder name you used for the raw files).

Fisrt, please install the dependecies you can find in the `requirements.txt` file;
then, for each source, the commands you need to run in order to learn the embeddings are;

```
python learn_graph_embs.py
python learn_text_embs.py ; build_user_profiles.py
python learn_vit_embs.py ; build_user_profiles.py
python learn_video_embs.py ; build_user_profiles.py
python learn_audio_embs.py ; build_user_profiles.py
```

These script will produce `.pkl` (`pickle`) files containing the embeddings; these files are `dict` that map the ID of the node (`user/item`) with the associated `numpy.ndarry` embedding.

## Train the Recommendation Model

Once the embeddings for the source are learnt, you can move them to the `GETAll/data/embedding/<name_of_the_source>/` folder;
then, you can run the following script:
```
python src/train_movielens.py
```
or 
```
python src/train_dbbook.py
```

These script will train the models in the configurations presented in the paper (5 sources for MovieLens and 3 sources for DBbook), bu if you want to change other parameters (such as, the numbero of sources, or the dropout values), you can edit the corresponding variables in the script; in our example, these variables are `feature_list` (that must contain lists of the embeddings to be used), the variable `getall_models` (if the `i-th` feature list that contains `n` source, then the `i-th` model in `getall_models` must handle `n` sources), and finallt the corresponding `dropout_values` (a list of the dropout values that must be applied to the embeddings).

Once the training is complete, in the `models` folder you will find the `.pth`, and in the `predictions` file you will find the `<configuration>_top5.tsv` file.

## Evaluate the prediction lists

These files can be evaluated with the [ClayRS](https://swapuniba.github.io/ClayRS/) framework: we suggest to consider the original documentation of the framework for more technical deatils.
In order to replicate our results, CalyRS needs the original training and testing files; to simplify everything, we choose to replicate them in the `datasets` folder.

The script we havbe provided are able to produce the results to answer the RQs of the paper, including ttests.

For example, to perform the evalutation for the first two RQs, you just need to run:
```
python eval_rq1_rq2.py 
```