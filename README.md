# Code2vec
A neural network for learning distributed representations of code.
This is an official implementation of the model described in:

[Uri Alon](http://urialon.cswp.cs.technion.ac.il), [Meital Zilberstein](http://www.cs.technion.ac.il/~mbs/), [Omer Levy](https://levyomer.wordpress.com) and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/),
"code2vec: Learning Distributed Representations of Code", POPL'2019 [[PDF]](https://urialon.cswp.cs.technion.ac.il/wp-content/uploads/sites/83/2018/12/code2vec-popl19.pdf)

_**October 2018** - the paper was accepted to [POPL'2019](https://popl19.sigplan.org)_!

_**April 2019** - The talk video is available [here](https://www.youtube.com/watch?v=EJ8okcxL2Iw)_.

An **online demo** is available at [https://code2vec.org/](https://code2vec.org/).

This is a TensorFlow implementation, designed to be easy and useful in research, 
and for experimenting with new ideas in machine learning for code tasks.
By default, it learns Java source code and predicts Java method names, but it can be easily extended to other languages, 
since the TensorFlow network is agnostic to the input programming language (see [Extending to other languages](#extending-to-other-languages).
Contributions are welcome.

<center style="padding: 40px"><img width="70%" src="https://github.com/tech-srl/code2vec/raw/master/images/network.png" /></center>

Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Configuration](#configuration)
  * [Features](#features)
  * [Extending to other languages](#extending-to-other-languages)
  * [Additional datasets](#additional-datasets)
  * [Citation](#citation)

## Requirements
On Ubuntu:
  * [Python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04). To check if you have it:
> python3 --version
  * TensorFlow - version 1.5 or newer ([install](https://www.tensorflow.org/install/install_linux)). To check TensorFlow version:
> python3 -c 'import tensorflow as tf; print(tf.\_\_version\_\_)'
  * If you are using a GPU, you will need CUDA 9.0 ([download](https://developer.nvidia.com/cuda-90-download-archive)) 
  as this is the version that is currently supported by TensorFlow. To check CUDA version:
> nvcc --version
  * For GPU: cuDNN (>=7.0) ([download](http://developer.nvidia.com/cudnn))
  * For [creating a new dataset](#creating-and-preprocessing-a-new-java-dataset) or [manually examining a trained model](#step-4-manual-examination-of-a-trained-model) (any operation that requires parsing of a new code example) - [Java JDK](https://openjdk.java.net/install/)

## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/tech-srl/code2vec
cd code2vec
```

### Step 1: Creating a new dataset from java sources
In order to have a preprocessed dataset to train a network on, you can either download our
preprocessed dataset, or create a new dataset of your own.

#### Download our preprocessed dataset of ~14M examples (compressed: 6.3GB, extracted 32GB)
```
wget https://s3.amazonaws.com/code2vec/data/java14m_data.tar.gz
tar -xvzf java14m_data.tar.gz
```
This will create a data/java14m/ sub-directory, containing the files that hold that training, test and validation sets,
and a vocabulary file for various dataset properties.

#### Creating and preprocessing a new Java dataset
In order to create and preprocess a new dataset (for example, to compare code2vec to another model on another dataset):
  * Edit the file [preprocess.sh](preprocess.sh) using the instructions there, pointing it to the correct training, validation and test directories.
  * Run the preprocess.sh file:
> source preprocess.sh

### Step 2: Training a model
You can either download an already-trained model, or train a new model using a preprocessed dataset.

#### Downloading a trained model (1.4 GB)
We already trained a model for 8 epochs on the data that was preprocessed in the previous step.
The number of epochs was chosen using [early stopping](https://en.wikipedia.org/wiki/Early_stopping), as the version that maximized the F1 score on the validation set. This model can be downloaded [here](https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz) or using:
```
wget https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz
tar -xvzf java14m_model.tar.gz
```

##### Note:
This trained model is in a "released" state, which means that we stripped it from its training parameters and can thus be used for inference, but cannot be further trained. If you use this trained model in the next steps, use 'saved_model_iter8.release' instead of 'saved_model_iter8' in every command line example that loads the model such as: '--load models/java14_model/saved_model_iter8'. To read how to release a model, see [Releasing the model](#releasing-the-model).

#### Downloading a trained model (3.5 GB) _which can be further trained_

A non-stripped trained model can be obtained [here](https://s3.amazonaws.com/code2vec/model/java14m_model_trainable.tar.gz) or using:

```
wget https://s3.amazonaws.com/code2vec/model/java14m_model_trainable.tar.gz
tar -xvzf trained_model.tar
```  

This model weights more than twice than the stripped version, and it is recommended only if you wish to continue training a model which is already trained. To continue training this trained model, use the `--load` flag to load the trained model; the `--data` flag to point to the new dataset to train on; and the `--save` flag to provide a new save path.

#### Training a model from scratch
To train a model from scratch:
  * Edit the file [train.sh](train.sh) to point it to the right preprocessed data. By default, 
  it points to our "java14m" dataset that was preprocessed in the previous step.
  * Before training, you can edit the configuration hyper-parameters in the file [common.py](common.py),
  as explained in [Configuration](#configuration).
  * Run the [train.sh](train.sh) script:
```
source train.sh
```

##### Notes:
  1. By default, the network is evaluated on the validation set after every training epoch.
  2. The newest 10 versions are kept (older are deleted automatically). This can be changed, but will be more space consuming.
  3. By default, the network is training for 20 epochs.
These settings can be changed by simply editing the file [common.py](common.py).
Training on a Tesla v100 GPU takes about 50 minutes per epoch. 
Training on Tesla K80 takes about 4 hours per epoch.

### Step 3: Evaluating a trained model
Once the score on the validation set stops improving over time, you can stop the training process (by killing it)
and pick the iteration that performed the best on the validation set.
Suppose that iteration #8 is our chosen model, run:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --test data/java14m/java14m.test.c2v
```
While evaluating, a file named "log.txt" is written with each test example name and the model's prediction.

### Step 4: Manual examination of a trained model
To manually examine a trained model, run:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --predict
```
After the model loads, follow the instructions and edit the file Input.java and enter a Java 
method or code snippet, and examine the model's predictions and attention scores.

## Configuration
Changing hyper-parameters is possible by editing the file [common.py](common
.py).

Here are some of the parameters and their description:
#### config.NUM_EPOCHS = 20
The max number of epochs to train the model. Stopping earlier must be done manually (kill).
#### config.SAVE_EVERY_EPOCHS = 1
After how many training iterations a model should be saved.
#### config.BATCH_SIZE = 1024 
Batch size in training.
#### config.TEST_BATCH_SIZE = config.BATCH_SIZE
Batch size in evaluating. Affects only the evaluation speed and memory consumption, does not affect the results.
#### config.READING_BATCH_SIZE = 1300 * 4
The batch size of reading text lines to the queue that feeds examples to the network during training.
#### config.NUM_BATCHING_THREADS = 2
The number of threads enqueuing examples.
#### config.BATCH_QUEUE_SIZE = 300000
Max number of elements in the feeding queue.
#### config.DATA_NUM_CONTEXTS = 200
The number of contexts in a single example, as was created in preprocessing.
#### config.MAX_CONTEXTS = 200
The number of contexts to use in each example.
#### config.WORDS_VOCAB_SIZE = 1301136
The max size of the token vocabulary.
#### config.TARGET_VOCAB_SIZE = 261245
The max size of the target words vocabulary.
#### config.PATHS_VOCAB_SIZE = 911417
The max size of the path vocabulary.
#### config.EMBEDDINGS_SIZE = 128
Embedding size for tokens and paths.
#### config.MAX_TO_KEEP = 10
Keep this number of newest trained versions during training.

## Features
Code2vec supports the following features: 

### Releasing the model
If you wish to keep a trained model for inference only (without the ability to continue training it) you can
release the model using:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --release
```
This will save a copy of the trained model with the '.release' suffix.
A "released" model usually takes 3x less disk space.

### Exporting the trained token vectors and target vectors
Token and target embeddings are available to download [here](http://urialon.cswp.cs.technion.ac.il/publications/).
The saved embeddings there are saved without subtoken-delimiters ("*toLower*" is saved as "*tolower*").

In order to export embeddings from a trained model, use the "--save_w2v" and "--save_t2v" flags:

Exporting the trained *token* embeddings:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --save_w2v models/java14_model/tokens.txt
```
Exporting the trained *target* (method name) embeddings:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --save_t2v models/java14_model/targets.txt
```
This saves the tokens/targets embedding matrices in word2vec format to the specified text file, in which:
the first line is: \<vocab_size\> \<dimension\>
and each of the following lines contains: \<word\> \<float_1\> \<float_2\> ... \<float_dimension\>

These word2vec files can be manually parsed or easily loaded and inspected using the [gensim](https://radimrehurek.com/gensim/models/word2vec.html) python package:
```python
python3
>>> from gensim.models import KeyedVectors as word2vec
>>> vectors_text_path = 'models/java14_model/targets.txt' # or: `models/java14_model/tokens.txt'
>>> model = word2vec.load_word2vec_format(vectors_text_path, binary=False)
>>> model.most_similar(positive=['equals', 'to|lower']) # or: 'tolower', if using the downloaded embeddings
>>> model.most_similar(positive=['download', 'send'], negative=['receive'])
```
The above python commands will result in the closest name to both "equals" and "to|lower", which is "equals|ignore|case".
Note: In embeddings that were exported manually using the "--save_w2v" or "--save_t2v" flags, the input token and target words are saved using the symbol "|" as a subtokens delimiter ("*toLower*" is saved as: "*to|lower*"). In the embeddings that are available to download (which are the same as in the paper), the "|" symbol is not used, thus "*toLower*" is saved as "*tolower*".

### Exporting the code vectors for the given code examples
The flag `--export_code_vectors` allows to export the code vectors for the given examples. 

If used with the `--test <TEST_FILE>` flag,
a file named `<TEST_FILE>.vectors` will be saved in the same directory as `<TEST_FILE>`. 
Each row in the saved file is the code vector of the code snipped in the corresponding row in `<TEST_FILE>`.
 
If used with the `--predict` flag, the code vector will be printed to console.


## Extending to other languages  
In order to extend code2vec to work with other languages other than Java, a new extractor (similar to the [JavaExtractor](JavaExtractor))
should be implemented, and be called by [preprocess.sh](preprocess.sh).
Basically, an extractor should be able to output for each directory containing source files:
  * A single text file, where each row is an example.
  * Each example is a space-delimited list of fields, where:
  1. The first "word" is the target label, internally delimited by the "|" character.
  2. Each of the following words are contexts, where each context has three components separated by commas (","). Each of these components cannot include spaces nor commas.
  We refer to these three components as a token, a path, and another token, but in general other types of ternary contexts can be considered.  

For example, a possible novel Java context extraction for the following code example:
```java
void fooBar() {
	System.out.println("Hello World");
}
```
Might be (in a new context extraction algorithm, which is different than ours since it doesn't use paths in the AST):
> foo|Bar System,FIELD_ACCESS,out System.out,FIELD_ACCESS,println THE_METHOD,returns,void THE_METHOD,prints,"hello_world" 

Consider the first example context "System,FIELD_ACCESS,out". 
In the current implementation, the 1st ("System") and 3rd ("out") components of a context are taken from the same "tokens" vocabulary, 
and the 2nd component ("FIELD_ACCESS") is taken from a separate "paths" vocabulary. 

## Additional datasets
We preprocessed additional three datasets used by the [code2seq](https://arxiv.org/pdf/1808.01400) paper, using the code2vec preprocessing.
These datasets are available in raw format (i.e., .java files) at [http://urialon.cswp.cs.technion.ac.il/publications/](http://urialon.cswp.cs.technion.ac.il/publications/),
and are also available to download in a preprocessed format (i.e., ready to train a code2vec model on) here:

### Java-small (compressed: 366MB, extracted 1.9GB)
```
wget https://s3.amazonaws.com/code2vec/data/java-small_data.tar.gz
```
This dataset is based on the dataset of [Allamanis et al. (ICML'2016)](http://groups.inf.ed.ac.uk/cup/codeattention/), with the difference that training/validation/test are split by-project rather than by-file.
This dataset contains 9 Java projects for training, 1 for validation and 1 testing. Overall, it contains about 700K examples.

### Java-med (compressed: 1.8GB, extracted 9.3GB)
```
wget https://s3.amazonaws.com/code2vec/data/java-med_data.tar.gz
```
A dataset of the 1000 top-starred Java projects from GitHub. It contains
800 projects for training, 100 for validation and 100 for testing. Overall, it contains about 4M examples.

### Java-large (compressed: 7.2GB, extracted 37GB)
```
wget https://s3.amazonaws.com/code2vec/data/java-large_data.tar.gz
```
A dataset of the 9500 top-starred Java projects from GitHub that were created
since January 2007. It contains 9000 projects for training, 200 for validation and 300 for
testing. Overall, it contains about 16M examples.

## Citation

[code2vec: Learning Distributed Representations of Code](https://urialon.cswp.cs.technion.ac.il/wp-content/uploads/sites/83/2018/12/code2vec-popl19.pdf)

```
@article{alon2019code2vec,
 author = {Alon, Uri and Zilberstein, Meital and Levy, Omer and Yahav, Eran},
 title = {Code2Vec: Learning Distributed Representations of Code},
 journal = {Proc. ACM Program. Lang.},
 issue_date = {January 2019},
 volume = {3},
 number = {POPL},
 month = jan,
 year = {2019},
 issn = {2475-1421},
 pages = {40:1--40:29},
 articleno = {40},
 numpages = {29},
 url = {http://doi.acm.org/10.1145/3290353},
 doi = {10.1145/3290353},
 acmid = {3290353},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Big Code, Distributed Representations, Machine Learning},
}
```
