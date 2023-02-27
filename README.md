# parasim
Paratope prediction and similarity via protein language models


You need to have ANARCI (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabpred/anarci/) installed for CDR annotation.

Install dependencies:

```
pip install transformers
pip install datasets
```
Sample dataset file is included as supplementary info for the paper.

Run annotation on the sequence data set file and run paratope prediction (you need to train your paratope prediction model first):
```
python annotate.py chains.csv chains_ann.csv #generates annotated sequence file.
cd paratope_prediction
python bert_finetune.py --predict_only 1 --lstm 0 --data <trainingset> --predict_data <test data> --predict_model results/<trained model checkpoint>
```
There is a `predictions.csv` file generated afterwards. Copy it to the current directory and generate the sequence file with predicted probabilities.
```
cp paratope_prediction/predictions.csv .
python generate_par_dataset.py chains_ann.csv predictions.csv #generates chains_ann.csvpredicted_pars.csv, sequence dataset file with probability values, needed for next step.
python gen_fps.py chains_ann.csvpredicted_pars.csv #generates embeddings npy file
python gen_distmat.py #generates distance matrix (as csv)
```
This generates a npy file that has the paratope embeddings and also a csv file containing the distance matrix.

For training, use option `--train_predict 1` instead of `--predict_only 1`

