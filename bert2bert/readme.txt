Data pre-processing steps:
- Data was read from csv file in folder ‘cnn_dailymail’
- Data was then cleaned and tokenized by the file `text2token.py`. All documents (article and summary) were tokenized into Bert token id, padded to vector of 512-dimension (articles longer than 512 words will discard the 513th words onwards) (padding token  = -1), and saved to .hdf5 file in `preprocessed` folder. All relevant files to bert2bert were moved to the folder `bert2bert` from here
- Model training: Weights and model architecture of the bert encoder and bert decoder were loaded by huggingface library. Due to limited computational resource, we freeze the whole encoder so that it does not update the weights, and train the decoder only. We also modified the decoder to use 4 layers of transformer instead of 12 as in the original library. Model were saved after 5 training epochs to a file named ‘bert2bert.pt’ in ‘models’ folder
- Model tuning: Model ‘bert2bert.pt’ was continued to be trained in additional 5 epochs.
- Inference: We set model to evaluate mode, let the whole test set to run though the model (in batches of 16 documents) and compute the rouge scores of the predicted summary and the ground truth summary

