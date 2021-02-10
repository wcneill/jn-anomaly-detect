## ICEWS Anomaly Detection
This describes the notebooks found inside the AD_ICEWS/ directory. 

### Anomolous Event Detection (Unaggregated ICEWS data)

Used a simple dense encoder/decoder network architecture to reconstruct single events/samples from ICEWS. Anomaly scores were simple reconstruction errors. This approach showed promise, but without ground truth data, it's hard to say whether detected anomalies are true events or dirty data:

- __anomaly_detect_nn1__, Define a "stable" dataset comprised of ICEWS event data from countries that are known to largely be at peace and train an auto-encoder network architecture to reconstruct this data. 

- __anomaly_detect_nn2__, Use the trained auto-encoder to estimate the distribution of loss on the entire "stable" dataset. Based on this distribution, choose a threshold for what is considered an anomaly.

- __anomaly_detect_nn3__, Select two new countries (one stable and one unstable) that the auto encoder hasn't seen and compare there respective loss values. The stable country should show fewer anomalous events than the unstable country.


### Timeseries Anomaly Detection (Aggregated ICEWS data)
This series of notebooks focuses on event characteristic daily rates as a time series. An LSTM based autoencoder architecture was used to reconstruct time series data. The following notebooks are an impementation (with tweaks) of this paper: https://arxiv.org/pdf/1607.00148.pdf.

- __1_AD_LSTM_preprocess.ipynb__, Preprocess "stable" ICEWS data into a time-series containing daily rates of specific quad classes and intensity of event. Save to file for training and validating models.

- __2_AD_LSTM_training.ipynb__, Train an LSTM based encoder/decoder model to reconstruct sub-sequences of ICEWS time-series data.

- __3_AD_LSTM_thresholding.ipynb__, Use trained model to build a distribution of anomaly scores of "stable" data.  

- __4_AD_LSTM_inference.ipynb__  Using a threshold in previous notebook, run anomaly detection on country of choice.

