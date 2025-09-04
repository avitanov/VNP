#  Stock Price Prediction with TFT 

## 🏗️ **SYSTEM ARCHITECTURE**

```
📊 DATA SOURCES → 🔧 FEATURE ENGINEERING → 🤖 TFT MODEL → 📈 PREDICTIONS
       ↓                    ↓                  ↓           ↓
   Alpha Vantage API    Custom Features     Deep Learning   Price Forecasts
   News Sentiment      Technical Analysis   Attention       Risk Analysis
   Economic Events     Cross-Asset Data     Time Series     Trading Signals
```

##  Project Overview

DATASET WORKFLOW : 

run_feature_pipeline.py
	↓
feature_pipeline.py
	↓
tft_preprocessor.py
	↓
AAPL_TFT_READY.csv


1. run_feature_pipeline.py
- Tuka e konfiguracijata na modelot (kolku features da zema, koi podatoci da gi koriste)
- go povikuva feature_pipeline.py
- na kraj vrakja dataset so podatocite od apito + uste tehnicki indikatore koi se presmetuvaat lokalno
ne od api i se potrebni za TFT da dade podobar rezultat + cross assets features + event features. (nad 100 features)

2. feature_pipeline.py 
- E povikan od strana na run_feature_pipeline.py.
- go loadnuva merge datasetot so tehnickite indikatori i site podatoci od apito (prices + news)
- mu dodava  coustom technical indicators koi gi nema vo apito a mu se potrebni na TFT za podobar rezultat.
- dodava cross-asset features (VIX, dollar index)
- dodava event features (Fed meetings, earnings, holidays)

3. tft_preprocessor.py
- Povikan od strana na feature_pipeline.py 
- Prava kategorizacija na features.
- presmetuva feature importance i gi zima najdobrite 63 (momentalno, moze da se proba so pomalku ~50, povekje mislam ne.)
- prava skaliranje na vrednostite, se koriste RobustScaler(najdobar za finanskiski time series podatoci, dobro se spravuva so outlayeri, stabilen e). Moze da se isproba i QuantileUniformTransformer.
- Gi formatira podatocite vo format koj gi ocekuva TFT modelot.

= > Na kraj se dobiva gotov csv dataset za trenirane.

=================================================================================================================


COMPLETE WORKFLOW :

python run_feature_pipeline.py  
    ↓
python run_tft_training.py      
    ↓  
python production_predictions.py


run_tft_training.py :
- Go zima gotoviot dataset od pipelinot so momentalno 63 features.
- Gi deli podatocite na 80 / 20
- parametrite na modelot (momentalno e staven na 20 epohi za brzo da se trenira, pak mu treba 30+ min, ama idealno
 bi bilo da se proba so okolu 70 i ke trae sigurno nad 2h.
 - modelot treba da se prilagoduva, momentalniot dropout e 0.2, mozno e da e visok, batch sizeot moze da se proba so 16.
 - Parametrite konfigurirani tuka se ovveridunuvat vo tft_trainer.

 production_predictions.py:
 - Go zima treniraniot TFT model
 - pravi predikcii za slednite 12 steps
 - spored cenite dobieni od tft modelot dava UP/DOWN signali
 

 

