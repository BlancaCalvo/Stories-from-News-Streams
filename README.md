# Stories-from-News-Streams

Source code for the paper "Finding Narratives in News Flows: The Temporal Dimensions of News Stories". A dataset for this corpus can be scraped using the code in https://github.com/BlancaCalvo/Spanish-Newspapers-Scraper 

Find here an interactive app to inspect the data used in the paper: https://blancacalvofigueras.shinyapps.io/shiny/  

## Requirements

Run the command:

```
pip install -r requirements.txt
```

## Clustering per weekly time window

```
python cluster_extraction.py corpus.csv --week 2018-00 --manual_eval True --plots True

```

Where corpus.csv is the name of your dataset, 2018-00 is the week to cluster, in this case the first week of 2018, manual evaluation is performed, and the extra plots are also shown. 

## Story Extraction

```
python story_extraction.py  --start_week 1 --similarity_threshold 0.35 --merging_threshold 0.33 --min_articles 10

```

Bridges the clusters between weeks starting at week 1. The articles are added to the central cluster if they have more than 0.35 cosin similarity. Clusters are merged to other clusters if they have 33% of the clusters in common. The minimum of new articles found in a colliding week is 10.

## Evaluation

```
python evaluation.py  --data data.csv --annotations annotations.csv
```

Where data.csv is the predicted labels for the annotated fraction and annotations.csv is the manually annotated data.
