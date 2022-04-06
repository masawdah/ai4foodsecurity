# AI4FoodSecurity Challenge - ESA's AI4EO Initiative

Third place solution for AI4FoodSecurity Challenge organized by **ESA, Planet, TUM, DLR, and Radiant Earth.**

The goal of this challenge was to classify crop types based on time series data from Sentinel-1, Sentinel-2 and Planet Fusion Monitoring Data. The datasets provided presented two main challenges to the community: exploit the temporal dimension for improved crop classification and ensure that models can handle a domain shift to a different year.

## Solution Description
The solution built based on accommodating compute infrastructure limitations by ensembling boosting models with 10 fold stratified sampling then blend it with light temporal convolutional nueral network to stable the predictions. 

More information about the solution and extracted fatures can be founf in the notebooks. 

## Additional information:

Modelling Environment
- Environment : Google Colab Pro
- python
- GPU : Nvidia P100
