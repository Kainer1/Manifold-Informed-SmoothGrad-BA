# Bachelor thesis code

This is the code for the bachelor's thesis **Smoothing on the Manifold: A Geometric Approach to Gradient-Based Explainable AI**.

## Samplers and configuration

The experiments use VGG16 (`IMAGENET1K_V1`) and 256×256 ImageNet images. The following samplers are implemented; listed hyperparameters are required unless marked optional.

| `sampler_name` | Sampling method | Hyperparameters in `config` |
| --- | --- | --- |
| `VanillaGradient` | Original image | None: `{}`; normally `n_samples=1` |
| `SmoothGrad` | Gaussian image noise | `p ≥ 0`: noise standard deviation as a fraction of the normalized image's value range |
| `LG-SmoothGrad` | Noise in a low-gradient mask | `p ≥ 0`: same noise factor; `q ∈ [0,1]`: gradient-magnitude quantile defining the mask; optional `CircleMask`: `"in"` or `"out"` |
| `VAE_SD2` | Gaussian noise in the SD VAE latent space | `p ≥ 0`: noise standard deviation as a fraction of the latent value range |
| `VAE_REPAE_ImageNet` | Gaussian noise in the ImageNet VAE latent space | `p ≥ 0`: same latent noise factor |
| `DiT` | Latent diffusion | `strength`, `actual_steps`, `class_conditioning` |
| `SD2` | Latent diffusion | `strength`, `actual_steps`, `class_conditioning` |
| `DeepFloyd` | Two-stage pixel diffusion (DF) | `strength`, `actual_steps`, `strength_2`, `actual_steps_2`, `class_conditioning` |
| `ADM` | Class-conditioned pixel diffusion | `strength`; class conditioning is always enabled |

Diffusion `strength` values are in `(0,1]`; `actual_steps` values are positive integers specifying the number of denoising updates, with `_2` selecting DF's second stage. DiT, SD2 and DF require a schedule length `L ≤ 1000` satisfying `int(L × strength) == actual_steps`; ADM uses `max(1, int(1000 × strength))` updates. `class_conditioning` is `true` or `false`.

LG-SmoothGrad fixes `p_tilde=0.09` and `n_tilde=50` in code. Optional `config.CircleMask` replaces its low-gradient mask with a centered circle: `"in"` adds noise inside, `"out"` outside; omit this setting to use the low-gradient mask.

Each configuration requires `sampler_name`, `group_name` (result group), `n_samples` (samples per image), `indices` (dataset rows, e.g. `[0, 1]` or `"0-99"`), `config` (hyperparameters) and `flags` (optional settings). A JSON file contains one object or a list; [configs/example.json](configs/example.json) includes every sampler and all applicable options with illustrative values.

Set `IMAGENET_H5_PATH` in [run.py](run.py) to your processed ImageNet HDF5 file. Its root datasets must be `inputs`: RGB `uint8`, shape `(N,3,256,256)`, and `labels`: `int64`, shape `(N,)`, with ImageNet class IDs `0–999`; images and labels must retain the experiment's row order. Inputs are converted to `float32/255` and normalized with mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.

Run from `Code/` after installing the dependencies:

```bash
python run.py --config configs/example.json
```

Python packages: torch, torchvision, diffusers, transformers, accelerate, sentencepiece, ftfy, beautifulsoup4, h5py, numpy, pandas, openpyxl, matplotlib, torchmetrics, lpips, cmcrameri, Pillow, scipy and Quantus. Use Quantus revision `3c01edfbf8c6152f77e11a22bd8a8dcb24cc5919` from its [repository](https://github.com/understandable-machine-intelligence-lab/Quantus); ADM additionally requires [guided-diffusion](https://github.com/openai/guided-diffusion) revision `22e0df8183507e13a7813f8d38d51b072ca1e67c`, installed with `pip install -e /path/to/guided-diffusion`. Set `ADM_WEIGHTS_PATH` in [Sampling/LoadModels.py](Sampling/LoadModels.py) to `256x256_diffusion.pt`; diffusion loaders read `HF_TOKEN` when accessing pretrained checkpoints.

## Always saved

All outputs are under `Results/VGG16/IMAGENET_256/`. Selected results are recalculated and overwritten on each run.

| Output | Contents |
| --- | --- |
| `Attributions/attributions_<sampler>.h5` | Signed mean of the ground-truth-logit input gradients, retaining all three color channels |
| `Samples/rawOutput/outputs_<sampler>.h5` | Up to four SSIM representatives per visualization image: two highest and two lowest, with sample scores and statistics |
| `Samples/<group>/viz_<sampler>.png` | Sample visualization |
| `Heatmaps/OnlyHeatmaps/heatmaps_<group>.png` | Heatmap overview |
| `Stats/<group>.csv` | SRG mean and standard deviation for each sampler |
| `Stats/AnalyzeAttributionAccuracy/<group>.xlsx` | Mean sample accuracy (%), target logit and target softmax (%) per image |
| `Stats/RawGradientValues/<group>.xlsx` | Absolute-gradient sums over all samples, their mean, the final attribution and first sample; sample count and a 16×16 gradient grid for the first sample of the first selected image |

Sample storage and visualization use the first ten selected image indices by default; heatmap plotting uses the same default independently. With fewer than four samples, all available samples are saved as representatives.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `saveAllSamples` | `false` | Save every generated sample as normalized model input to `Samples/allSamples/samples_<sampler>.h5` |
| `SampleVisualizationIndices` | First ten selected indices | Integer list such as `[0, 1]`; replaces the selection for SSIM representative storage and sample plots |
| `SafeHeatmapPNG` | `false` | Additionally save every displayed heatmap as an individual PNG |
| `SafeHeatmapIndices` | First ten selected indices | Integer list; replaces the selection for both the heatmap overview and individual PNGs |

Both visualization index lists must be subsets of the main `indices` selection. SRG, sample plots, heatmap overviews and raw-gradient statistics always run.

## Additional statistics from saved samples

The sample-based scripts below read `Samples/allSamples/`, created by enabling `saveAllSamples` during sampling; gradient analyses recompute gradients from these saved inputs. `averagedHeatmapCoarse.py` reads saved averaged attributions instead and does not require saved samples or model inference.

| Script in `PostProcessing/` | Analysis |
| --- | --- |
| `attributionSampleSimilarity.py` | Similarity between sample input-gradient heatmaps: Cosine, Spearman and relative L1-mass difference; also Complexity, Sparseness and Coarse |
| `sampleToVanillaSimilarity.py` | Cosine and Spearman between each sample input-gradient heatmap and Vanilla Gradient of the matching original image |
| `averagedHeatmapCoarse.py` | Coarse 8/16/32/64 of the saved averaged attribution heatmaps, as in Fig. C.1; scores per image and means across images |
| `reluActiveAnalysis.py` | Counts and fractions of active ReLU units per layer and across the network |
| `reluGradientAmounts.py` | Incoming and passed absolute-gradient amounts at ReLU layers, blocked fractions and input-gradient amounts |
| `sampleFourierAnalysis.py` | Radial Fourier amplitude and power; means and quantiles for one sample or all samples of a sampler |
| `sampleSRG.py` | SRG for every individual sample gradient, evaluated at its original image as in the old `n_samples=1` runs; CSV in `Stats/SampleSRG/` |

After running the example configuration, use these commands from `Code/`. For other configurations, replace the sampler names with their saved filenames without `samples_` and `.h5`.

```bash
python PostProcessing/attributionSampleSimilarity.py --sampler_names example_SmoothGrad_n4_p0.09
python PostProcessing/sampleToVanillaSimilarity.py --samplers example_SmoothGrad_n4_p0.09 example_ADM_n4_strength0.08 --reference-sampler example_VanillaGradient_n1
python PostProcessing/reluActiveAnalysis.py
python PostProcessing/reluGradientAmounts.py --samplers example_VanillaGradient_n1 example_ADM_n4_strength0.08 example_SmoothGrad_n4_p0.09
python PostProcessing/sampleFourierAnalysis.py --sampler example_SmoothGrad_n4_p0.09
python PostProcessing/sampleFourierAnalysis.py --sampler example_SmoothGrad_n4_p0.09 --index 0 --sample-index 0
python PostProcessing/sampleSRG.py --samplers example_SmoothGrad_n4_p0.09 example_ADM_n4_strength0.08
python PostProcessing/averagedHeatmapCoarse.py --samplers example_VanillaGradient_n1 example_SmoothGrad_n4_p0.09 example_ADM_n4_strength0.08
```

The ReLU-gradient analysis uses every saved sample of each selected image; sample counts may differ between samplers and images. Fourier analysis defaults to quantiles 0.1 and 0.9; change them with `--quantiles`.

Sample SRG uses all complete stored indices and all their samples; optionally select image indices with `--indices 0 1`. It uses `IMAGENET_H5_PATH` from `run.py` for the matching original images and writes `Samplername`, `Gruppenname`, `Index`, `SampleIndex`, `Label` and `SRG` to `Stats/SampleSRG/samples_<sampler>.csv`.

The Vanilla comparison requires a saved VanillaGradient run with `n_samples=1` and `saveAllSamples=true` as its reference. It compares raw RGB-L2-pooled gradient maps without clipping or display normalization. It uses every complete image index and all samples; sample counts may vary. Selected samplers must cover the same complete indices, or use `--indices` to select an explicit set available in every sampler and the reference. Labels must match. Outputs in `Stats/SampleToVanillaSimilarity/` include per-sample, per-image and summary CSVs plus an Excel workbook. Summary means average finite sample coefficients, so images with more valid samples receive more weight. Zero-vector Cosine and constant-map Spearman are undefined; valid counts are exported.

`averagedHeatmapCoarse.py` reads `Attributions/attributions_<sampler>.h5`, applies RGB-L2 pooling to the stored signed mean gradient, and computes Coarse before display normalization. It exports `per_image.csv`, `summary.csv` and `averaged_heatmap_coarse.xlsx` to `Stats/AveragedHeatmapCoarse/`. By default, selected samplers must cover identical image indices; use `--indices 0 1` to select an explicit subset. These scores measure the averaged heatmaps and differ from the mean of individual sample heatmap scores in `attributionSampleSimilarity.py`.

## Independent model evaluations

These scripts use the original ImageNet HDF5 dataset and do not require saved samples. Run them from `Code/`:

```bash
python PostProcessing/vgg16_crop_accuracy.py
python PostProcessing/benchmark_model_runtimes.py --repeats 5 --output Results/VGG16/IMAGENET_256/Stats/model_runtime_benchmark.json
```

`vgg16_crop_accuracy.py` preserves the original comparison: evaluate every dataset row at 256×256 and as a 224×224 center crop, without resizing, with ImageNet normalization and VGG16 `IMAGENET1K_V1`. Classifier inference uses CUDA autocast, matching the original SmoothGrad baseline accuracy calculation; CPU inference uses float32. It prints Top-1 and Top-5 accuracy for both resolutions. Defaults are batch size 50 and CUDA when available; use `--h5-path`, `--batch-size` or `--device` to override them. The default dataset path is `IMAGENET_H5_PATH` in `run.py`.

`benchmark_model_runtimes.py` preserves the CUDA-synchronized timing protocol: one image (index 0), batch size one, one warm-up, median generation time, and no checkpoint loading inside the timed calls. The original CLI default is three repeats; the command above uses the five repeats reported in the thesis. `--models` selects from `smoothgrad`, `repae`, `sd_vae`, `dit`, `sd2`, `adm` and `deepfloyd`. The benchmark uses the current `Code/` samplers and their class conditioning without CFG; the original benchmark used stronger guidance for DiT, SD2 and DeepFloyd. Model configurations are recorded in the output.

The JSON retains the original timing fields, including `projected_total_s_batch1` (image setup plus sample generation). `projected_sample_s_batch1` reports only `project-images × project-samples × sample_s_median`, as used in the thesis runtime table; both projection counts default to 100. Without `--output`, results are printed only. A CUDA GPU is required, pretrained checkpoints use the local cache by default, and `--allow-download` permits downloads. ADM uses `ADM_WEIGHTS_PATH` from `Sampling/LoadModels.py`.
