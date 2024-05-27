# The MERIT Dataset :school_satchel::page_with_curl::trophy:
### Modelling and Efficiently Rendering Interpretable Transcripts

The MERIT Dataset is a multimodal dataset (image + text + layout) designed for training and benchmarking Large Language Models (LLMs) on Visually Rich Document Understanding (VrDU) tasks.

**You can find more about our work here:**
+ **[Paper]():** Read all the details of our pipeline and dataset.Under revision :hammer_and_wrench:
+ **The [MERIT Dataset]()**: explore the raw data. Under development :hammer_and_wrench:
+ **Plug-and-play [subsets]()**. The subsets that we have used in our benchmark by using the HuggingFace pipeline 🤗. Under development :hammer_and_wrench:
+ **Our [benchmark](https://wandb.ai/iderodrigo/MERIT_Dataset?nw=nwuseriderodrigo)**. We include all the details of our benchmark with result metrics.



## Introduction :information_source:

<p align="center">
  <img src="figs/dataset_figs/config/visualAbstract.png" alt="Visual Abstract" width="600" />
</p>

## Pipeline :arrows_counterclockwise:
<p align="center">
  <img src="figs/pipeline_figs/pipeline_overview.png" alt="Pipeline Overview" width="600" />
</p>

## Dataset :bookmark_tabs::camera:

<p align="center">
  <img src="figs/dataset_figs/config/dataset_structure.png" alt="Dataset Structure" width="600" />
</p>


You can fin [here]() our raw dataset (please notice the complete dataset weights +200GB). You can train with our plug-and-play [subsets]() by using the HuggingFace 🤗 pipeline. (Under development :hammer_and_wrench:)

## Software :woman_technologist:

We run the pipeline on Ubuntu 20.04. It is designed to run on Windows too, although it has not been tested yet.

Requirements file under development :hammer_and_wrench:

## Hardware :gear:
We run the pipeline on an MSI Meg Infinite X 10SF-666EU with an Intel Core i9-10900KF and an Nvidia RTX 2080 GPU, running on Ubuntu 20.04. We consumed 0.016 kWh/1000 samples when generating the digital samples and 0.366 kWh/1000 samples when modifying them in Blender.

## Benchmark :muscle:

We train the LayoutLM family models on token classification to demonstrate the suitability of our dataset. The MERIT Dataset poses a challenging scenario with more than 400 labels.

We benchmark on three scenarios with an increasing presence of Blender-modified samples.

+ Scenario 1: We train and test on digital samples.
+ Scenario 2: We train with digital samples and test with Blender-modified samples.
+ Scenario 3: We train and test with Blender-modified samples.


|                  | **Scenario 1** | **Scenario 2** | **Scenario 3** | **FUNSD/** | **Lang.** | **(Tr./Val./Test)** |
|------------------|----------------|----------------|----------------|------------|-----------|----------------------|
|                  | Dig./Dig.      | Dig./Mod.      | Mod./Mod       | XFUND      |           |                      |
|                  | **F1**         | **F1**         | **F1**         | **F1**     |           |                      |
| LayoutLMv2       | 0.5536         | 0.3764         | 0.4984         | 0.8276     | Eng.      | 7324/1831/4349       |
| LayoutLMv3       | 0.3452         | 0.2681         | 0.6370         | 0.9029     | Eng.      | 7324/1831/4349       |
| LayoutXLM        | 0.5977         | 0.3295         | 0.4489         | 0.7550     | Spa.      | 8115/2028/4426       |

**Access our [paper]() to read our conclusions**. 

## Biases :detective:
Although synthetic, we acknowledge that our data generation involves sensitive topics. A solution to avoid biases might be to randomize students' grades. However, this approach would be useless if we want to detect biases in LLMs. Therefore, we have decided to associate students' grades based on [PISA](https://www.oecd-ilibrary.org/content/publication/53f23881-en) reports. You can discover more information in the paper.

## Team

We are researchers from **[Comillas Pontifical University](https://www.iit.comillas.edu/)**
 - **Ignacio de Rodrigo [@nachoDRT](https://github.com/nachoDRT)**: PhD Student. Software Development, Dataset Curation, Benchmark Design.
 - **Alberto Sánchez [@ascuadrado](https://github.com/ascuadrado)**: Research Assistant. Software Development.
 - **Mauro Liz [@mauroliz](https://github.com/mauroliz)**: Research Assistant. Software Assistance.
 - **Jaime Boal [@jboal](https://github.com/jboalml)**: Supervisor.
 - **Álvaro López [@allopez](https://www.iit.comillas.edu/personas/allopez)**: Supervisor.

## Citation
If you find our research interesting, please cite our work. :page_with_curl::black_nib: