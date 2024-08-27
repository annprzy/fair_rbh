# Master's Thesis 


# Analysis of fair predictions of classifiers learnt from imbalanced data


In order to run the code, first all of the dependencies need to be installed. The list of the dependencies is saved in the `requirements.txt` file. To install all the dependencies, please run `pip install -r requirement.txt` in the terminal.

All methods are implemented in the `src` directory. As mentioned before, this directory consists of four main modules: `classification`, `datasets`, `evaluation`, `preprocessing`. Furthermore, in the `src` directory, methods used for the hyperparameter tuning and experiments can be found in the files beginning with words `validation` and `experiments`. The file `generate_sampled_data.py` includes the methods needed to generate data for the second experiment.

Moreover, in the `notebooks` directory, some Jupyter notebooks can be run to replicate the experiments and see the obtained results.
To replicate the experiments, first, the distance selection needs to be performed using the `distance_selection.ipnyb` notebook. The results of the distance selection step can be visualized using the `validation_results_analysis.ipnyb` notebook, after adjusting some parameters (such as the name of the algorithms that we want to analyze). After running the `distance_selection.ipnyb`, the experimental notebooks can be executed. 
The `experiment1.ipnyb` notebook contains the components needed to perform the first experiment. However, for the first step -- hyperparameter tuning -- one needs to adjust the distance accordingly to earlier obtained results. After running the hyperparameter tuning, the appropriate parameters need to be changed in the `config` files. Finally, the rest of the notebook can be run -- the results for the experiments will be saved in csv files under the provided path. Similarly, the second experiment can be performed using `experiment2.ipnyb`. 

The results of the experiments can be visualized using the `results_analysis.ipnyb` notebook. Moreover, the information about datasets can be displayed using `dataset_info.ipnyb` notebook.
