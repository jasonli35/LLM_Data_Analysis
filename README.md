## File structure

LLM_Data_Analysis/

│── .hypothesis/examples/      
│── additional_code/           
│    │── MMLU_basic_analysis.ipynb  
│    │── main.py                   
│    │── preprocess.ipynb           
│    │── rank.ipynb                 
│── .gitignore                
│── How to choose a good LLM.pdf  
│── README.md                 
│── data_4072.json            
│── main.py                  
│── visualization.ipynb       

The main.py file contains the code for the ranking algorithm. The preprocess directory includes hypothesis testing to filter out bad data. The visualization.ipynb file contains all visualizations. The additional_code folder includes all the source code we worked on for this project. Demonstration of ranking of different industry can be found in additional_code/rank.ipynb 




## How to run the code?


The get_rank function ranks models based on selected criteria. If no criteria are provided, it computes the average of all available metrics. This function allows users to define numerical and string-based criteria, specify confidence levels, filter models based on metric ranges, and adjust ranking order.

Importing the Function

To use get_rank, first import it from main.py:

from main import get_rank

Function Parameters

1. num_criteria (list, default: [])

A list of numerical metrics to prioritize.

If left empty, the function averages all available numerical metrics.

Example:

num_criteria = ["#Params (B)", "MUSR"]

2. confident (float, default: 0.5)

A confidence level (between 0 and 1) determining how much weight should be given to num_criteria and str_criteria.

Higher values prioritize the selected criteria more.

3. num_top_model (int, default: 5)

Specifies the number of top models to return.

Example:

num_top_model = 3  # Returns the top 3 models

4. str_criteria (list, default: [])

A list of string-based criteria to prioritize.

Example:

str_criteria = [("Architecture", "GPTNeoXForCausalLM"), ("Type", "🔶 fine-tuned on domain-specific datasets")]

5. range_matrices (dict, default: {})

Defines acceptable numerical ranges for metrics.

Example:

range_matrices = {"CO₂ cost (kg)": (10, 10000)}

This filters models with CO2 cost between 10 and 10000 

6. rank_reversed (bool, default: False)

If True, reverses the ranking order (useful for metrics where lower is better, such as inference time).

Example:

## Thrid party module
- hypothesis module - The hypothesis module is a third-party Python library used for property-based testing. Instead of writing specific test cases, hypothesis generates a wide range of test inputs automatically and checks if the code behaves correctly for all of them.
- panda
- numpy
- matplotlib
- datasets - The Hugging Face datasets library is an Python package for loading, processing, and using large-scale datasets, particularly for machine learning (ML) and natural language processing (NLP) tasks.
- third-party Python library built on top of Matplotlib that is designed for making statistical visualizations more attractive and insightful


## Data visualization 
data visualization code can be found in visualization.ipynb






