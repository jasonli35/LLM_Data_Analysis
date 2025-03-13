import pandas as pd
from datasets import load_dataset

ds = load_dataset("open-llm-leaderboard/contents")
df = pd.read_parquet("hf://datasets/open-llm-leaderboard/contents/data/train-00000-of-00001.parquet")
df = pd.read_json('data_4072.json', orient='records')


from hypothesis import given, strategies as st, assume
@given(st.just(df))  # Use `st.just()` to test the existing DataFrame
def test_existing_dataframe_null_values(df):
    """Check if the existing DataFrame contains None, NaN, or null values."""

    assume(df is not None)  # Ensure df exists
    assert not df.isna().any().any(), "The DataFrame should contain at least one None/NaN value"

test_existing_dataframe_null_values()

def get_rank(num_criteria = [], confident=0.5, num_top_model=5, str_criteria = [], range_matrices = {},rank_reversed = False):
    """
    The function will rank the models based on selected criteria. If the criteria is empty, we will take the average of all the metrics.
    @param criteria: The metrics that is most important for this specific tasks. If it is None, we will take the average of these matrics.
    @param confident: how much confident user think num_criteria and str_criteria are correct in range of 0-1.
    @param num_top_model: The number of top models we want to show.
    @param str_criteria: The criteria 
    @param range_matrices: The range of the criteria that we want to consider, where the key are matrices and 
    @param rank_reversed: If True, the rank will be reversed.

    @return: The rank of the models.
    """
    assert type(num_criteria) == list, "criteria should be a list"
    # matrices = ["MMLU-PRO", "MUSR", "GPQA", "MATH Lvl 5", "BBH", "IFEval", "CO₂ cost (kg)", "Flagged", "MoE", "#Params (B)", "Hub ❤️"]
    matrices = ["MMLU-PRO", "MUSR", "GPQA", "MATH Lvl 5", "BBH", "IFEval", "CO₂ cost (kg)", "Flagged", "MoE", "#Params (B)", "Hub ❤️"]
    for c in num_criteria:
        assert c in matrices, f"criteria {c} is not allowed. Note that it can only take one of the following values: ['MMLU-PRO', 'MUSR', 'GPQA', 'MATH Lvl 5', 'BBH', 'IFEval', 'CO₂ cost (kg)', 'Flagged', 'MoE', '#Params (B)', 'Hub ❤️']" 
    assert type(confident) == float, "confident should be a float"
    assert 0 <= confident <= 1, "confident should be between 0 and 1"
    assert isinstance(range_matrices, dict), "range_matrices should be a dict"
    assert type(rank_reversed) == bool, "rank_reversed should be a bool"
 
    assert type(str_criteria) == list, "str_criteria should be a list"
    for c in str_criteria:
        assert type(c) == tuple, "Entry of the str_criteria should be a tuple"
        assert c[0] in df.columns, f"criteria {c[0]} is not allowed. Note that it can only take one of the following values: {df.columns}"
        assert c[1] in df[c[0]].unique(), f"criteria {c[1]} is not allowed for {c[0]}. Note that it can only take one of the following values: {df[c[0]].unique()}"

    for matric, range in range_matrices.items():
        assert matric in matrices, f"matric {matric} is not allowed. Note that it can only take one of the following values: {matrices}"
        assert isinstance(range, tuple), "Entry of the range_matrices should be a tuple"
        assert len(range) == 2, "Entry of the range_matrices should be a tuple of length 2"
        assert (isinstance(range[0], float) or isinstance(range[0], int)) and (isinstance(range[1], float) or isinstance(range[1], int)), "Entry of the range_matrices should be a tuple of number"
        assert range[0] < range[1], "Entry of the range_matrices should be a tuple of floats where the first element is smaller than the second element"
    MAX_SCORE = 100
    

    this_df = df.copy()
    this_df["new_score"] = 0    

    #add more condition
    if len(num_criteria) == 0: 
        this_df["new_score"] = df["Average ⬆️"]
    #add a new column new_score that get the average of the selected criteria in df 
    else:
        
        for c in num_criteria:
            this_df["new_score"] += this_df[c]
        
        for s_tuple in str_criteria:
        #     if s_tuple[1] == this_df[s_tuple[0]]:
        #         this_df["str_score"] += MAX_SCORE
            this_df.loc[this_df[s_tuple[0]] == s_tuple[1], "new_score"] += MAX_SCORE
        this_df["new_score"] = this_df["new_score"] / (len(num_criteria) + len(str_criteria))
        this_df["new_score"] = this_df["new_score"] * confident + df["Average ⬆️"] * (1 - confident)
    
    for matric, range in range_matrices.items():
        this_df.loc[this_df[matric] < range[0], "new_score"] = 0
        this_df.loc[this_df[matric] > range[1], "new_score"] = 0


    if rank_reversed:
        this_df = this_df.sort_values(by=["new_score"], ascending=True)
    else:
        this_df = this_df.sort_values(by=["new_score"], ascending=False)
    
    

    
    return this_df["eval_name"][:num_top_model]





    
#tech industry
tech_cri = ["#Params (B)", "MUSR"]
tech_industry_models = [("Architecture", "GPTNeoXForCausalLM"), ("Architecture", "LlamaForCausalLM"), ("Architecture", "Qwen2ForCausalLM"), ("Architecture", "Qwen2MoeForCausalLM"), ("Architecture","T5ForConditionalGeneration"), ("Architecture", "CohereForCausalLM"), ("Architecture", "GPTJForCausalLM")]
print("Ranking for tech industry: ",get_rank(tech_cri, confident=0.5, num_top_model=3, str_criteria = tech_industry_models))
print(">>>>>>>>>>>>>>>>")

academic_num_cri = ["MUSR", "MATH Lvl 5", "GPQA"]
print("Ranking for academic industry: ", get_rank(academic_num_cri, confident=0.5, num_top_model=3))
print(">>>>>>>>>>>>>>>>")



manufac_num_cri = ["MMLU-PRO", "#Params (B)", "BBH", "IFEval"]
best_architecture_for_manufacturing = [("Architecture", "LlamaForCausalLM"), ("Architecture", "GPTJForCausalLM"), ("Architecture", "CohereForCausalLM"), 
                                       ("Architecture", "T5ForConditionalGeneration"), ("Architecture", "RwkvForCausalLM")]
find_tune = [("Type", "🔶 fine-tuned on domain-specific datasets")]
manu_str_cri = best_architecture_for_manufacturing + find_tune

print("Ranking for manufacture industry: ", get_rank(manufac_num_cri, confident=0.5, num_top_model=3, str_criteria=manu_str_cri, range_matrices = {"CO₂ cost (kg)": (0, 8), "#Params (B)": (0, 10)}))  

print(">>>>>>>>>>>>>>>>")


cs_num_cri = ["IFEval", "MMLU-PRO"]
cs_str_cri = [("T", "💬"), ("Type", "💬 chat models (RLHF, DPO, IFT, ...)")]

print("Ranking for customer service industry: ", get_rank(cs_num_cri, confident=0.8, num_top_model=10, str_criteria=cs_str_cri))
print(">>>>>>>>>>>>>>>>")

co2_cri = ["CO₂ cost (kg)"]
print("ranking for co2 with range(10, 10000)")
print(get_rank(co2_cri, confident=1.0, rank_reversed=True, num_top_model=3,range_matrices = {"CO₂ cost (kg)": (10, 10000)}))

print("normal co2 rank")
print("ranking for cos2", get_rank(co2_cri, confident=1.0, num_top_model=3, rank_reversed=True))


