# Get stat for each trial
from Agreement_corr_cal import compare_human_llm
# read file
llm_df = pd.read_csv('.../model_annotations/WMT_D/LLM_annotated/JM_ofshe_Gemini_Flash_OFF.csv') # replace this with llm annotated file path
all_annotators = pd.read_csv('.../human annotations/WMT_D_annotations.csv')
compare_human_llm(llm_df, all_annotators) # get agreement and correlations

