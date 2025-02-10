from termcolor import colored
import dotenv
import sys
import dspy
import os

from tqdm.auto import tqdm

sys.path.append('../pipeline_v2/')
import main
dotenv.load_dotenv('../.env', override=True)

from utils import print_header

import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['mistral', 'gemini', 'llama', 'deepseek', 'claude', 'gpt4o'])
parser.add_argument('--num_trials', type=int, default=3)
parser.add_argument('--output_file', type=str, default=f'results_v2_{parser.parse_args().model}.pkl')
args = parser.parse_args()
model = args.model
num_trials = args.num_trials

def print_final_result(statement, verdict, confidence, reasoning, gold_verdict=None):
    print("\nFinal Fact-Check Result:")
    print_header(f"Statement: {colored(statement, 'white')}", level=1)
    print_header(f"Overall Verdict: {colored(verdict, 'green')}", level=1)
    print_header(f"Overall Confidence: {colored(str(confidence), 'yellow')}", level=1)
    print_header(f"Overall Reasoning: {colored(reasoning, 'cyan')}", level=1)
    if gold_verdict: print_header(f"Gold Verdict: {colored(gold_verdict, 'green')}", level=1)

### Load data
output_file = args.output_file
if os.path.exists(output_file):
    df = pd.read_pickle(output_file)
else: 
    df = pd.read_csv('../data/pilot_updated_v2.csv')

    # # Drop unneeded columns
    df.drop(columns=['Assignee', 'questions to verify the statement', 'Gold Label', 'factcheck_date'], inplace=True)

    # Reformat dates
    df['statement_date'] = pd.to_datetime(df['statement_date']).dt.strftime("%B %d, %Y")

# Set custom constants for whole pipeline
main.VERBOSE = False # Print intermediate results
main.NUM_SEARCH_RESULTS = 10
# main.VERDICTS=["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking"]

# Initialize DSPy
if args.model == 'gemini':
    lm = dspy.LM('gemini/gemini-1.5-flash', api_key=os.getenv('GOOGLE_GEMINI_API_KEY'), cache=False)
elif args.model == 'mistral':
    lm = dspy.LM('ollama_chat/mistral-custom', api_base='http://localhost:11434', api_key='', cache=False, temperature=0.3)
    # lm = dspy.LM('openrouter/mistralai/mistral-7b-instruct:free', api_key=os.getenv('OPENROUTER_API_KEY'), cache=False)
elif args.model == 'llama':
    lm = dspy.LM('ollama_chat/llama3.1:8b', api_base='http://localhost:11434', api_key='', cache=False)
elif args.model == 'deepseek':
    lm = dspy.LM('ollama_chat/deepseek-r1:7b-custom', api_base='http://localhost:11434', api_key='', cache=False, temperature=0.3)
    # lm = dspy.LM('openrouter/deepseek/deepseek-r1-distill-llama-70b:free', api_key=os.getenv('OPENROUTER_API_KEY'), cache=False)
elif args.model == 'claude':
    lm = dspy.LM('anthropic/claude-3-5-sonnet-20240620', api_key=os.getenv('ANTHROPIC_API_KEY'), cache=False)
elif args.model == 'gpt4o':
    lm = dspy.LM('openai/gpt-4o', api_key=os.getenv('OPENAI_API_KEY'), cache=False)
else:
    raise ValueError(f"Model {args.model} not supported")

dspy.settings.configure(lm=lm, temperature=0.3)

# Initialize pipeline and baseline models
pipeline = main.FactCheckPipeline(
    # search_provider=main.SearchProvider(provider="duckduckgo"),
    search_provider=main.SearchProvider(provider="serper", api_key=os.getenv('SERPER_API_KEY')),
    model_name=lm, 
    embedding_model=main.EMBEDDING_MODEL,
    retriever_k=5
)

class StatementFactCheckerSignature(dspy.Signature):
    f"""Fact check the given statement into one of the following verdicts: {", ".join(main.VERDICTS)}"""
    statement = dspy.InputField(desc="Statement to evaluate")
    verdict = dspy.OutputField(desc=f"Truthful classification of the statement into one of the following verdicts: {', '.join(main.VERDICTS)}")

class StatementFactChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.fact_check = dspy.ChainOfThought(StatementFactCheckerSignature)

    def forward(self, statement: str):
        result = self.fact_check(statement=statement)
        verdict = result["verdict"]
        reasoning = result["reasoning"]
        return verdict, reasoning
    
baseline = StatementFactChecker()

# Initialize results columns if they don't exist
for col in [f'{model}_pipeline_results', f'{model}_baseline_results']:
    if col not in df.columns: 
        df[col] = None
    df[col] = df[col].astype(object)

# Run experiments for each row
for index in tqdm(range(len(df))):
    # Get current results for both pipeline and baseline
    pipeline_results = df.loc[index, f'{model}_pipeline_results'] or []
    baseline_results = df.loc[index, f'{model}_baseline_results'] or []
    
    # Skip if both experiments have completed all trials
    if len(pipeline_results) == num_trials and len(baseline_results) == num_trials:
        print(f"Skipping row {index} - all trials completed")
        continue
        
    print(f"Running row {index}")
    print(f"Pipeline: {len(pipeline_results)}/{num_trials} trials completed")
    print(f"Baseline: {len(baseline_results)}/{num_trials} trials completed")
    
    # Get statement data
    statement = df.iloc[index]['statement']
    statement_originator = df.iloc[index]['statement_originator']
    statement_date = df.iloc[index]['statement_date']
    gold_verdict = df.iloc[index]['verdict']
    formatted_statement = f"On {statement_date}, {statement_originator} claimed: {statement}"
        
    # Run remaining trials for baseline
    for _ in tqdm(range(num_trials-len(baseline_results)), leave=False, desc="Baseline trials"):
        verdict, reasoning = baseline(formatted_statement)
        print("\n=== Baseline Result ===")
        print_final_result(statement, verdict, 'N/A', reasoning, gold_verdict)
        
        baseline_results.append({
            'verdict': verdict,
            'reasoning': reasoning
        })
        df.at[index, f'{model}_baseline_results'] = baseline_results
        df.to_pickle(output_file)

    # Run remaining trials for pipeline
    for _ in tqdm(range(num_trials-len(pipeline_results)), leave=False, desc="Pipeline trials"):
        pipeline.retriever.clear()
        verdict, confidence, reasoning, claims = pipeline.fact_check(
            statement=formatted_statement
        )
        print("\n=== Pipeline Result ===")
        print_final_result(statement, verdict, confidence, reasoning, gold_verdict)
        
        pipeline_results.append({
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning,
            'claims': claims
        })
        df.at[index, f'{model}_pipeline_results'] = pipeline_results
        df.to_pickle(output_file)