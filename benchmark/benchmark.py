from termcolor import colored
import dotenv
import sys
import dspy
import os

from tqdm.auto import tqdm

sys.path.append('../pipeline_v2/')
import main 
dotenv.load_dotenv('../.env')

from utils import print_header

import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['mistral', 'gemini', 'llama', 'deepseek'])
parser.add_argument('--num_trials', type=int, default=3)
args = parser.parse_args()

def print_final_result(statement, verdict, confidence, reasoning, gold_verdict=None):
    print("\nFinal Fact-Check Result:")
    print_header(f"Statement: {colored(statement, 'white')}", level=1)
    print_header(f"Overall Verdict: {colored(verdict, 'green')}", level=1)
    print_header(f"Overall Confidence: {colored(str(confidence), 'yellow')}", level=1)
    print_header(f"Overall Reasoning: {colored(reasoning, 'cyan')}", level=1)
    if gold_verdict: print_header(f"Gold Verdict: {colored(gold_verdict, 'green')}", level=1)

### Load data
if os.path.exists('results_v2.pkl'):
    df = pd.read_pickle('results_v2.pkl')
else: 
    df = pd.read_csv('../data/[FINAL] Pilot - Pilot Claims copy.csv')

    # Drop unneeded columns
    df.drop(columns=['Assignee', 'questions to verify the statement', 'Gold Label', 'GPT-4-Label', 'Claude3-Sonnet-Label', 'mistral_fs_results', 'mistral_verdicts', 'mistral_fs_label', 'GPT3.5(Claude problem)'], inplace=True)

    # Reformat dates
    df['statement_date'] = pd.to_datetime(df['statement_date']).dt.strftime("%B %d, %Y")

# Set custom constants for whole pipeline
main.VERBOSE = False # Print intermediate results
# main.VERDICTS=["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking"]

# Initialize DSPy
if args.model == 'gemini':
    lm = dspy.LM('gemini/gemini-1.5-pro', api_key=os.getenv('GOOGLE_GEMINI_API_KEY'), cache=False)
elif args.model == 'mistral':
    lm = dspy.LM('ollama_chat/mistral', api_base='http://localhost:11434', api_key='', cache=False)
elif args.model == 'llama':
    lm = dspy.LM('ollama_chat/llama3.1:8b', api_base='http://localhost:11434', api_key='', cache=False)
elif args.model == 'deepseek':
    lm = dspy.LM('ollama_chat/deepseek-r1:7b', api_base='http://localhost:11434', api_key='', cache=False)
else:
    raise ValueError(f"Model {args.model} not supported")

dspy.settings.configure(lm=lm)

pipeline = main.FactCheckPipeline(
    search_provider=main.SearchProvider(provider="duckduckgo"),
    model_name=lm,
    embedding_model=main.EMBEDDING_MODEL,
    retriever_k=2
)

model = args.model
num_trials = args.num_trials

# If column doesn't exist, create it
if f'{model}_results' not in df.columns: df[f'{model}_results'] = None
df[f'{model}_results'] = df[f'{model}_results'].astype(object)

for index in tqdm(range(len(df))):
    # If results already exist, skip if num_trials is reached
    if df.loc[index, f'{model}_results'] is not None: 
        if len(df.loc[index, f'{model}_results']) == num_trials:
            continue
        else:
            results = df.loc[index, f'{model}_results']
    else: 
        results = []

    for trial_i in tqdm(range(num_trials-len(results)), leave=False):
        statement = df.iloc[index]['statement']
        statement_originator = df.iloc[index]['statement_originator']
        statement_date = df.iloc[index]['statement_date']
        gold_verdict = df.iloc[index]['verdict']

        verdict, confidence, reasoning, claims = pipeline.fact_check(
            statement=statement, 
            context=f"Statement Originator: {statement_originator}, Date Claim Was Made: {statement_date}"
        )   
        print_final_result(statement, verdict, confidence, reasoning, gold_verdict)
        results.append({
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning,
            'claims': claims
        })
        df.at[index, f'{model}_results'] = results

        df.to_pickle('results_v2.pkl')
