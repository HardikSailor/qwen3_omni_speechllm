import os
import torch
import transformers

from tqdm import tqdm

import random
import re

from openai import OpenAI
from multiprocessing import Pool


def llama3_70b_as_judge_one_sample(args):

    PROMPT_TEMPLATE, tokenizer, question, reference, prediction, transcript = args

    # Find all substrings enclosed in either single or double quotes
    extracted_quotes = re.findall(r'["](.*?)["]', prediction)

    if len(extracted_quotes) == 0:
        sample_rating_detail = {
            'question'              : question,
            'transcript'            : transcript,
            'reference'             : reference,
            'model_prediction'      : prediction,
            'judge_response'        : 'Model judge is skipped model prediction contains no direct quotes.',
            'quotation_score'       : 0.0,
            'groundedness_score'    : 0.0,
            'relevance_score'       : 0.0,
            'success'               : 1.0,
        }
        return sample_rating_detail

    evaluation_prompt = PROMPT_TEMPLATE.format(question=question, extracted_quotes=extracted_quotes, reference=reference, transcript=transcript)

    messages = [
        {"role": "user", "content": evaluation_prompt},
    ]

    templated_sample = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=False,
    )

    # Model
    port = os.environ.get('MY_VLLM_PORT_JUDGE', 5000)
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    completion = client.completions.create(
    model      = model,
    prompt     = templated_sample,
    max_tokens = 512,
    n          = 1,
        )
    
    output = completion.choices[0].text.strip()

    try:
        groundedness_score = float(re.search(r"Groundedness Score: \s*(\d)", output).group(1))
        relevance_score = float(re.search(r"Relevance Score: \s*(\d)", output).group(1))
        success = 1
    except:
        groundedness_score = 0.0
        relevance_score = 0.0
        success = 0

    sample_rating_detail = {
        'question'              : question,
        'transcript'            : transcript,
        'reference'             : reference,
        'model_prediction'      : prediction,
        'judge_response'        : output,
        'quotation_score'       : 1.0,
        'groundedness_score'    : groundedness_score,
        'relevance_score'       : relevance_score,
        'success'               : success,
    }

    return sample_rating_detail


def llama3_70b_as_judge_emotion_reasoning(model_path, input_data):
    """ Compute the score of the model on the given data."""

    # Prompt template
    PROMPT_TEMPLATE = """\
        [Ground Truth Emotion]
        {reference}

        [Ground Truth Transcript]
        {transcript}

        [Extracted Quotes from Model Prediction]
        {extracted_quotes}

        [Evaluation Task]
        Evaluate the extracted quotes using the following three criteria.

        **Groundedness Score**
        Assess whether the extracted quotes are grounded in the ground truth transcript.
        Scoring Guide:
        Score0: The quotes do not appear in the ground truth transcript and are not semantically aligned (i.e., hallucinated or generic).
        Score1: The quotes partially match the ground truth transcript. There may be loose paraphrasing or selective grounding.
        Score2: The quotes are clearly derived from the ground truth transcript, through direct quotes or faithful paraphrases.

        **Relevance Score**
        Assess whether the extractd quotes support the ground truth emotion label.
        Scoring Guide:
        Score0: The quotes are irrelevant or inconsistent with the ground truth emotion. They may even suggest a different emotion.
        Score1: The quotes are loosely related to the ground truth emotion but lack clarity, specificity, or completeness.
        Score2: The quotes clearly and directly support the ground truth emotion.

        Respond with the following structured format:

        Ground Truth Emotion: (string)
        Ground Truth Transcript: (string)
        Extracted Quotations from Model Prediction: (list of strings)
        Groundedness Score: (int)
        Relevance Score: (int)
        Explanation: (string - justify the assigned scores)
        """

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(os.environ.get("OMNI_JUDGE_TOKENIZER", model_path), device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Generation
    questions, references, predictions, transcripts = input_data

    num_processes = int(os.environ.get("OMNI_JUDGE_PROCS", 200))

    with Pool(processes=num_processes) as pool:
        all_details = list(
            tqdm(
                pool.imap(llama3_70b_as_judge_one_sample,
                    zip([PROMPT_TEMPLATE]*len(questions), [tokenizer]*len(questions), questions, references, predictions, transcripts)),
                total=len(questions),
                desc="Processing"
            )
        )

    all_quotation_scores    = [detail['quotation_score'] for detail in all_details]
    all_groundedness_scores = [detail['groundedness_score'] for detail in all_details if detail['quotation_score']==1]
    all_relevance_scores    = [detail['relevance_score'] for detail in all_details if detail['quotation_score']==1]

    avg_quotation_score     = sum(all_quotation_scores) / len(all_quotation_scores) * 100
    avg_groundedness_score  = sum(all_groundedness_scores) / len(all_groundedness_scores) * (100/2)
    avg_relevance_score     = sum(all_relevance_scores) / len(all_relevance_scores) * (100/2)
    success_rate            = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {
        'judge_quotation_score': avg_quotation_score,
        'judge_groundedness_score': avg_groundedness_score,
        'judge_relevance_score': avg_relevance_score,
        'success_rate': success_rate}

    return judge_results, all_details


def llama3_70b_as_judge_sentiment_reasoning(model_path, input_data):
    """ Compute the score of the model on the given data."""

    # Prompt template
    PROMPT_TEMPLATE = """\
        [Ground Truth Sentiment]
        {reference}

        [Ground Truth Transcript]
        {transcript}

        [Extracted Quotes from Model Prediction]
        {extracted_quotes}

        [Evaluation Task]
        Evaluate the extracted quotes using the following three criteria.

        **Groundedness Score**
        Assess whether the extracted quotes are grounded in the ground truth transcript.
        Scoring Guide:
        Score0: The quotes do not appear in the ground truth transcript and are not semantically aligned (i.e., hallucinated or generic).
        Score1: The quotes partially match the ground truth transcript. There may be loose paraphrasing or selective grounding.
        Score2: The quotes are clearly derived from the ground truth transcript, through direct quotes or faithful paraphrases.

        **Relevance Score**
        Assess whether the extractd quotes support the ground truth sentiment label.
        Scoring Guide:
        Score0: The quotes are irrelevant or inconsistent with the ground truth sentiment. They may even suggest a different sentiment.
        Score1: The quotes are loosely related to the ground truth sentiment but lack clarity, specificity, or completeness.
        Score2: The quotes clearly and directly support the ground truth sentiment.

        Respond with the following structured format:

        Ground Truth Sentiment: (string)
        Ground Truth Transcript: (string)
        Extracted Quotations from Model Prediction: (list of strings)
        Groundedness Score: (int)
        Relevance Score: (int)
        Explanation: (string - justify the assigned scores)
        """

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(os.environ.get("OMNI_JUDGE_TOKENIZER", model_path), device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Generation
    questions, references, predictions, transcripts = input_data

    num_processes = int(os.environ.get("OMNI_JUDGE_PROCS", 200))

    with Pool(processes=num_processes) as pool:
        all_details = list(
            tqdm(
                pool.imap(llama3_70b_as_judge_one_sample,
                    zip([PROMPT_TEMPLATE]*len(questions), [tokenizer]*len(questions), questions, references, predictions, transcripts)),
                total=len(questions),
                desc="Processing"
            )
        )

    all_quotation_scores    = [detail['quotation_score'] for detail in all_details]
    all_groundedness_scores = [detail['groundedness_score'] for detail in all_details if detail['quotation_score']==1]
    all_relevance_scores    = [detail['relevance_score'] for detail in all_details if detail['quotation_score']==1]

    avg_quotation_score     = sum(all_quotation_scores) / len(all_quotation_scores) * 100
    avg_groundedness_score  = sum(all_groundedness_scores) / len(all_groundedness_scores) * (100/2)
    avg_relevance_score     = sum(all_relevance_scores) / len(all_relevance_scores) * (100/2)
    success_rate            = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {
        'judge_quotation_score': avg_quotation_score,
        'judge_groundedness_score': avg_groundedness_score,
        'judge_relevance_score': avg_relevance_score,
        'success_rate': success_rate}

    return judge_results, all_details