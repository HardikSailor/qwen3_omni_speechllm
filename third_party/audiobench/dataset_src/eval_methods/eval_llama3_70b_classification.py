import os
import torch
import transformers

from tqdm import tqdm

import random

from openai import OpenAI
from multiprocessing import Pool


def llama3_70b_as_judge_one_sample(args):

    PROMPT_TEMPLATE, tokenizer, question, reference, prediction = args

    evaluation_prompt = PROMPT_TEMPLATE.format(question=question, prediction=prediction, reference=reference)

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
        rate_score = float(output.split()[-1])
        success = 1
    except:
        rate_score = 0.0
        success = 0

    sample_rating_detail = {
        'question'        : question,
        'reference'       : reference,
        'model_prediction': prediction,
        'judge_response'  : output,
        'rate_score'      : rate_score,
        'success'         : success,
    }

    return sample_rating_detail


def llama3_70b_as_judge_binary_emotion(model_path, input_data, emotion_options):
    """ Compute the score of the model on the given data."""

    # Prompt template
    PROMPT_TEMPLATE = """\
        [Reference Answer]
        {reference}

        [Model Answer]
        {prediction}

        [Question]
        {question}

        [Task]
        First, extract the part of sentence containing the predicted emotion {emotion_options} from the model's answer.
        Next, rate the extracted output based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details.
        Criteria: Assess if the extracted output mirrors the reference in terms of content, accuracy, and relevance. Please give a score of 0 or 1.
        Score0: The extracted output is refusing to give concrete results, providing something like 'cannot decide'.
        Score0: The extracted output is wrong, providing incorrect or irrelevant information compared to the reference.
        Score1: The extracted output is correct, capturing or covering the meaning from the reference.

        Your response should be formatted as follows:
        Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the extracted output. "The reference answer is [XXX], while the extracted output is [YYY]. I think ...")
        Rating: (int)"""

    formatted_emotion_options = f"({', '.join(emotion_options)})"
    PROMPT_TEMPLATE = PROMPT_TEMPLATE.replace("{emotion_options}", formatted_emotion_options)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(os.environ.get("OMNI_JUDGE_TOKENIZER", model_path), device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Generation
    questions, references, predictions = input_data

    num_processes = int(os.environ.get("OMNI_JUDGE_PROCS", 200))

    with Pool(processes=num_processes) as pool:
        all_details = list(
            tqdm(
                pool.imap(llama3_70b_as_judge_one_sample,
                    zip([PROMPT_TEMPLATE]*len(questions), [tokenizer]*len(questions), questions, references, predictions)),
                total=len(questions),
                desc="Processing"
            )
        )

    all_scores   = [detail['rate_score'] for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores) * 100
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate}

    return judge_results, all_details


def llama3_70b_as_judge_binary_iemocap_emotion(model_path, input_data):
    """ Compute the score of the model on the given data."""

    emotion_options = ['anger', 'disgust', 'excited', 'fear', 'frustration', 'happiness', 'neutral', 'sad', 'surprise', 'other']
    return llama3_70b_as_judge_binary_emotion(model_path, input_data, emotion_options)


def llama3_70b_as_judge_binary_meld_emotion(model_path, input_data):
    """ Compute the score of the model on the given data."""

    emotion_options = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    return llama3_70b_as_judge_binary_emotion(model_path, input_data, emotion_options)


def llama3_70b_as_judge_binary_m3ed_emotion(model_path, input_data):
    """ Compute the score of the model on the given data."""

    emotion_options = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    return llama3_70b_as_judge_binary_emotion(model_path, input_data, emotion_options)


def llama3_70b_as_judge_binary_ytb_eval_human_emotion(model_path, input_data):
    """ Compute the score of the model on the given data."""

    emotion_options = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    return llama3_70b_as_judge_binary_emotion(model_path, input_data, emotion_options)


def llama3_70b_as_judge_binary_sentiment(model_path, input_data):
    """ Compute the score of the model on the given data."""

    # Prompt template
    PROMPT_TEMPLATE = """\
        [Reference Answer]
        {reference}

        [Model Answer]
        {prediction}

        [Question]
        {question}

        [Task]
        First, extract the part of sentence containing the predicted sentiment (neutral, positive, negative) from the model's answer.
        Next, rate the extracted output based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details.
        Criteria: Assess if the extracted output mirrors the reference in terms of content, accuracy, and relevance. Please give a score of 0 or 1.
        Score0: The extracted output is refusing to give concrete results, providing something like 'cannot decide'.
        Score0: The extracted output is wrong, providing incorrect or irrelevant information compared to the reference.
        Score1: The extracted output is correct, capturing or covering the meaning from the reference.

        Your response should be formatted as follows:
        Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the extracted output. "The reference answer is [XXX], while the extracted output is [YYY]. I think ...")
        Rating: (int)"""

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(os.environ.get("OMNI_JUDGE_TOKENIZER", model_path), device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Generation
    questions, references, predictions = input_data

    num_processes = int(os.environ.get("OMNI_JUDGE_PROCS", 200))

    with Pool(processes=num_processes) as pool:
        all_details = list(
            tqdm(
                pool.imap(llama3_70b_as_judge_one_sample,
                    zip([PROMPT_TEMPLATE]*len(questions), [tokenizer]*len(questions), questions, references, predictions)),
                total=len(questions),
                desc="Processing"
            )
        )

    all_scores   = [detail['rate_score'] for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores) * 100
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate}

    return judge_results, all_details
