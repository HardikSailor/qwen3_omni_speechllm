"""
Evaluates the reasoning given for its alignment with the transcript.
"""

import random
import logging
import re
from collections import defaultdict

er_instructions = [
    "What sentiment do you sense in the speaker's voice (neutral, positive, negative)?",
    "Can you determine the speaker's sentiment from their speech (neutral, positive, negative)?",
    "How would you describe the speaker's sentiment based on their speech (neutral, positive, negative)?",
    "What sentiment signals can you hear in the speaker's speech (neutral, positive, negative)?",
    "How would you interpret the sentiment expressed in the speaker's voice (neutral, positive, negative)?",
    "What sentiment do you think the speaker is conveying through their speech (neutral, positive, negative)?",
    "Can you recognize the sentiment in the speaker's speech (neutral, positive, negative)?",
    "How does the speaker's speech indicate their sentiment (neutral, positive, negative)?",
    "What sentiment tone do you hear in the speaker's speech (neutral, positive, negative)?",
    "What sentiment is conveyed through the speaker's voice (neutral, positive, negative)?"
]


class meld_sentiment_reasoning_test_dataset(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        self.prompt   = er_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        for sample in self.raw_data:
            audio       = sample['context']['audio']
            transcript  = sample['context']['text']
            instruction = random.choice(self.prompt)
            reference   = sample['answer']['text']
            input_data.append({
                                "audio"     : audio,
                                "transcript": transcript,
                                "text"      : instruction,
                                "answer"    : reference,
                                "task_type" : "ER_Reasoning"
                                })

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data


    def format_model_predictions(self, input_data, model_predictions):

        data_with_model_predictions = []
        for sample in input_data:
            new_sample = sample.copy()
            del new_sample["audio"]
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)
        return data_with_model_predictions


    def compute_score(self, data_with_model_predictions, metrics=None):
        
        questions   = []
        references  = []
        predictions = []
        transcripts = []

        for item in data_with_model_predictions:
        
            question         = item["text"]
            answer           = item["answer"]
            model_prediction = item["model_prediction"]
            transcript       = item["transcript"]

            questions.append(question)
            references.append(answer)
            predictions.append(model_prediction)
            transcripts.append(transcript)

        if metrics == 'llama3_70b_judge':
            from dataset_src.eval_methods.eval_llama3_70b_reasoning import llama3_70b_as_judge_sentiment_reasoning
            llama3_70b_judge_results, all_details = llama3_70b_as_judge_sentiment_reasoning("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions, transcripts])
            return {'llama3_70b_judge': llama3_70b_judge_results, 'details': all_details}

        else:
            raise ValueError("Invalid metrics: {}".format(metrics))

    def do_analysis(self, details):

        emotion_list = ["positive", "negative", "neutral"]

        # Initialize accumulators
        quotation_scores    = defaultdict(list)
        groundedness_scores = defaultdict(list)
        relevance_score     = defaultdict(list)
        successes           = defaultdict(list)

        # Track unmatched references
        unmatched = []

        # Process results
        for idx, result in enumerate(details):
            reference = result['reference'].lower()
            matched_emotions = set()

            for keyword in emotion_list:
                if re.search(r'\b' + re.escape(keyword) + r'\b', reference):
                    matched_emotions.add(keyword)

            if len(matched_emotions) != 1:
                print(f"[Warning] Result {idx} matched {len(matched_emotions)} emotion categories: {matched_emotions}")
                unmatched.append(result['reference'])
                continue  # Skip or handle ambiguous cases differently

            emotion = matched_emotions.pop()
            quotation_scores[emotion].append(result['quotation_score'])
            groundedness_score[emotion].append(result['groundedness_score'])
            relevance_score[emotion].append(result['relevance_score'])
            successes[emotion].append(result['success'])

        # Compute averages
        analysis_stats = {}
        for emotion in emotion_list:
            if quotation_scores[emotion]:
                avg_quotation_score = sum(quotation_scores[emotion]) / len(quotation_scores[emotion])
                avg_groundedness_score = sum(groundedness_scores[emotion]) / len(groundedness_scores[emotion])
                avg_relevance_score = sum(relevance_scores[emotion]) / len(relevance_scores[emotion])
                avg_success = sum(successes[emotion]) / len(successes[emotion])
            else:
                avg_quotation_score = 0.0
                avg_groundedness_score = 0.0
                avg_relevance_score = 0.0
                avg_success = 0.0
            analysis_stats[emotion] = {
                'judge_quotation_score': round(avg_quotation_score, 3) * 100,
                'judge_groundedness_score': round(avg_groundedness_score, 3) * 100,
                'judge_relevance_score': round(avg_relevance_score, 3) * 100,
                'success_rate': round(avg_success, 3),
                'count': len(scores[emotion])
            }

        analysis_stats['unmatched'] = unmatched
        if unmatched:
            print(f"\n{len(unmatched)} results were not assigned to exactly one emotion.")

        return(analysis_stats)