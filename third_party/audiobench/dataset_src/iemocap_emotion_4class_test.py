import random
import logging
import re
from collections import defaultdict

er_instructions = [
    "How do you perceive the speaker's emotional state from their speech (anger, neutral, happiness, sad)?",
    "What emotions do you detect in the speaker's voice (anger, neutral, happiness, sad)?",
    "Can you identify the speaker's emotional state from their speech (anger, neutral, happiness, sad)?",
    "Based on their speech, how would you describe the speaker's emotions (anger, neutral, happiness, sad)?",
    "What emotional cues can you pick up from the speaker's speech (anger, neutral, happiness, sad)?",
    "How would you describe the emotions conveyed in the speaker's voice (anger, neutral, happiness, sad)?",
    "What do you think the speaker is feeling based on their speech (anger, neutral, happiness, sad)?",
    "Can you interpret the emotions in the speaker's speech (anger, neutral, happiness, sad)?",
    "How does the speaker's speech reflect their emotional state (anger, neutral, happiness, sad)?",
    "What is the emotional tone of the speaker's speech (anger, neutral, happiness, sad)?"
]


class iemocap_emotion_4class_test_dataset(object):

    def __init__(self, raw_data, number_of_samples):

        if number_of_samples != -1:
            raw_data = raw_data.shuffle(seed=42)
            raw_data = raw_data.select(range(number_of_samples))
        
        self.raw_data = raw_data
        self.prompt   = er_instructions
        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        input_data = []
        emotion_4class = ["anger", "neutral", "happiness", "sad"]
        for sample in self.raw_data:
            audio       = sample['context']
            instruction = random.choice(self.prompt)
            reference   = sample['answer']

            # only include samples with answers containing anger, neutral, happiness, sad
            if not any(emotion in reference.lower() for emotion in emotion_4class):
                continue

            input_data.append({
                                "audio"    : audio,
                                "text"     : instruction,
                                "answer"   : reference,
                                "task_type": "ER"
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
        emotion_4class = ["anger", "neutral", "happiness", "sad"]

        for item in data_with_model_predictions:

            question         = item["text"]
            answer           = item["answer"]
            model_prediction = item["model_prediction"]

            if metrics == "llama3_70b_judge_on_label_only":
                model_prediction = re.split(r'[,.]', model_prediction)[0].strip() + "."

            # only include samples with answers containing anger, neutral, happiness, sad
            if not any(emotion in answer.lower() for emotion in emotion_4class):
                continue

            questions.append(question)
            references.append(answer)
            predictions.append(model_prediction)

        if metrics == 'llama3_70b_judge':
            from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge_binary
            llama3_70b_judge_results, all_details = llama3_70b_as_judge_binary("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions])
            return {'llama3_70b_judge': llama3_70b_judge_results, 'details': all_details}

        if metrics == 'llama3_70b_judge_on_label_only':
            from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge_binary
            llama3_70b_judge_results, all_details = llama3_70b_as_judge_binary("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions])
            return {'llama3_70b_judge_on_label_only': llama3_70b_judge_results, 'details': all_details}

        if metrics == 'llama3_70b_judge_on_extracted_label':
            from dataset_src.eval_methods.eval_llama3_70b_classification import llama3_70b_as_judge_binary_iemocap_emotion as llama3_70b_as_judge_binary
            llama3_70b_judge_results, all_details = llama3_70b_as_judge_binary("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions])
            return {'llama3_70b_judge_on_extracted_label': llama3_70b_judge_results, 'details': all_details}

        # elif metrics == 'llama3_70b_judge_binary':
        #     from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge_binary
        #     llama3_70b_judge_binary_results, all_details = llama3_70b_as_judge_binary("meta-llama/Meta-Llama-3-70B-Instruct", [questions, references, predictions])
        #     return {'llama3_70b_judge_binary': llama3_70b_judge_binary_results, 'details': all_details}        

        elif metrics == 'llama3_8b_judge':
            from dataset_src.eval_methods.eval_llama3_8b import llama3_8b_as_judge
            llama3_8b_judge_results = llama3_8b_as_judge("../prepared_models/Meta-Llama-3-8B-Instruct-hf", [questions, references, predictions])
            return {'llama3_8b_judge': llama3_8b_judge_results}
        
        elif metrics == 'prometheus2_judge':
            from dataset_src.eval_methods.eval_prometheus2 import prometheus2_as_judge
            prometheus2_judge_results = prometheus2_as_judge("../prepared_models/prometheus-7b-v2.0", [questions, references, predictions])
            return {'prometheus2_judge': prometheus2_judge_results}
        
        elif metrics == 'gpt4o_judge':
            from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge
            gpt4o_judge_results, all_details = gpt4o_as_judge("", [questions, references, predictions])
            return {'gpt4o_judge': gpt4o_judge_results, 'details': all_details}
        
        elif metrics == 'gpt4o_judge_binary':
            from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
            gpt4o_judge_binary_results, all_details = gpt4o_as_judge_binary("", [questions, references, predictions])
            return {'gpt4o_judge_binary': gpt4o_judge_binary_results, 'details': all_details}
        
        else:
            raise ValueError("Invalid metrics: {}".format(metrics))


    def do_analysis(self, details):

        emotion_list = ["anger", "happiness", "neutral", "sad"]

        # Initialize accumulators
        scores = defaultdict(list)
        successes = defaultdict(list)

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
            scores[emotion].append(result['rate_score'])
            successes[emotion].append(result['success'])

        # Compute averages
        analysis_stats = {}
        for emotion in emotion_list:
            if scores[emotion]:
                avg_score = sum(scores[emotion]) / len(scores[emotion])
                avg_success = sum(successes[emotion]) / len(successes[emotion])
            else:
                avg_score = 0.0
                avg_success = 0.0
            analysis_stats[emotion] = {
                'judge_score': round(avg_score, 3) * 100,
                'success_success': round(avg_success, 3),
                'count': len(scores[emotion])
            }

        analysis_stats['unmatched'] = unmatched
        if unmatched:
            print(f"\n{len(unmatched)} results were not assigned to exactly one emotion.")

        return(analysis_stats)