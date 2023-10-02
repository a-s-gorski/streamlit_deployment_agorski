import os

import onnxruntime
import spacy
from typing import List, Optional, Dict, Tuple
from nltk import TweetTokenizer
from morfeusz2 import Morfeusz
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
import concurrent
import numpy as np
from numpy.typing import NDArray
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from scipy.special import softmax

morfeusz = Morfeusz()
tw = TweetTokenizer()
processor = spacy.load('pl_core_news_sm')
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")


def process_token(token) -> Optional[str]:
    if token.ent_type:
        return token.ent_type_
    if not token.is_space and not token.is_punct:
        analysis = morfeusz.analyse(token.text)
        lemma = analysis[0][2][1].split(":")[0]
        return lemma
    else:
        return None


def process_line(line: List[str]):
    return list(map(process_token, line))


def process_lines(lines: List[str]):
    lines_tw = list(map(tw.tokenize, lines))
    lines_tw = list(map(" ".join, lines_tw))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        lines_tokenized = list(tqdm(executor.map(processor, lines_tw), total=len(lines_tw)))
    lines_tokenized_processed = list(map(process_line, lines_tokenized))
    lines_tokenized_processed = [[item for item in inner_list if item is not None and item != '@'] for inner_list in
                                 lines_tokenized_processed]
    lines_tokenized_processed = list(map(" ".join, lines_tokenized_processed))
    return lines_tokenized_processed


def process_tokenize_input(lines: List[str], preprocess: bool) -> Dict[str, NDArray]:
    if preprocess:
        lines = process_lines(lines)
    model_input = tokenizer.batch_encode_plus(lines, padding='longest', max_length=256, truncation=True,
                                              return_tensors='pt', add_special_tokens=True)
    input_ids = model_input['input_ids'].numpy()
    attention_mask = model_input['attention_mask'].numpy()
    token_type_ids = model_input['token_type_ids'].numpy()
    ort_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
    return ort_inputs


def calculate_confidences(ort_outputs: List[NDArray]) -> List[float]:
    smoothed_outputs = softmax(ort_outputs, axis=-1)
    confidences = list(map(lambda x: max(x) / sum(x) * 100, smoothed_outputs))
    return confidences


def calculate_output_classes(ort_outputs: List[NDArray]) -> List[int]:
    return np.argmax(ort_outputs, axis=-1).tolist()


def get_ort_session() -> InferenceSession:
    model_path = os.getenv("MODEL_PATH")
    EP_list = ['CPUExecutionProvider', ]
    return onnxruntime.InferenceSession(model_path, providers=EP_list)


def run_inference(session: InferenceSession, ort_inputs: Dict[str, NDArray]) -> Tuple[
    List[int], List[float], List[List[float]]]:
    ort_outputs = session.run(None, ort_inputs)[0].tolist()
    confidences = calculate_confidences(ort_outputs)
    output_classes = calculate_output_classes(ort_outputs)

    return output_classes, confidences, ort_outputs
