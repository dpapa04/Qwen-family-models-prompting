# Qwen-family-models-prompting

## Introduction

The aim of this project is to categorize responses given during political interviews into three classes:
Clear Reply, Ambivalent, and Clear Non-Reply, using instruct-tuned language models from the
Qwen family. Here we examine the models used and do an error analysis that led to the performance
of the models.

## Models and Methodology

The models used from the Qwen 3.5 family are:
– Qwen3.5-0.8B
– Qwen3.5-2B
– Qwen3.5-4B

The three strategies that where used with each of the models above are:
– Zero-Shot: A direct instruction to classify the text.
– Few-Shot: Has an example that have been taken from the training data.
– Chain-of-Thought (CoT): Encourages the model to “Think step-by-step” before providing a
label.

For each of the strategies above we make the specialized prompt providing the examples and CoT
description to show the step by step way of thinking.

## Experement Results

These are the results of the f1-scores from each model and strategy used to evaluate the test set of
the dataset:
Model Size Zero-Shot Few-Shot Chain-of-Thought
Qwen-0.8B 0.55 0.57 0.57
Qwen-2B 0.12 0.30 0.60
Qwen-4B 0.10 0.43 0.10

## Observations and Findings:

– Unusual Scaling : Normally in language modeling large models always perform better than
small ones, this does not seem to be the case in this project. Here the Qwen-0.8B model showed
incredible stability through the use of all the strategies.
– CoT Effectiveness: Qwen-2B experienced very high sensitivity to prompting, it performed
significantly well after being prompted with Chain-of-thought strategy. The performance of
Qwen-2B increased from very poor performance at 0.13 (Zero-Shot) to good at 0.60 (CoT).
– 4B Problems: Qwen-4B model had significantly poor performance at both zero-shot and CoT
prompts. This is because the use of larger models may have lead to failure to follow specific
formats, resulting in settling on one particular class. Few-shot strategy proved to be very
stabilizing for the 4B model (0.44).
– Consistency of 0.8B: The model 0.8B is more safe compared to others. As it is relatively
small in size, it might contain lesser distractions, which makes it consistent when following
basic classification instructions, despite its slightly low efficiency.

## Error Analysis

– Class Difficulty: The Clear Non-Reply class is the hardest to predict. All the models strugled
with this class probably due to the nature of such answers and how similar they sometimes are
to Clear Reply. Both have very straight forward answers with clear grammar and vocabulary
which the model seems to take as a truth indicator.
– Linguistic patterns: From the examples we show in the cells, we detect that when the answer
of the politician starts with positive words like ‘Yes’ or ‘Thank you’ the model tends to think
that the answer belongs in the Clear Reply category.
– Ideal Prompting: The ideal prompting would contain an example from each category and
especially for the CoT strategy, would contain a separete way of thinking for each and every
one of the Labels.

## Changes:

– After running the model for the first time with a batch size of 8, the full extend of the available
memory was quickly used. In order to counter that, the first measure against it was to add del
model and torch.cuda.empty_cache() which helped until the 4B parameter model which
ran out of memory. Then the batch size was reduced to 4 which helped with the memory
management but made the model run slower.
– Firstly, the prompt for the CoT strategy was written by hand, something that appeared to be
hurtful as the model was not able to get a real example that would prepare it for the actual test
sentences. In order to change that, a question-answer pair was chosen from the training data,
along with the prediction and label column as an chain of thought example. This helped the
model make more accurate thought patterns, which eventually helped with the final prediction
labels .
– Added a "fallback" to the "Ambivalent" label. While this prevented code crashes, it likely
increased our False Positive rate for that class.

## Conclusion

The use of the Chain-of-Thought strategy with a 2B model was more effective than utilizing a
4B model without the suitable prompt. This shows the danger of believing that increasing the
parameter size will enhance classification performance of a model automatically.
