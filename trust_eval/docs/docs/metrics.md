# List of available metrics

Sample output:

```javascript
{   // refusal response: "I apologize, but I couldn't find an answer..."
    
    // basic statistics
    "answered_ratio": 50.0, // Ratio of (# answered qns / total # qns)
    "answered_num": 5, // # of qns where response is not refusal response
    "answerable_num": 7, // # of qns that ground truth answerable, given the documents
    "overlapped_num": 5, // # of qns that are both answered and answerable
    "regular_length": 46.6, // Average length of all responses
    "answered_length": 28.0, // Average length of non-refusal responses

    // Refusal groundedness metrics
    
    // # qns where (model refused to respond & is ground truth unanswerable) / # qns is ground truth unanswerable
    "refusal_rec": 100.0, 

    // # qns where (model refused to respond & is ground truth unanswerable) / # qns where model refused to respond
    "refusal_prec": 60.0,

    // F1 of refusal_rec and refusal_prec
    "refusal_f1": 75.0,

    // # qns where (model respond & is ground truth answerable) / # qns is ground truth answerable
    "answerable_rec": 71.42857142857143,

    // # qns where (model respond & is ground truth answerable) / # qns where model responded
    "answerable_prec": 100.0,

    // F1 of answerable_rec and answerable_prec
    "answerable_f1": 83.33333333333333,

    // Avg of refusal_rec and answerable_rec
    "macro_avg": 85.71428571428572,

    // Avg of refusal_f1 and answerable_f1
    "macro_f1": 79.16666666666666,

    // Response correctness metrics

    // Regardless of response type (refusal or answered), check if ground truth claim is in the response. 
    "regular_str_em": 41.666666666666664,

    // Only for qns with answered responses, check if ground truth claim is in the response. 
    "answered_str_em": 66.66666666666666,

    // Calculate EM for all qns that are answered and answerable, avg by # of answered questions (EM_alpha)
    "calib_answered_str_em": 100.0,

    // Calculate EM for all qns that are answered and answerable, avg by # of answerable questions (EM_beta)
    "calib_answerable_str_em": 71.42857142857143,

    // F1 of calib_answered_str_em and calib_answerable_str_em
    "calib_str_em_f1": 83.33333333333333,

    // EM score of qns that are answered and ground truth unanswerable, indicating use of parametric knowledge
    "parametric_str_em": 0.0,

    // Citation quality metrics

    // (Avg across all qns) Does the set of citations support statement si? 
    "regular_citation_rec": 28.333333333333332,

    // (Avg across all qns) Any redundant citations? (1) Does citation ci,j fully support statement si? (2) Is the set of citations without ci,j insufficient to support statement si? 
    "regular_citation_prec": 35.0,

    // F1 of regular_citation_rec and regular_citation_prec
    "regular_citation_f1": 31.315789473684212,

    // (Avg across answered qns)
    "answered_citation_rec": 50.0,

    // (Avg across answered qns)
    "answered_citation_prec": 60.0,

    // F1 answered_citation_rec and answered_citation_prec
    "answered_citation_f1": 54.54545454545455,

    // Avg (macro_f1, calib_str_em_f1, answered_citation_f1)
    "trust_score": 72.34848484848486
}
```
