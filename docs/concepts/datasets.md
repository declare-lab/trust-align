# Datasets

The benchmark datasets used in this paper are adapted from [ALCE](https://arxiv.org/abs/2305.14627). The `question`, `answers`, and document fields (`title`, `text`, and `rec_score`) are sourced from the benchmark datasets released in [ALCE](https://arxiv.org/abs/2305.14627). In addition, we provide annotations for the `answers_found` field, which are specifically designed to support answer calibration in our metrics.

<table>
<caption>Table 1: Benchmark datasets, retrieval corpora, question types, and their corresponding examples. Adapted from <a href="https://arxiv.org/abs/2305.14627">ALCE</a>.</caption>
<tr>
  <th>Dataset</th>
  <th>Corpus (#passages)</th>
  <th>Question type</th>
  <th>Example</th>
</tr>
<tr>
  <td>ASQA</td>
  <td>Wikipedia (21M)</td>
  <td>Factoid (ambiguous)</td>
  <td>
    <b>Q:</b> When did the US break away from England?<br>
    <b>A:</b> The US declared independence on July 2, 1776 [1][2] ... The Treaty of Paris was later signed on September 3, 1783 [3].
  </td>
</tr>
<tr>
  <td>QAMPARI</td>
  <td>Wikipedia (21M)</td>
  <td>Factoid (list)</td>
  <td>
    <b>Q:</b> Which films have Gong Li as a member of their cast?<br>
    <b>A:</b> The Story of Qiu Ju [1], Farewell My Concubine [2], The Monkey King 2 [3], Mulan [3], Saturday Fiction [3] ...
  </td>
</tr>
<tr>
  <td>ELI5</td>
  <td>Sphere (899M)</td>
  <td>Why/How/What</td>
  <td>
    <b>Q:</b> How do student loans affect getting a mortgage?<br>
    <b>A:</b> Student loans can affect the debt-to-income ratio [1], which is a key factor in determining the amount that ... [2][3]
  </td>
</tr>
<tr>
  <td>ExpertQA</td>
  <td>Sphere (899M)</td>
  <td>Why/How/What/Open-ended/Summarization/Opinion/Hypothetical</td>
  <td>
    <b>Q:</b> Apple and Jimmy Iovine did their best to try and curb the onset of digitally pirated music. Is there any evidence that their efforts were successful?<br>
    <b>A:</b> Apple and Jimmy Iovine, ...,made efforts to curb the onset of digitally pirated music through the implementation of FairPlay DRM technology [1][2].
  </td>
</tr>
</table>

## ASQA

```json
// A list of dictionaries where each dictionary contains the information for one sample
[ ...
    {   // The question asked.
        "question": "Who has the highest goals in world football?", 
        
        // A list containing all correct (short) answers to the question, represented as arrays where each element contains variations of the answer. 
        "answers": [
            ["Daei", "Ali Daei"],                // Variations for Ali Daei
            ["Bican", "Josef Bican"],            // Variations for Josef Bican
            ["Sinclair", "Christine Sinclair"]   // Variations for Christine Sinclair
        ],
        
        // A list of 100 dictionaries where each dictionary contains one document.
        "docs": [
            {   
                // The title of the document being referenced.
                "title": "Argentina\u2013Brazil football rivalry",
                
                // A snippet of text from the document.
                "text": "\"Football Player of the Century\", ...",
                
                // A binary list where each element indicates if the respective answer was found in the document (1 for found, 0 for not found).
                "answers_found": [0, 0, 0],
                
                // A recall score calculated as the percentage of correct answers that the document entails.
                "rec_score": 0.0
            },
            ...
        ]
    },
...
]

```

## QAMPARI

```json
[ ...
  // A list of dictionaries where each dictionary contains the information for one sample
  {
    // The question asked.
    "question": "What manga was drawn by Ryoichi Ikegami?",

    // A list containing all correct (short) answers to the question, represented as arrays where each element contains variations of the answer.
    "answers": [["Heat"],["Mai, the Psychic Girl"],["Wounded Man"],["Sanctuary"],["Crying Freeman"],["Strain"]],
    
    // A list of 100 dictionaries where each dictionary contains one document.
    "docs": [
        {
          // The title of the document being referenced.
            "title": "Ryu Fujisaki",
            
            // A snippet of text from the document.
            "text": "Ryu Fujisaki He won prizes in 39th and the 40th Tezuka Awards. He made his professional ...",

            // A binary list where each element indicates if the respective answer was found in the document (1 for found, 0 for not found).
            "answers_found": [0,0,0,0,0,0],
            
            // A recall score calculated as the percentage of correct answers that the document entails.
            "rec_score": 0.0
        },
        ...
    ]
  }
  ...
]
            
```

## ELI5

```json
[...
  // A list of dictionaries where each dictionary contains the information for one sample
  {
    // The question asked.
    "question": "How are firms like snapchat, uber etc valued so highly while still not making a profit? ...",

    // A list containing all correct answers to the question. Generated using text-davinci-003 to break gold long form answer into three claims.
    "answers": [
        ["Firms like Snapchat and Uber need to establish their brand and amass users before introducing ads."],
        ["Introducing ads too early can deter potential users."],
        ["Uber is reinvesting a lot of money to make their service better."]
    ],

    // A list of 100 dictionaries where each dictionary contains one document.
    "docs": [
        {
          // The title of the document being referenced.
            "title": "Is Snapchat really worth $19 billion? - CSMonitor.com",

            // A snippet of text from the document.
            "text": "reporting that the Los Angeles-based company is ...",

            // A binary list where each element indicates if the respective answer was found in the document (1 for found, 0 for not found).
            "answers_found": [0,0,0],

            // A recall score calculated as the percentage of correct answers that the document entails.
            "rec_score": 0.0
        },
        ...
    ]
  }
  ...
]
```

## ExpertQA

```json
[...
  // A list of dictionaries where each dictionary contains the information for one sample
  {
    // The question asked.
    "question": "What are signs and study findings that would indicate follicular lymphoma has transformed to diffuse large B-cell lymphoma?",

    // A list containing all correct answers to the question. Generated using text-davinci-003 to break gold long form answer into three claims.
    "answers": [
        ["Rapid enlargement of lymph nodes can indicate a transformation of follicular lymphoma to diffuse large B-cell lymphoma."],
        ["B-symptoms such as fever, unintended weight loss, and drenching night sweats may suggest that follicular lymphoma has transformed."],
        ["A biopsy is the only reliable method to diagnose the transformation of follicular lymphoma to diffuse large B-cell lymphoma."]
    ],

    // A list of 100 dictionaries where each dictionary contains one document.
    "docs": [
        {
          // The title of the document being referenced.
            "title": "Follicular lymphoma",

            // A snippet of text from the document.
            "text": "Follicular lymphoma Follicular lymphoma is a type of blood cancer...",

            // A binary list where each element indicates if the respective answer was found in the document (1 for found, 0 for not found).
            "answers_found": [0,0,0],

            // A recall score calculated as the percentage of correct answers that the document entails.
            "rec_score": 0.0
        },
        ...
          ]
  }
  ...
]
```
