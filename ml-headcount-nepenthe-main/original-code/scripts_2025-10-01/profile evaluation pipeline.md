## profile evaluation pipeline

1. Took all company profiles, looking at employee count we removed the biggest ones to stay below 100k people profiles.
2. Pulled a dataset of 85k profiles from bright data with the current company matching the companies from the first step.
3. Validated the following models with different tempeature 0 and their respective default temperature* (usually 1.0 or 0.7)
 against the rated CVs using two different prompts.
 Google: gemini-2.5-pro, gemini-2.5-flash and gemini-2.5-flash-lite
 OpenAI: gpt-5-2025-08-07, gpt-5-mini-2025-08-07, gpt-5-nano-2025-08-07
 Anthropic: opus-4-1-20250805, sonnet-4-20250514

 *: Except for gpt-5 models, which do not support a temperature parameter.
 
4. Selected gemini-2.5-flash, sonnet-4-20250514, and gpt-5-mini-2025-08-07 as they provided robust results while being
  relatively fast and cheap. 

5. We ran the evaluation of the entire 85k profiles using the batching APIs of the three models.
  Sonnet 4 while being the most expensive of the three, but also completing all batches in an impressive 2.5 minutes.

6. We matched and grouped the results with the respective companies from the input dataset, which cleaned out any 
  hallucinated profiles, and summed up the results by company.