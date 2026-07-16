### TEST INDEX SETUP

The functional_test_generation folder contains the code for generating tests based on existing indices
(such as the chorus example index) and to generate new indices containing custom scenarios
for the agent to solve. This basically solves a few problems:
- its cumbersome to run analysis on every index change and update the tests. The provided code uses
  tools also used by the agent to derive expected values
- its cumbersome to look for suitable index entries / query cases that reflect a certain scenario
  we want the agent to detect and solve. The provided code generates indices with given properties
  such that test cases can be based on it.

When we speak of index here, we actually mean:
- a document index
- a ubi query index
- a ubi event index

We are splitting the test cases here by complexity:
- fast, more atomic requests, such as
  - top N queries
  - total nr of clicks on product N for query M
- slow, involving analysis, hypothesis forming and validation steps
  - more complex cases based on constructed and / or existing scenarios


### SETUP
- Run the main method in `src/index_scenarios.py`
  - generates index data within index_data subfolder, in folder named according to the created scenario (e.g scenario1)
- Index the data into their own test indices by running: ` ./scripts/create_all_indices.sh`
- then run test generation based on this data (and the normal chorus index) by running main method in `src/test_generation.py`
  - this generates csv files containing the generated tests, one per fast / slow type:
    - `functional_tests_fast.csv` (basic cases involving tool usage, but without more involved analysis cases)
    - `functional_tests_slow.csv` (cases involving analysis, hypothesis building and validation. Those are slow and can take well over 10 minutes)
- run the generated tests via 
  ```shell
  AWS_BEARER_TOKEN_BEDROCK=[AWS_BEARER_TOKEN_BEDROCK] promptfoo eval -c promptfooconfig_csv_[fast/slow].yaml --no-cache
  ```


### NOTES
- the tests combining hypothesis generation and testing those are expensive and can run a long time (e.g > 10 mins)
- quicker checks that can be answered simply by calling an index tool are fast. Thus it is advised to split those
  cases and further on adding a case rather run single tests than the full test suite



