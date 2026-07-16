### Promptfoo Getting Started
NOTE: when starting up the ag ui server, your working directory needs to be in the src folder
of the opensearch-agent-server project, otherwise strands lib fails due to tools folder
not found, which it expects in the working directory from which the server is started
(startup example for port 8002 `uvicorn server.ag_ui_app:app --host 0.0.0.0 --port 8002`).


Short descriptions to directly run evals. See more context below.

- install promptfoo (global install cmd): `npm install -g promptfoo`
- install `@aws-sdk/client-bedrock-runtime` via: `npm install @aws-sdk/client-bedrock-runtime`
- see .env in project root for promptfoo-related properties and fill in
- a script that combines index entry generation, indexing test generation and test running
  can be found within `tests/evals/scripts/evaluate_agents.sh` where test suites to run can be 
  passed, and depending on the selected suites different agents will be tested. Initially
  this only contains the tests for the relevancy agent.
  The below steps are only needed if you need to call them separately for some reason.
- in the tests_promptfoo folder you find a script folder to run all relevant steps:
  - `generate_index_data.sh`: generates synthetic index data for programmatic test cases. 
    - events
    - queries
    - documents
  - `create_all_indices.sh`: clears test indices and fills them with synthetic index data as generated in 
    `generate_index_data.sh`. 
    Related indices:
    - `test_event_index`
    - `test_query_index`
    - `test_doc_index`
  - `index_data.sh`: used by `create_all_indices.sh` to index data, dont need to call directly
  - `generate_test_cases.sh`: generates test cases based on available indices. Assumes indices to be available:
    - Example WANDS index as provided by chorus project
    - indices as created by `generate_index_data.sh` and indexed by `create_all_indices`.
  - `run_eval.sh`: runs existing test suites on the current system as configured in .env file in root folder
  - `run_redteam_generate.sh`: based on the redteam config in `promptfooconfig_redteam.yaml`,
    generate test cases
  - `upload_last_results.sh`: upload last generated results to configured promptfoo eval server (if available)


Thus an example flow of scripts would contain the following calls (in the given sequence):
 
Used to (re-)generate index and test config data to run evals (generated files already pushed
and suitable if you use chorus setup).
- `./scripts/generate_index_data.sh` (by default writes to scenario1 subfolder)
- `./scripts/create_all_indices.sh -s [scenario_name, e.g scenario1]` (use scenario_name as corresponding to the scenario name used in previous script)
- `./scripts/generate_test_cases.sh`: configures test cases basd on available index data (assuming chorus WAND example index and the synthetically generated one (see above))

You can start from here, since configurations generated in above steps
are already contained in the pushed state:
- `./scripts/run_eval.sh -k functional -t fast` for functional tests 
  - set optional `-n` option if cached results shall be ignored to rerun tests
  - note that the valid values for param `-k` are `functional`and `redteam`
  - the valid values for param -t are the corresponding subfolder names in `tests_promptfoo/test_cases/functional`
- `./scripts/run_eval.sh -k redteam -t small_engl` for redteam tests 
  - set optional `-n` option if cached results shall be ignored to rerun tests
  - note that the valid values for param `-k` are `functional`and `redteam`
  - the valid values for param -t are the corresponding subfolder names in `tests_promptfoo/test_cases/redteam`
- `./scripts/tun_redteam_generate.sh -t test`
  - the value for param `-t` is the subfolder within `tests_promptfoo/test_cases/redteam` to which tests shall be generated
  - the redteam config used to generate tests can be found at `tests_promptfoo/promptfooconfig_redteam.yaml` 
  - the location of the used promptfooconfig can be changed with param `-c`
- `upload_last_results.sh`: uploads last generated test results to the eval server (if any)
  - see script for env vars that need to be set
- after eval run is complete, can view results via `promptfoo view`


### Sharing Data With Self-Hosted Eval Instance
Link: `https://www.promptfoo.dev/docs/usage/self-hosting/`
Two options:
- permanent config in promptfooconfig.yaml:
  """
  sharing:
    apiBaseUrl: http://your-server-address:3000
    appBaseUrl: http://your-server-address:3000
  """
- env vars: 
  """
  export PROMPTFOO_REMOTE_API_BASE_URL=http://your-server-address:3000
  export PROMPTFOO_REMOTE_APP_BASE_URL=http://your-server-address:3000
  """
  - here you will still need to call `promptfoo share` or combine eval with share as in `promptfoo eval --share`
- passing env vars with the cmd:
  - `PROMPTFOO_REMOTE_API_BASE_URL=http://your-server-address:3000 PROMPTFOO_REMOTE_APP_BASE_URL=http://your-server-address:3000 promptfoo share`


### Notes on threadId and runId in test settings
- in multi-turn tests, make sure to keep the same threadId between the turns
- for single-turn tests, even if threadId changes, make sure to not set the runId, as this will cache responses and leave those tests out for which the result is already available


### On Result Persistence
If you use a credential reference in the configs, even if it is just
a template reference to env.[key-env-var-name], single result files fill contain the credential / key in clear text. 
Thus result folders were added to `.gitignore` settings.
Note that we handle keys only via set env vars, not as any type of reference in the
config files themselves. This also leads to the clear text value not appearing in the
generated configs / results.


### Caution When Deploying A Promptfoo Eval Server
Do not publicly expose the server with own provider keys (e.g OpenAI) as currently its possible to create evaluations 
with configured providers, so anyone could configure this and run their own evals, which can be expensive.


### HTTP Api Reference: 
- https://www.promptfoo.dev/docs/api-reference


### Example Overview Of Results (Eval Server)
![Eval Multiturn](./images/example_multiturn.png)
![Redteam Eval Small](./images/example_redteam_small_engl.png)
![Redteam Eval Medium](./images/example_redteam_medium_engl.png)


### Attack Generation
- `https://www.promptfoo.dev/docs/red-team/configuration/#how-attacks-are-generated`
- by default, promptfoo uses the api key for the configured provider key for attack generation and grading
- IF no key set, promptfoo proxies requests to their own API for generation and
  grading. The evaluation of target model is always performed locally.
- `redteam.provider` config controls both attack generation and grading (!!)
- Can enforce 100 % local generation by setting `PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION` env var to true.
  - this also disables promotfoos remote service optimized for generating adversarial inputs.
- Can also pass provider on the cmd line: `promptfoo redteam generate --provider openai:chat:gpt-5-mini`


### Data Sharing
- even if custom LLM provider is configured, you will see that some calls are made to 
  the promptfoo api. Details see here: `https://github.com/promptfoo/promptfoo/issues/5808`
  - details known to be send:
    - Instructions/prompts
    - Conversation history (messages)
    - User email (from getUserEmail())
    - Promptfoo version
    - System purpose description

Excerpt from above github issue:
```text
To clarify, Promptfoo is 100% local for LLM evaluations, and prompts always remain local.

Promptfoo does require remote generation for a subset of red team plugins, listed here. 
Any plugin with the globe icon requires remote inference. It is possible to self-host the inference endpoint as part of 
our enterprise offering. The "system purpose description" is used by the remote server to generate attacks. 
The target's prompt and other target configuration details are never sent.

As you mentioned, you can disable remote generation using export PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION=true. 
This would prevent you from generating attacks for a subset of plugins that rely on remote inference.
```

- Hosting urself is an option, but for the inference server this is limited to the enterprise version
- Keys do not seem to leave local though
- fully local only when combining
  - a local provider (e.g ollama): `redteam.provider: ollama:chat:llama3.2`
  - disable remote generation: `PROMPTFOO_DISABLE_REMOTE_GENERATION`
- even when LLM provider is configured and attacks are available (e.g readteam.yaml), the eval call still calls the
  promptfoo api and sends the agent response. Seems to be only avoidable with options
  - fully local eval call (see above config)
  - alternative: enterprise deployment of own server
- recommendation by the team (`https://www.promptfoo.dev/docs/red-team/troubleshooting/grading-results/`):
  - use remote llm for generation of attacks
  - grading works well locally as well


### Non-redteam test cases
 - answer-relevance (functional eval): https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/answer-relevance/
 - llm-rubric (functional eval): https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/llm-rubric
 - pi (functional eval): https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/pi/
 - classifier (tone and so on): https://www.promptfoo.dev/docs/configuration/expected-outputs/classifier/
 - select-best (if wanting to compare several different providers): https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/select-best/
 - moderation (ensure safe outputs and compliance with usage policies): https://www.promptfoo.dev/docs/configuration/expected-outputs/moderation/
 - context-based (context-recall, context-relevance, context-faithfulness) (might be useful to check that on multi-step chat the context is enriched the right way via MCP and such)
  - https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/context-recall/
  - https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/context-relevance/
  - https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/context-faithfulness/ 
 - conversation-relevance (ensure responses remain relevant throughout a conversation): https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/conversation-relevance/
 - see example for multi-turn-conversation here: https://github.com/promptfoo/promptfoo/blob/main/examples/multiple-turn-conversation/promptfooconfig.yaml


