import json
import random
import uuid
from collections.abc import Callable, Collection
from dataclasses import field
from datetime import UTC, datetime
from typing import Literal
import numpy as np
from pydantic import BaseModel


class Entries:
    # NOTE: the nr of events used in the example index is higher, here
    # just showing a relevant subset as per index generation
    # by scenarios (and utilized by the tools for test case generation
    # in test_generation.py). Extend where needed.
    type ValidEventAction = Literal["click", "add_to_cart", "purchase", "impression"]

    class Doc(BaseModel):
        asin: str = field(
            default_factory=lambda: str(uuid.uuid4()).replace("-", "")[:10]
        )
        bullet_points: str | None = None
        description: str | None = None
        color: str | None = None
        category: str
        title: str
        price: float
        brand: str

        def ndjson_index_step(self, index: str = "test_ecommerce"):
            """
            Generate ndjson format, two rows per doc index cmd
            """
            doc_json = self.model_dump()
            doc_json["category"] = [
                x.strip() for x in doc_json["category"].split(">>>")
            ]
            return "\n".join(
                [
                    json.dumps({"index": {"_index": index, "_id": self.asin}}),
                    json.dumps(doc_json),
                ]
            )

    # NOTE: user_query gives the actual query, while query_id is a single instance of the query
    # being used, not referring to a single instance / representation of query
    class UbiQuery(BaseModel):
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        application: str = "default_app"
        query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        client_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        user_query: str
        timestamp: str = field(
            default_factory=lambda: datetime.now(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )

        def ndjson_ubindex_event_step(self, index: str = "test_ubi_queries"):
            return "\n".join(
                [
                    json.dumps({"index": {"_index": index, "_id": self.id}}),
                    self.model_dump_json(),
                ]
            )

    class UbiEventDoc(BaseModel):
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        application: str = "default_app"
        query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        client_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        action_name: "Entries.ValidEventAction"
        user_query: str
        doc_id: str
        position: int | None
        timestamp: str = field(
            default_factory=lambda: datetime.now(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )

        def get_event_attributes(self) -> dict[str, str]:
            return {
                "event_attributes": {
                    "object": {"object_id": self.doc_id, "object_id_field": "asin"},
                    "position": {"ordinal": self.position},
                }
            }

        def ndjson_ubindex_event_step(self, index: str = "test_ubi_events"):
            return "\n".join(
                [
                    json.dumps({"index": {"_index": index, "_id": self.id}}),
                    json.dumps(self.model_dump() | self.get_event_attributes()),
                ]
            )


class Sampling:
    class FieldSamplingStrategy(BaseModel):
        def sample(self) -> any:
            pass

    class ConstantSamplingStrategy(FieldSamplingStrategy):
        value: float | str | int

        def sample(self) -> any:
            return self.value

    class StringFromTokenSamplingStrategy(FieldSamplingStrategy):
        tokens: list[str] | set[str] | tuple[str]
        min_num: int = 1
        max_num: int | None = None
        add_tokens: list[str] | tuple[str] = tuple([])

        def model_post_init(self, __context) -> None:
            self.tokens = tuple(set(self.tokens))
            if not self.max_num:
                self.max_num = len(self.tokens)

        def sample(self) -> any:
            num_draws = random.choice(range(self.min_num, self.max_num + 1))
            return " ".join(
                random.sample(self.tokens, num_draws) + list(self.add_tokens)
            )

    class FromToFloatSamplingStrategy(FieldSamplingStrategy):
        min_value: float = 10.0
        max_value: float = 100.0
        step_size: float = 1.0

        def sample(self) -> any:
            return float(
                random.choice(
                    np.arange(
                        self.min_value, self.max_value + self.step_size, self.step_size
                    )
                )
            )


class Scenario(BaseModel):
    docs: list[Entries.Doc] | tuple[Entries.Doc, ...]
    queries: list[Entries.UbiQuery] | tuple[Entries.UbiQuery, ...]
    events: list[Entries.UbiEventDoc] | tuple[Entries.UbiEventDoc, ...]

    def add_scenario(self, other: "Scenario"):
        return Scenario(
            docs=tuple(list(self.docs) + list(other.docs)),
            queries=tuple(list(self.queries) + list(other.queries)),
            events=tuple(list(self.events) + list(other.events)),
        )

    def doc_index_ndjson(self, index: str = "test_doc_index"):
        return "\n".join([x.ndjson_index_step(index=index) for x in self.docs]) + "\n"

    def queries_index_ndjson(self, index: str = "test_query_index"):
        return (
            "\n".join([x.ndjson_ubindex_event_step(index=index) for x in self.queries])
            + "\n"
        )

    def events_index_ndjson(self, index: str = "test_event_index"):
        return (
            "\n".join([x.ndjson_ubindex_event_step(index=index) for x in self.events])
            + "\n"
        )

    @staticmethod
    def write_index_file(path: str, content_callable: Callable[[], str]):
        content = content_callable()
        with open(path, "w") as f:
            f.write(content)


def compose_events_by_doc_filters(
    user_query: str,
    docs: Collection[Entries.Doc],
    filter_rule: Callable[[Entries.Doc], bool],
    num_events: int,
    action_name: Entries.ValidEventAction,
    query_id_pool: Collection[str],
):
    """Compose per-doc events, assigning each event a query_id from
    ``query_id_pool`` round-robin. This lets one search instance (query_id) carry
    multiple events and guarantees every event references a search that exists in
    the query table (so searches_with_clicks <= search_volume by construction)."""
    filtered_docs = [x for x in docs if filter_rule(x)]
    pool = list(query_id_pool)
    all_events = []
    counter = 0
    for doc in filtered_docs:
        for _ in range(num_events):
            if not pool:
                raise ValueError(
                    "query_id_pool is empty but events were requested; ensure "
                    "num_queries (and num_clicking_searches for clicks) is large enough."
                )
            all_events.append(
                Entries.UbiEventDoc(
                    query_id=pool[counter % len(pool)],
                    action_name=action_name,
                    user_query=user_query,
                    doc_id=doc.asin,
                    position=None,
                )
            )
            counter += 1
    return all_events


def sample_doc(field_to_sampling: dict[str, Sampling.FieldSamplingStrategy]):
    return {x: y.sample() for x, y in field_to_sampling.items()}


class ScenarioGeneration:
    @staticmethod
    def doc_sampling_mapping(title: str):
        return {
            "category": Sampling.ConstantSamplingStrategy(value="some category"),
            "title": Sampling.ConstantSamplingStrategy(value=title),
            "description": Sampling.ConstantSamplingStrategy(value="some description"),
            "brand": Sampling.ConstantSamplingStrategy(value="other brand"),
            "price": Sampling.FromToFloatSamplingStrategy(
                min_value=10.0, max_value=100.0, step_size=1.0
            ),
        }

    type DocField = Literal[
        "title", "category", "bullet_points", "description", "brand", "color"
    ]

    @staticmethod
    def create_fix_ranking_via_brand_boost_scenario(
        query: str,
        holdout_token: str,
        boost_field: DocField = "brand",  # field where the token to be boosted is in the high performers
        other_field: DocField = "description",  # field where the token to be boosted is in the low performers
        event_name: Literal["click", "add_to_cart"] = "click",
        num_queries: int = 300,
        num_clicking_searches: int = 0,
        num_low_performer: int = 10,
        num_high_performer: int = 10,
        num_clicks_low_performer: int = 2,
        num_clicks_high_performer: int = 20,
        num_impressions_low_performer: int = 20,
        num_impressions_high_performer: int = 20,
        low_performer_doc_prefix: str = "low",
        high_performer_doc_prefix: str = "high",
    ):
        """
        NOTE: query should have at least two tokens cause the logic below simply
        uses one of the tokens (given by holdout_token) occuring in the brand field as indicator whether it is
        a high perormer or low performer doc! holdout_token needs to be one of the tokens
        in the query (split by whitespace).

        Simplest fixable-by-boost scenario:
        - query with two tokens, one in title, the other in description for lower performers,
          and in 'brand' for better performers
        - boosting on brand field will improve ranking and thus should be
          within the agent suggestions on search optimization.

        Such as in the query:
        ```
        Help me improve the query samsung. Here is my current DSL query:
        POST ecommerce/_search
        {
          "query": {
            "multi_match": {
              "query": "shoe super",
              "fields": [
                "asin",
                "title",
                "category",
                "bullet_points",
                "description",
                "brand",
                "color"
              ]
            }
          }
        }
        ```
        """
        query_tokens: list[str] = [x.strip() for x in query.split(" ")]
        query_without_holdout_token = " ".join(
            [x for x in query_tokens if x != holdout_token]
        )

        base_field_sampling_mapping = ScenarioGeneration.doc_sampling_mapping(
            title=query_without_holdout_token
        )
        low_performer_doc_field_sampling_mapping = base_field_sampling_mapping | {
            other_field: Sampling.ConstantSamplingStrategy(value=holdout_token)
        }
        high_performer_doc_field_sampling_mapping = base_field_sampling_mapping | {
            boost_field: Sampling.ConstantSamplingStrategy(value=holdout_token),
        }
        low_performer_docs = [
            Entries.Doc(
                **(
                    sample_doc(low_performer_doc_field_sampling_mapping)
                    | {"asin": f"{low_performer_doc_prefix}{idx}"}
                )
            )
            for idx in range(num_low_performer)
        ]
        high_performer_docs = [
            Entries.Doc(
                **(
                    sample_doc(high_performer_doc_field_sampling_mapping)
                    | {"asin": f"{high_performer_doc_prefix}{idx}"}
                )
            )
            for idx in range(num_high_performer)
        ]
        # One search instance per query_id. Both tables reference these ids, so a
        # query_id carries multiple events (impressions + clicks). Clicks land on
        # a subset of searches, so CTR = num_clicking_searches / num_queries.
        search_ids = [str(uuid.uuid4()) for _ in range(num_queries)]
        clicking_ids = search_ids[:num_clicking_searches]

        low_performer_click_events = compose_events_by_doc_filters(
            user_query=query,
            docs=low_performer_docs,
            filter_rule=lambda x: True,
            num_events=num_clicks_low_performer,
            action_name=event_name,
            query_id_pool=clicking_ids,
        )
        low_performer_impression_events = compose_events_by_doc_filters(
            user_query=query,
            docs=low_performer_docs,
            filter_rule=lambda x: True,
            num_events=num_impressions_low_performer,
            action_name="impression",
            query_id_pool=search_ids,
        )
        high_performer_click_events = compose_events_by_doc_filters(
            user_query=query,
            docs=high_performer_docs,
            filter_rule=lambda x: True,
            num_events=num_clicks_high_performer,
            action_name=event_name,
            query_id_pool=clicking_ids,
        )
        high_performer_impression_events = compose_events_by_doc_filters(
            user_query=query,
            docs=high_performer_docs,
            filter_rule=lambda x: True,
            num_events=num_impressions_high_performer,
            action_name="impression",
            query_id_pool=search_ids,
        )

        # one query-table row per search instance (same query_ids as the events)
        query_events = tuple(
            Entries.UbiQuery(query_id=qid, user_query=query) for qid in search_ids
        )
        return Scenario(
            docs=list(low_performer_docs) + list(high_performer_docs),
            queries=query_events,
            events=(
                list(low_performer_click_events)
                + list(high_performer_click_events)
                + list(low_performer_impression_events)
                + list(high_performer_impression_events)
            ),
        )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Index Scenario Args")
    parser.add_argument(
        "--scenario_name", required=True, help="the name of the scenarios"
    )
    args = parser.parse_args()
    script_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
    scenario_path = f"{script_path}/../../index_data/{args.scenario_name}"

    shoe_scenario = ScenarioGeneration.create_fix_ranking_via_brand_boost_scenario(
        query="shoe super",
        holdout_token="super",
        boost_field="brand",
        other_field="description",
        event_name="click",
        num_queries=300,
        num_clicking_searches=106,  # ~35.33% CTR (reflects current 220/620=35.48%)
        num_low_performer=10,
        num_high_performer=10,
        num_clicks_low_performer=2,
        num_clicks_high_performer=20,
        num_impressions_low_performer=20,
        num_impressions_high_performer=20,
        low_performer_doc_prefix="shoe_low",
        high_performer_doc_prefix="shoe_high",
    )
    color_trousers_scenario = (
        ScenarioGeneration.create_fix_ranking_via_brand_boost_scenario(
            query="sweet trousers green",
            holdout_token="green",
            boost_field="color",
            other_field="description",
            event_name="click",
            num_queries=400,
            num_clicking_searches=71,  # ~17.75% CTR (reflects current 540/3040=17.76%)
            num_low_performer=40,
            num_high_performer=10,
            num_clicks_low_performer=1,
            num_clicks_high_performer=50,
            num_impressions_low_performer=50,
            num_impressions_high_performer=50,
            low_performer_doc_prefix="trousers_green_low",
            high_performer_doc_prefix="trousers_green_high",
        )
    )
    misspelling_no_click_no_impression_1 = (
        ScenarioGeneration.create_fix_ranking_via_brand_boost_scenario(
            query="ninhendo swithc",
            holdout_token="switch",
            boost_field="color",
            other_field="description",
            event_name="click",
            num_queries=200,
            num_clicking_searches=0,  # misspelling: searched but no clicks/impressions
            num_low_performer=0,
            num_high_performer=0,
            num_clicks_low_performer=0,
            num_clicks_high_performer=0,
            num_impressions_low_performer=0,
            num_impressions_high_performer=0,
            low_performer_doc_prefix="misspelling_low",
            high_performer_doc_prefix="misspelling_high",
        )
    )

    corrected_spelling_no_click_no_impression_1 = (
        ScenarioGeneration.create_fix_ranking_via_brand_boost_scenario(
            query="nintendo switch",
            holdout_token="switch",
            boost_field="title",
            other_field="description",
            event_name="click",
            num_queries=200,
            num_clicking_searches=57,  # ~28.5% CTR (reflects current 600/2100=28.57%)
            num_low_performer=10,
            num_high_performer=5,
            num_clicks_low_performer=10,
            num_clicks_high_performer=100,
            num_impressions_low_performer=100,
            num_impressions_high_performer=100,
            low_performer_doc_prefix="misspelling_low",
            high_performer_doc_prefix="misspelling_high",
        )
    )

    combined_scenario = (
        shoe_scenario.add_scenario(color_trousers_scenario)
        .add_scenario(misspelling_no_click_no_impression_1)
        .add_scenario(corrected_spelling_no_click_no_impression_1)
    )

    if not os.path.exists(scenario_path):
        print(f"Creating output dir: {scenario_path}")
        os.makedirs(scenario_path)

    with open(f"{scenario_path}/test_doc_index.ndjson", "w") as f:
        f.write(combined_scenario.doc_index_ndjson())
    with open(f"{scenario_path}/test_query_index.ndjson", "w") as f:
        f.write(combined_scenario.queries_index_ndjson())
    with open(f"{scenario_path}/test_event_index.ndjson", "w") as f:
        f.write(combined_scenario.events_index_ndjson())
