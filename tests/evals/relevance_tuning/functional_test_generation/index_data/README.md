### INDEX SCHEMAS

The index schemas here roughly reflect the corresponding schemas:
- `doc_index_schema.json`: the schema used for example doc index in Chorus
  - Adjustment: brand and color fields added
- `event_index_schema.json`: 
  - reflecting the `ubi_events` schema to allow same queries our tools use (e.g correct field types for
    aggregations), with some minor deletions
- `query_index_schema.json`:
  - same as `ubi_queries` schema 
